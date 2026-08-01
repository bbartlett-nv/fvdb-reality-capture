# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import logging
import os
from typing import Literal

import cv2
import numpy as np
import torch
import tqdm
from fvdb.types import NumericMaxRank1, to_VecNf
from scipy.spatial import ConvexHull

from fvdb_reality_capture.sfm_scene import SfmCache, SfmPosedImageMetadata, SfmScene
from fvdb_reality_capture.sfm_scene.scene_attribute import CROP_MASK_BBOX_ATTRIBUTE, PerImageValueAttribute

from .base_transform import BaseTransform, transform

_MASK_BBOX_MANIFEST_VERSION = 1
_MASK_BBOX_MANIFEST_VERSION_KEY = "crop_mask_bbox_manifest_version"
_MASK_BBOX_IMAGE_IDS_KEY = "crop_mask_bbox_image_ids"
_MAX_RASTER_WORKSPACE_BYTES = 64 * 1024 * 1024


def _mask_bbox_xyxy_count(mask: np.ndarray) -> np.ndarray:
    """Return ``[xmin, ymin, xmax, ymax, count]`` for pixels that are valid under dataset mask semantics."""
    if mask.ndim == 3:
        mask = mask[..., 0]
    elif mask.ndim != 2:
        raise ValueError(f"Unsupported mask shape: {mask.shape}. Must have 2D or 3D shape.")

    if mask.dtype == np.bool_:
        valid = mask
    else:
        is_normalized = mask.size > 0 and np.max(mask) <= 1
        if np.issubdtype(mask.dtype, np.floating):
            # Normalized floating-point masks use conventional probability semantics.
            threshold = 0.5 if is_normalized else 127
        else:
            # Binary integer masks use 0/1, while byte-range masks use 0/255.
            threshold = 0 if is_normalized else 127
        valid = mask > threshold
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        return np.zeros((5,), dtype=np.int64)

    valid_rows = np.flatnonzero(np.any(valid, axis=1))
    valid_columns = np.flatnonzero(np.any(valid, axis=0))
    return np.array(
        [valid_columns[0], valid_rows[0], valid_columns[-1] + 1, valid_rows[-1] + 1, valid_count],
        dtype=np.int64,
    )


def _read_mask(mask_path: str) -> np.ndarray:
    if mask_path.strip().endswith(".npy"):
        return np.load(mask_path)
    if mask_path.strip().endswith((".png", ".jpg", ".jpeg")):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask {mask_path}")
        return mask
    raise ValueError(f"Unsupported mask file format: {mask_path}")


def _rasterize_convex_hull_mask(convex_hull: ConvexHull, image_height: int, image_width: int) -> np.ndarray:
    """Rasterize a 2D convex hull with the existing integer-pixel, closed-half-space semantics.

    Processing only the hull's clipped image-space bounds in bounded row chunks avoids the full-image
    ``[H, W, 2]`` coordinate grid and ``[H, W, num_edges]`` distance tensors used previously.
    ``cv2.fillConvexPoly`` is deliberately not used here because its sub-pixel edge coverage differs from
    the existing closed-half-space test by boundary pixels.
    """
    inside_mask = np.zeros((image_height, image_width), dtype=bool)
    hull_vertices = convex_hull.points[convex_hull.vertices]

    min_x = max(0, int(np.ceil(np.min(hull_vertices[:, 0]))))
    max_x = min(image_width, int(np.floor(np.max(hull_vertices[:, 0]))) + 1)
    min_y = max(0, int(np.ceil(np.min(hull_vertices[:, 1]))))
    max_y = min(image_height, int(np.floor(np.max(hull_vertices[:, 1]))) + 1)
    if min_x >= max_x or min_y >= max_y:
        return inside_mask

    hull_normals = convex_hull.equations[:, :-1]
    hull_offsets = convex_hull.equations[:, -1]
    bounded_width = max_x - min_x
    bytes_per_pixel = np.dtype(np.int64).itemsize * 2 + np.dtype(np.float64).itemsize * len(hull_offsets) + 1
    rows_per_chunk = max(1, _MAX_RASTER_WORKSPACE_BYTES // max(1, bounded_width * bytes_per_pixel))

    pixel_u = np.arange(min_x, max_x)
    for row_start in range(min_y, max_y, rows_per_chunk):
        row_stop = min(max_y, row_start + rows_per_chunk)
        pixel_v = np.arange(row_start, row_stop)
        grid_u, grid_v = np.meshgrid(pixel_u, pixel_v, indexing="xy")
        pixel_coords = np.stack([grid_u, grid_v], axis=-1)
        signed_distances = pixel_coords @ hull_normals.T + hull_offsets[np.newaxis, np.newaxis, :]
        inside_mask[row_start:row_stop, min_x:max_x] = np.all(signed_distances <= 0.0, axis=-1)

    return inside_mask


def _get_cached_mask_bboxes(transform_data: dict, image_ids: np.ndarray) -> np.ndarray | None:
    if transform_data.get(_MASK_BBOX_MANIFEST_VERSION_KEY) != _MASK_BBOX_MANIFEST_VERSION:
        return None

    cached_image_ids = np.asarray(transform_data.get(_MASK_BBOX_IMAGE_IDS_KEY, []), dtype=np.int64)
    cached_bboxes = np.asarray(transform_data.get(CROP_MASK_BBOX_ATTRIBUTE, []), dtype=np.int64)
    if cached_image_ids.shape != image_ids.shape or not np.array_equal(cached_image_ids, image_ids):
        return None
    if cached_bboxes.shape != (len(image_ids), 5):
        return None
    return cached_bboxes


def _write_transform_manifest(
    output_cache: SfmCache,
    transformation_matrix: np.ndarray,
    image_ids: np.ndarray,
    mask_bboxes: np.ndarray,
) -> None:
    output_cache.write_file(
        "transform",
        {
            "transform": transformation_matrix,
            _MASK_BBOX_MANIFEST_VERSION_KEY: _MASK_BBOX_MANIFEST_VERSION,
            _MASK_BBOX_IMAGE_IDS_KEY: image_ids,
            CROP_MASK_BBOX_ATTRIBUTE: mask_bboxes,
        },
        data_type="pt",
    )


def _crop_scene_to_bbox(
    input_scene: SfmScene,
    transform_name: str,
    composite_with_existing_masks: bool,
    mask_format: str,
    bbox: np.ndarray,
    logger: logging.Logger,
):
    if bbox.shape != (6,):
        raise ValueError("Bounding box must be a 1D array of shape (6,)")

    output_cache_prefix = f"{transform_name}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}_{bbox[4]}_{bbox[5]}_{mask_format}_{composite_with_existing_masks}"
    output_cache_prefix = output_cache_prefix.replace(" ", "_")  # Ensure no spaces in the cache prefix
    output_cache_prefix = output_cache_prefix.replace(".", "_")  # Ensure no dots in the cache prefix
    output_cache_prefix = output_cache_prefix.replace("-", "neg")  # Ensure no dashes in the cache prefix

    input_cache: SfmCache = input_scene.cache

    output_cache = input_cache.make_folder(
        output_cache_prefix,
        description=f"Image masks ({mask_format}) for cropping to bounding box {bbox}",
    )

    # Create a mask over all the points which are inside the bounding box
    points_mask = np.logical_and.reduce(
        [
            input_scene.points[:, 0] > bbox[0],
            input_scene.points[:, 0] < bbox[3],
            input_scene.points[:, 1] > bbox[1],
            input_scene.points[:, 1] < bbox[4],
            input_scene.points[:, 2] > bbox[2],
            input_scene.points[:, 2] < bbox[5],
        ]
    )

    # Mask the scene using the points mask
    masked_scene = input_scene.filter_points(points_mask)

    # How many zeros to pad the image index in the mask file names
    num_zeropad = len(str(len(masked_scene.images))) + 2
    image_ids = np.asarray([image.image_id for image in masked_scene.images], dtype=np.int64)

    new_image_metadata = []
    cached_mask_paths: list[str] = []
    transform_data: dict = {}

    regenerate_cache = False
    if output_cache.num_files != len(masked_scene.images) + 1:
        if output_cache.num_files == 0:
            logger.info("No masks found in the cache for cropping.")
        else:
            logger.info(
                f"Inconsistent number of masks for images. Expected {len(masked_scene.images)}, found {output_cache.num_files}. "
                f"Clearing cache and regenerating masks."
            )
        output_cache.clear_current_folder()
        regenerate_cache = True
    if output_cache.has_file("transform"):
        _, loaded_transform_data = output_cache.read_file("transform")
        if not isinstance(loaded_transform_data, dict):
            logger.info("Transform metadata does not match expected format. Clearing cache and regenerating transform.")
            output_cache.clear_current_folder()
            regenerate_cache = True
        else:
            transform_data = loaded_transform_data
            cached_transform: np.ndarray | None = transform_data.get("transform", None)
            if cached_transform is None:
                logger.info("Transform metadata does not match expected format. No 'transform' key in cached file.")
                output_cache.clear_current_folder()
                regenerate_cache = True
            elif not isinstance(cached_transform, np.ndarray) or cached_transform.shape != (4, 4):
                logger.info(
                    "Transform metadata does not match expected format. Expected 'transform'."
                    "Clearing the cache and regenerating transform."
                )
                output_cache.clear_current_folder()
                regenerate_cache = True
            elif not np.allclose(cached_transform, input_scene.transformation_matrix):
                logger.info(
                    "Cached transform does not match input scene transform. "
                    "Clearing the cache and regenerating transform."
                )
                output_cache.clear_current_folder()
                regenerate_cache = True
    else:
        logger.info("No transform found in cache, regenerating.")
        output_cache.clear_current_folder()
        regenerate_cache = True

    for image_meta in masked_scene.images:
        if regenerate_cache:
            break
        image_cache_filename = f"mask_{image_meta.image_id:0{num_zeropad}}"
        if not output_cache.has_file(image_cache_filename):
            logger.info(
                f"Mask for image {image_meta.image_id} not found in cache. Clearing cache and regenerating masks."
            )
            output_cache.clear_current_folder()
            regenerate_cache = True
            break

        key_meta = output_cache.get_file_metadata(image_cache_filename)
        if key_meta.get("data_type", "") != mask_format:
            logger.info(
                f"Output cache masks metadata does not match expected format. Expected '{mask_format}'."
                f"Clearing the cache and regenerating masks."
            )
            output_cache.clear_current_folder()
            regenerate_cache = True
            break
        mask_path = str(key_meta["path"])
        cached_mask_paths.append(mask_path)
        new_image_metadata.append(
            SfmPosedImageMetadata(
                world_to_camera_matrix=image_meta.world_to_camera_matrix,
                camera_to_world_matrix=image_meta.camera_to_world_matrix,
                camera_metadata=image_meta.camera_metadata,
                camera_id=image_meta.camera_id,
                image_id=image_meta.image_id,
                image_path=image_meta.image_path,
                mask_path=mask_path,
                point_indices=image_meta.point_indices,
            )
        )

    mask_bboxes = None
    if not regenerate_cache:
        mask_bboxes = _get_cached_mask_bboxes(transform_data, image_ids)
        if mask_bboxes is None:
            logger.info("Upgrading cached crop masks with per-image bounding-box metadata.")
            mask_bboxes = np.asarray(
                [_mask_bbox_xyxy_count(_read_mask(mask_path)) for mask_path in cached_mask_paths], dtype=np.int64
            ).reshape((-1, 5))
            _write_transform_manifest(
                output_cache,
                input_scene.transformation_matrix,
                image_ids,
                mask_bboxes,
            )

    if regenerate_cache:
        logger.info("Computing image masks for cropping and saving to cache.")
        new_image_metadata = []
        mask_bbox_rows: list[np.ndarray] = []

        min_x, min_y, min_z, max_x, max_y, max_z = bbox

        # (8, 4)-shaped array representing the corners of the bounding cube containing the input points
        # in homogeneous coordinates
        cube_bounds_world_space_homogeneous = np.array(
            [
                [min_x, min_y, min_z, 1.0],
                [min_x, min_y, max_z, 1.0],
                [min_x, max_y, min_z, 1.0],
                [min_x, max_y, max_z, 1.0],
                [max_x, min_y, min_z, 1.0],
                [max_x, min_y, max_z, 1.0],
                [max_x, max_y, min_z, 1.0],
                [max_x, max_y, max_z, 1.0],
            ]
        )

        for image_meta in tqdm.tqdm(masked_scene.images, unit="imgs", desc="Computing image masks for cropping"):
            cam_meta = image_meta.camera_metadata

            # Transform the cube corners to camera space
            cube_bounds_cam_space = image_meta.world_to_camera_matrix @ cube_bounds_world_space_homogeneous.T  # [4, 8]
            # Divide out the homogeneous coordinate -> [3, 8]
            cube_bounds_cam_space = cube_bounds_cam_space[:3, :] / cube_bounds_cam_space[-1, :]

            # Project the camera-space cube corners into image space [3, 3] * [8, 3] - > [8, 2]
            cube_bounds_pixel_space = cam_meta.projection_matrix @ cube_bounds_cam_space  # [3, 8]
            # Divide out the homogeneous coordinate and transpose -> [8, 2]
            cube_bounds_pixel_space = (cube_bounds_pixel_space[:2, :] / cube_bounds_pixel_space[2, :]).T

            # Compute and rasterize the pixel-space convex hull of the cube corners.
            convex_hull = ConvexHull(cube_bounds_pixel_space)
            image_width = image_meta.camera_metadata.width
            image_height = image_meta.camera_metadata.height
            inside_mask = _rasterize_convex_hull_mask(convex_hull, image_height, image_width)

            # If the mask already exists, load it and composite this one into it
            mask_to_save = inside_mask.astype(np.uint8) * 255  # Convert to uint8 mask
            if os.path.exists(image_meta.mask_path) and composite_with_existing_masks:
                existing_mask = _read_mask(image_meta.mask_path)
                if existing_mask.ndim == 3:
                    # Ensure the mask is 3D to match the input mask
                    inside_mask = inside_mask[..., np.newaxis]
                elif existing_mask.ndim != 2:
                    raise ValueError(f"Unsupported mask shape: {existing_mask.shape}. Must have 2D or 3D shape.")

                if existing_mask.shape[:2] != inside_mask.shape[:2]:
                    raise ValueError(
                        f"Existing mask shape {existing_mask.shape[:2]} does not match computed mask shape {inside_mask.shape[:2]}."
                    )
                mask_to_save = existing_mask * inside_mask

            cache_file_meta = output_cache.write_file(
                name=f"mask_{image_meta.image_id:0{num_zeropad}}",
                data=mask_to_save,
                data_type=mask_format,
            )
            if mask_format == "jpg":
                # JPEG is lossy, so metadata must describe the mask that downstream code will decode.
                mask_bbox_rows.append(_mask_bbox_xyxy_count(_read_mask(str(cache_file_meta["path"]))))
            else:
                mask_bbox_rows.append(_mask_bbox_xyxy_count(mask_to_save))

            new_image_metadata.append(
                SfmPosedImageMetadata(
                    world_to_camera_matrix=image_meta.world_to_camera_matrix,
                    camera_to_world_matrix=image_meta.camera_to_world_matrix,
                    camera_metadata=image_meta.camera_metadata,
                    camera_id=image_meta.camera_id,
                    image_id=image_meta.image_id,
                    image_path=image_meta.image_path,
                    mask_path=str(cache_file_meta["path"]),
                    point_indices=image_meta.point_indices,
                )
            )

        mask_bboxes = np.asarray(mask_bbox_rows, dtype=np.int64).reshape((-1, 5))
        _write_transform_manifest(
            output_cache,
            input_scene.transformation_matrix,
            image_ids,
            mask_bboxes,
        )

    new_attrs = {}
    for attr_name, attr in masked_scene.attributes.items():
        new_attrs[attr_name] = attr.on_crop_scene(
            attr_name=attr_name,
            bbox=bbox,
            output_cache=output_cache,
        )

    if mask_bboxes is None:
        raise RuntimeError("Crop mask bounding-box metadata was not initialized")
    new_attrs[CROP_MASK_BBOX_ATTRIBUTE] = PerImageValueAttribute(
        [bbox_row.copy() for bbox_row in mask_bboxes]
    )

    output_scene = masked_scene.replace(
        images=new_image_metadata,
        scene_bbox=bbox,
        transformation_matrix=input_scene.transformation_matrix,
        cache=output_cache,
        attributes=new_attrs,
    )

    return output_scene


@transform
class CropScene(BaseTransform):
    """
    A :class:`~base_transform.BaseTransform` which crops the input
    :class:`~fvdb_reality_capture.sfm_scene.SfmScene` points to lie within a specified bounding box.
    This transform additionally and updates the scene's masks to nullify pixels whose rays do not intersect
    the bounding box.

    .. note::

        If the input scene already has masks, these new masks will be composited with the existing masks to ensure that
        pixels outside the cropped region are properly masked. This can be disabled by setting
        ``composite_with_existing_masks`` to ``False``.

    Example usage:

    .. code-block:: python

        # Example usage:
        from fvdb_reality_capture import transforms
        from fvdb_reality_capture.sfm_scene import SfmScene
        import numpy as np

        # Bounding box in the format (min_x, min_y, min_z, max_x, max_y, max_z)
        scene_transform = transforms.CropScene(bbox=np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]))

        input_scene: SfmScene = ...  # Load or create an SfmScene

        # The transformed scene will have points only within the bounding box, and posed images will have
        # masks updated to nullify pixels corresponding to regions outside the cropped scene.
        transformed_scene: SfmScene = scene_transform(input_scene)

    """

    version = "1.0.0"

    def __init__(
        self,
        bbox: NumericMaxRank1,
        mask_format: Literal["png", "jpg", "npy"] = "png",
        composite_with_existing_masks: bool = True,
    ):
        """
        Create a new :class:`CropScene` transform with a bounding box.

        Args:
            bbox (NumericMaxRank1): A bounding box in the format ``(min_x, min_y, min_z, max_x, max_y, max_z)``.
            mask_format (Literal["png", "jpg", "npy"]): The format to save the masks in. Defaults to "png".
            composite_with_existing_masks (bool): Whether to composite the masks generated into existing masks for
                pixels corresponding to regions outside the cropped scene. If set to ``True``, existing masks
                will be loaded and composited with the new mask. Defaults to ``True``. The resulting composited
                mask will allow a pixel to be valid if it is valid in both the existing and new mask.
        """
        super().__init__()
        bbox = to_VecNf(bbox, 6, dtype=torch.float64).numpy()
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        if not len(bbox) == 6:
            raise ValueError("Bounding box must be a tuple of the form (min_x, min_y, min_z, max_x, max_y, max_z).")
        self._bbox = np.asarray(bbox).astype(np.float32)
        self._mask_format = mask_format
        if self._mask_format not in ["png", "jpg", "npy"]:
            raise ValueError(
                f"Unsupported mask format: {self._mask_format}. Supported formats are 'png', 'jpg', and 'npy'."
            )
        self._composite_with_existing_masks = composite_with_existing_masks

    @staticmethod
    def name() -> str:
        """
        Return the name of the :class:`CropScene` transform. **i.e.** ``"CropScene"``.

        Returns:
            str: The name of the :class:`CropScene` transform. **i.e.** ``"CropScene"``.
        """
        return "CropScene"

    @staticmethod
    def from_state_dict(state_dict: dict) -> "CropScene":
        """
        Create a :class:`CropScene` transform from a state dictionary created with :meth:`state_dict`.

        Args:
            state_dict (dict): The state dictionary for the transform.

        Returns:
            transform (CropScene): An instance of the :class:`CropScene` transform.
        """
        bbox = state_dict.get("bbox", None)
        if bbox is None:
            raise ValueError("State dictionary must contain 'bbox' key with bounding box coordinates.")
        if not isinstance(bbox, np.ndarray) or len(bbox) != 6:
            raise ValueError(
                "Bounding box must be a tuple or array of the form (min_x, min_y, min_z, max_x, max_y, max_z)."
            )
        return CropScene(bbox)

    def state_dict(self) -> dict:
        """
        Return the state of the :class:`CropScene` transform for serialization.

        You can use this state dictionary to recreate the transform using :meth:`from_state_dict`.

        Returns:
            state_dict (dict[str, Any]): A dictionary containing information to serialize/deserialize the transform.
        """
        return {
            "name": self.name(),
            "version": self.version,
            "bbox": self._bbox,
            "mask_format": self._mask_format,
            "composite_into_existing_masks": self._composite_with_existing_masks,
        }

    def __call__(self, input_scene: SfmScene) -> SfmScene:
        """
        Return a new :class:`~fvdb_reality_capture.sfm_scene.SfmScene` with points cropped to lie within the bounding
        box specified at initialization, and with masks updated to nullify pixels whose rays do not intersect the
        bounding box.

        Args:
            input_scene (SfmScene): The scene to be cropped.

        Returns:
            output_scene (SfmScene): The cropped scene.
        """
        # Ensure the bounding box is a numpy array of length 6
        bbox = np.asarray(self._bbox, dtype=np.float32)
        if bbox.shape != (6,):
            raise ValueError("Bounding box must be a 1D array of shape (6,)")

        self._logger.info(f"Cropping scene to bounding box: {self._bbox}")

        return _crop_scene_to_bbox(
            input_scene=input_scene,
            transform_name=self.name(),
            composite_with_existing_masks=self._composite_with_existing_masks,
            mask_format=self._mask_format,
            bbox=bbox,
            logger=self._logger,
        )


@transform
class CropSceneToPoints(BaseTransform):
    """
    A :class:`~base_transform.BaseTransform` which crops the input
    :class:`~fvdb_reality_capture.sfm_scene.SfmScene` points to lie within the bounding box around its points plus
    or minus a padding margin. This transform additionally and updates the scene's masks to nullify pixels whose rays
    do not intersect the bounding box.

    .. note::

        If the input scene already has masks, these new masks will be composited with the existing masks to ensure that
        pixels outside the cropped region are properly masked. This can be disabled by setting
        ``composite_with_existing_masks`` to ``False``.

    .. note::

        You may want to use this over :class:`CropScene` if you want the bounding box to depend on the input scene
        points rather than being fixed (*e.g.* if you don't know the bounding box ahead of time). This transform
        is also useful if you just want to apply conservative masking to the input scene based on its points.

    .. note::

        The margin is specified as a fraction of the bounding box size. For example, a margin of 0.1 will expand the
        bounding box by 10% (5% in all directions). So if the scene's bounding box is ``(0, 0, 0)`` to ``(1, 1, 1)``,
        a margin of ``0.1`` will result in a bounding box of ``(-0.05, -0.05, -0.05)`` to ``(1.05, 1.05, 1.05)``.
        The margin can also be negative to shrink the bounding box.

    Example usage:

    .. code-block:: python

        # Example usage:
        from fvdb_reality_capture import transforms
        from fvdb_reality_capture.sfm_scene import SfmScene
        import numpy as np

        # Crop the scene to be 0.1 times smaller than the bounding box around its points
        # (i.e. a margin of -0.1)
        scene_transform = transforms.CropSceneToPoints(margin=-0.1)

        input_scene: SfmScene = ...  # Load or create an SfmScene

        # The transformed scene will have points only within the bounding box of its points
        # minus a factor of 0.1 times the size. (i.e. a margin of -0.1).
        # Posed images will have masks updated to nullify pixels corresponding to regions outside the cropped scene.
        transformed_scene: SfmScene = scene_transform(input_scene)

    """

    version = "1.0.0"

    def __init__(
        self,
        margin: float = 0.0,
        mask_format: Literal["png", "jpg", "npy"] = "png",
        composite_with_existing_masks: bool = True,
    ):
        """
        Create a new :class:`CropSceneToPoints` transform with the given margin.

        Args:
            margin (float): The margin factor to apply around the bounding box of the points.
                Can be negative to shrink the bounding box. This is a fraction of the bounding box size.
                For example, a margin of ``0.1`` will expand the bounding box by 10% (5% in all directions),
                while a margin of ``-0.1`` will shrink the bounding box by 10% (-5% in all directions).
                Defaults to ``0.0``.
            mask_format (Literal["png", "jpg", "npy"]): The format to save the masks in. Defaults to "png".
            composite_with_existing_masks (bool): Whether to composite the masks generated into existing masks for
                pixels corresponding to regions outside the cropped scene. If set to True, existing masks
                will be loaded and composited with the new mask. Defaults to True.
        """
        super().__init__()
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._margin = margin

        self._mask_format = mask_format
        if self._mask_format not in ["png", "jpg", "npy"]:
            raise ValueError(
                f"Unsupported mask format: {self._mask_format}. Supported formats are 'png', 'jpg', and 'npy'."
            )
        self._composite_with_existing_masks = composite_with_existing_masks

    @staticmethod
    def name() -> str:
        """
        Return the name of the :class:`CropSceneToPoints` transform. *i.e.* ``"CropSceneToPoints"``.

        Returns:
            str: The name of the :class:`CropSceneToPoints` transform. *i.e.* ``"CropSceneToPoints"``.
        """
        return "CropSceneToPoints"

    @staticmethod
    def from_state_dict(state_dict: dict) -> "CropSceneToPoints":
        """
        Create a :class:`CropSceneToPoints` transform from a state dictionary generated with :meth:`state_dict`.

        Args:
            state_dict (dict[str, Any]): A dictionary containing information to serialize/deserialize the transform.

        Returns:
            transform (:class:`CropSceneToPoints`): An instance of the :class:`CropSceneToPoints` transform loaded
                from the state dictionary.
        """
        margin = state_dict.get("margin", None)
        if margin is None:
            raise ValueError("State dictionary must contain 'margin' key with margin value.")
        if not isinstance(margin, (float, int)):
            raise ValueError("Margin must be a non-negative float.")

        mask_format = state_dict.get("mask_format", None)
        if mask_format is None:
            raise ValueError("State dictionary must contain 'mask_format' key with mask format value.")
        if mask_format is not None and mask_format not in ["png", "jpg", "npy"]:
            raise ValueError(f"Unsupported mask format: {mask_format}. Supported formats are 'png', 'jpg', and 'npy'.")

        composite_into_existing_masks = state_dict.get("composite_into_existing_masks", None)
        if composite_into_existing_masks is None:
            raise ValueError("State dictionary must contain 'composite_into_existing_masks' key with boolean value.")
        if not isinstance(composite_into_existing_masks, bool):
            raise ValueError("composite_into_existing_masks must be a boolean.")
        return CropSceneToPoints(
            margin=margin, mask_format=mask_format, composite_with_existing_masks=composite_into_existing_masks
        )

    def state_dict(self) -> dict:
        """
        Return the state of the :class:`CropSceneToPoints` transform for serialization.

        You can use this state dictionary to recreate the transform using :meth:`from_state_dict`.

        Returns:
            state_dict (dict[str, Any]): A dictionary containing information to serialize/deserialize the transform.
        """
        return {
            "name": self.name(),
            "version": self.version,
            "margin": self._margin,
            "mask_format": self._mask_format,
            "composite_into_existing_masks": self._composite_with_existing_masks,
        }

    def __call__(self, input_scene: SfmScene) -> SfmScene:
        """
        Return a new :class:`~fvdb_reality_capture.sfm_scene.SfmScene` with points cropped to lie within the
        bounding box of the input scene's points plus or minus the margin specified at initialization,
        and with masks updated to nullify pixels whose rays do not intersect the bounding box.

        Args:
            input_scene (SfmScene): The scene to be cropped.

        Returns:
            output_scene (SfmScene): The cropped scene.
        """
        points_min = input_scene.points.min(axis=0)
        points_max = input_scene.points.max(axis=0)
        box_size = points_max - points_min
        padding = self._margin * box_size / 0.5
        points_min -= padding
        points_max += padding
        bbox = np.array(
            [
                points_min[0],
                points_min[1],
                points_min[2],
                points_max[0],
                points_max[1],
                points_max[2],
            ],
            dtype=np.float32,
        )

        if bbox.shape != (6,):
            raise ValueError("Bounding box must be a 1D array of shape (6,)")

        self._logger.info(f"Cropping scene to point bounding box {bbox} using margin {self._margin}")

        return _crop_scene_to_bbox(
            input_scene=input_scene,
            transform_name=self.name(),
            composite_with_existing_masks=self._composite_with_existing_masks,
            mask_format=self._mask_format,
            bbox=bbox,
            logger=self._logger,
        )
