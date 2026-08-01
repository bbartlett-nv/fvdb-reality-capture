# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import torch
import torch.utils.data
import torchvision

from fvdb_reality_capture.sfm_scene import (
    DepthMapAttribute,
    PerImageRasterAttribute,
    PerImageValueAttribute,
    SfmCameraMetadata,
    SfmPosedImageMetadata,
    SfmScene,
)
from fvdb_reality_capture.sfm_scene.scene_attribute import CROP_MASK_BBOX_ATTRIBUTE

MaskRasterizationMode = Literal["off", "tiles", "bbox"]


@dataclass(frozen=True)
class _MaskRasterizationInfo:
    """Precomputed per-image metadata used by mask-aware rasterization."""

    full_image_size: tuple[int, int]
    raster_crop: tuple[int, int, int, int]
    image_loss_scale: float
    raster_tile_mask: np.ndarray | None = None


def _load_legacy_binary_mask(mask_path: str) -> np.ndarray:
    """Load a mask with the exact pre-ROI dataset format and threshold semantics."""
    if mask_path.endswith((".jpg", ".jpeg")):
        img_data = torchvision.io.read_file(mask_path)
        mask = torchvision.io.decode_jpeg(img_data, device="cpu")[0].numpy()
    elif mask_path.endswith(".png"):
        img_data = torchvision.io.read_file(mask_path)
        mask = torchvision.io.decode_png(img_data)[0].numpy()
    else:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        assert mask is not None, f"Failed to load mask: {mask_path}"
    return mask > 127


def _load_binary_mask(mask_path: str) -> np.ndarray:
    """Load a mask as a two-dimensional boolean array."""
    path = Path(mask_path)
    if not path.is_file():
        raise FileNotFoundError(f"Mask file does not exist: {mask_path}")

    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        img_data = torchvision.io.read_file(mask_path)
        mask = torchvision.io.decode_jpeg(img_data, device="cpu")[0].numpy()
    elif suffix == ".png":
        img_data = torchvision.io.read_file(mask_path)
        mask = torchvision.io.decode_png(img_data)[0].numpy()
    elif suffix == ".npy":
        mask = np.load(mask_path)
    else:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to load mask: {mask_path}")

    if mask.ndim != 2:
        raise ValueError(f"Mask must have shape (H, W); got {mask.shape} from {mask_path}")
    if mask.dtype == np.bool_:
        return np.ascontiguousarray(mask)
    is_normalized = mask.size == 0 or (np.min(mask) >= 0 and np.max(mask) <= 1)
    if np.issubdtype(mask.dtype, np.floating) and is_normalized:
        threshold = 0.5
    elif np.issubdtype(mask.dtype, np.integer) and is_normalized:
        threshold = 0
    else:
        threshold = 127
    return np.ascontiguousarray(mask > threshold)


def _raw_mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int, int]:
    """Return the tight half-open ``(x0, y0, x1, y1, count)`` bounds of a mask."""
    if mask.ndim != 2:
        raise ValueError(f"Mask must have shape (H, W); got {mask.shape}")
    valid_rows = np.flatnonzero(np.any(mask, axis=1))
    valid_cols = np.flatnonzero(np.any(mask, axis=0))
    valid_count = int(np.count_nonzero(mask))
    if valid_count == 0:
        return 0, 0, 0, 0, 0
    return int(valid_cols[0]), int(valid_rows[0]), int(valid_cols[-1]) + 1, int(valid_rows[-1]) + 1, valid_count


def _aligned_bbox_crop(
    raw_bbox: Sequence[int], image_size: tuple[int, int], context_pixels: int, tile_size: int
) -> tuple[int, int, int, int]:
    """Expand half-open XYXY bounds by context, align outward to tiles, and clip to the sensor."""
    if context_pixels < 0:
        raise ValueError(f"context_pixels must be non-negative; got {context_pixels}")
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive; got {tile_size}")
    if len(raw_bbox) != 4:
        raise ValueError(f"raw_bbox must contain four XYXY coordinates; got {raw_bbox}")

    image_height, image_width = image_size
    bbox_x0, bbox_y0, bbox_x1, bbox_y1 = (int(value) for value in raw_bbox)
    if not (0 <= bbox_x0 < bbox_x1 <= image_width and 0 <= bbox_y0 < bbox_y1 <= image_height):
        raise ValueError(f"raw_bbox {tuple(raw_bbox)} is empty or outside image size {image_size}")

    raw_x0 = max(0, bbox_x0 - context_pixels)
    raw_y0 = max(0, bbox_y0 - context_pixels)
    raw_x1 = min(image_width, bbox_x1 + context_pixels)
    raw_y1 = min(image_height, bbox_y1 + context_pixels)
    x0 = (raw_x0 // tile_size) * tile_size
    y0 = (raw_y0 // tile_size) * tile_size
    x1 = min(image_width, ((raw_x1 + tile_size - 1) // tile_size) * tile_size)
    y1 = min(image_height, ((raw_y1 + tile_size - 1) // tile_size) * tile_size)
    return x0, y0, x1 - x0, y1 - y0


def _aligned_mask_crop(mask: np.ndarray, context_pixels: int, tile_size: int) -> tuple[int, int, int, int]:
    """Return a clipped, context-expanded, tile-aligned ``(x, y, w, h)`` mask crop."""
    if context_pixels < 0:
        raise ValueError(f"context_pixels must be non-negative; got {context_pixels}")
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive; got {tile_size}")

    raw_x0, raw_y0, raw_x1, raw_y1, valid_count = _raw_mask_bbox(mask)
    if valid_count == 0:
        image_height, image_width = mask.shape
        return 0, 0, min(tile_size, image_width), min(tile_size, image_height)
    return _aligned_bbox_crop(
        (raw_x0, raw_y0, raw_x1, raw_y1), mask.shape, context_pixels=context_pixels, tile_size=tile_size
    )


def _dilated_mask_to_tiles(mask: np.ndarray, context_pixels: int, tile_size: int) -> np.ndarray:
    """Dilate a pixel mask for SSIM context, then compact it to an any-pixel tile mask."""
    if mask.ndim != 2:
        raise ValueError(f"Mask must have shape (H, W); got {mask.shape}")
    if context_pixels < 0:
        raise ValueError(f"context_pixels must be non-negative; got {context_pixels}")
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive; got {tile_size}")

    if context_pixels == 0:
        dilated = np.ascontiguousarray(mask, dtype=np.bool_)
    else:
        kernel_size = 2 * context_pixels + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        dilated = cv2.dilate(mask.astype(np.uint8, copy=False), kernel, borderType=cv2.BORDER_CONSTANT) > 0

    image_height, image_width = dilated.shape
    tile_height = (image_height + tile_size - 1) // tile_size
    tile_width = (image_width + tile_size - 1) // tile_size
    padded = np.zeros((tile_height * tile_size, tile_width * tile_size), dtype=np.bool_)
    padded[:image_height, :image_width] = dilated
    return np.ascontiguousarray(padded.reshape(tile_height, tile_size, tile_width, tile_size).any(axis=(1, 3)))


class SfmDataset(torch.utils.data.Dataset, Iterable):
    """
    A torch dataset encoding posed images from a Structure from Motion (SfM) pipeline.

    This class provides an interface to load and manipulate datasets from SfM pipelines
    (e.g. those generated from COLMAP).

    Each item in the dataset is an image with a corresponding camera pose, projection matrix,
    and optionally mask and depth information.

    The class also provides methods to access camera to world matrices, projection matrices,
    scene scale, and 3D points within the SFM scene.

    The dataset provides an API for common transformations on this kind of data used in reality capture.
    In particular it supports normalization of the scene, filtering points based on percentiles, and downsampling images.
    """

    def __init__(
        self,
        sfm_scene: SfmScene,
        dataset_indices: Sequence[int] | np.ndarray | torch.Tensor | None = None,
        patch_size: int | None = None,
        return_visible_points: bool = False,
        load_attributes: list[str] | None = None,
        mask_rasterization_mode: MaskRasterizationMode = "off",
        raster_tile_size: int = 16,
        raster_context_pixels: int = 10,
    ):
        """
        Create a new SfmDataset instance.

        Args:
            sfm_scene: The SfmScene for this dataset
            dataset_indices: Indices of images to include in the dataset. If None, all images will be used.
            patch_size: If not None, images will be randomly cropped to this size.
            return_visible_points: If True, depths of visible points will be loaded and included in each datum.
            load_attributes: Optional list of custom attribute names to load and include in each datum.
                For :class:`PerImageRasterAttribute`, the raster file is loaded and included as a tensor.
                For :class:`PerImageValueAttribute`, the in-memory value is included directly.
            mask_rasterization_mode: Optional mask-aware rasterization data to emit. ``"bbox"`` crops each datum
                to a context-expanded, tile-aligned mask bounding box. ``"tiles"`` retains full-frame data and
                emits a compact mask over raster tiles. ``"off"`` preserves the existing dataset behavior.
            raster_tile_size: Raster tile size used to align bounding boxes and compact tile masks.
            raster_context_pixels: Pixel context retained around the residual mask. The default of 10 preserves
                the current full-frame loss semantics for the 11x11 SSIM kernel.
        """
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

        self._sfm_scene = sfm_scene

        self.patch_size = patch_size
        self._return_visible_points = return_visible_points
        self._load_attributes = list(load_attributes) if load_attributes is not None else []
        if mask_rasterization_mode not in ("off", "tiles", "bbox"):
            raise ValueError(
                f"mask_rasterization_mode must be one of 'off', 'tiles', or 'bbox'; got {mask_rasterization_mode!r}"
            )
        if raster_tile_size <= 0:
            raise ValueError(f"raster_tile_size must be positive; got {raster_tile_size}")
        if raster_context_pixels < 0:
            raise ValueError(f"raster_context_pixels must be non-negative; got {raster_context_pixels}")
        if mask_rasterization_mode != "off" and patch_size is not None:
            raise ValueError("patch_size is not supported when mask_rasterization_mode is enabled")
        if mask_rasterization_mode == "bbox" and return_visible_points:
            raise ValueError("return_visible_points is not supported with mask_rasterization_mode='bbox'")
        self.mask_rasterization_mode: MaskRasterizationMode = mask_rasterization_mode
        self.raster_tile_size = raster_tile_size
        self.raster_context_pixels = raster_context_pixels

        _RESERVED_KEYS = {
            "projection",
            "camera_to_world",
            "world_to_camera",
            "camera_model",
            "distortion_coeffs",
            "image",
            "image_id",
            "image_path",
            "mask",
            "mask_path",
            "median_depth",
            "sparse_depth",
            "sparse_depth_uv",
            "raster_crop",
            "full_image_size",
            "image_loss_scale",
            "raster_tile_mask",
        }
        # Validate that every key each attribute will emit into the datum dict is
        # unique -- both against the reserved dataset keys and against keys emitted by
        # other requested attributes. A DepthMapAttribute emits both `<name>` and
        # `<name>_valid`, so e.g. requesting both "depth" and "depth_valid" would have
        # the former clobber the latter in __getitem__.
        emitted_by: dict[str, str] = {}
        for name in self._load_attributes:
            attr = self._sfm_scene.get_attribute(name)  # raises KeyError if not registered
            keys = [name]
            if isinstance(attr, DepthMapAttribute):
                keys.append(f"{name}_valid")
            for key in keys:
                if key in _RESERVED_KEYS:
                    raise ValueError(
                        f"Attribute '{name}' would emit dataset key '{key}', which collides with a "
                        f"reserved dataset key. Reserved keys: {sorted(_RESERVED_KEYS)}"
                    )
                if key in emitted_by:
                    raise ValueError(
                        f"Attribute '{name}' would emit dataset key '{key}', which is already emitted "
                        f"by attribute '{emitted_by[key]}'. Rename one of the attributes to avoid the collision."
                    )
                emitted_by[key] = name

        # If you specified image indices, we'll filter the dataset to only include those images.
        if dataset_indices is None:
            dataset_indices = np.arange(self._sfm_scene.num_images)
        elif isinstance(dataset_indices, torch.Tensor):
            dataset_indices = dataset_indices.cpu().numpy()
        else:
            dataset_indices = np.asarray(dataset_indices)
        if dataset_indices.dtype not in (np.int16, np.int32, np.int64, np.uint16, np.uint32, np.uint64):
            raise ValueError("Dataset indices must be integers")
        dataset_indices = dataset_indices.astype(np.int64)

        self._indices: np.ndarray = dataset_indices
        self._mask_rasterization_info: dict[int, _MaskRasterizationInfo] = {}
        if self.mask_rasterization_mode != "off":
            self._precompute_mask_rasterization_info()

    def _precompute_mask_rasterization_info(self) -> None:
        """Validate selected masks and precompute their compact rasterization metadata."""
        bbox_attribute: PerImageValueAttribute | None = None
        if self.mask_rasterization_mode == "bbox":
            try:
                candidate = self._sfm_scene.get_attribute(CROP_MASK_BBOX_ATTRIBUTE)
            except KeyError:
                candidate = None
            if candidate is not None and not isinstance(candidate, PerImageValueAttribute):
                raise TypeError(
                    f"Internal attribute {CROP_MASK_BBOX_ATTRIBUTE!r} must be a PerImageValueAttribute, "
                    f"got {type(candidate).__name__}"
                )
            bbox_attribute = candidate

        for scene_index_raw in self._indices:
            scene_index = int(scene_index_raw)
            image_meta = self._sfm_scene.images[scene_index]
            if image_meta.mask_path == "":
                raise ValueError(
                    f"mask_rasterization_mode={self.mask_rasterization_mode!r} requires a mask for every selected "
                    f"image; scene index {scene_index} ({image_meta.image_path}) has no mask"
                )

            camera_meta = image_meta.camera_metadata
            full_image_size = (camera_meta.height, camera_meta.width)
            image_height, image_width = full_image_size
            if self.mask_rasterization_mode == "bbox" and bbox_attribute is not None:
                if not Path(image_meta.mask_path).is_file():
                    raise FileNotFoundError(f"Mask file does not exist: {image_meta.mask_path}")
                raw_value = np.asarray(bbox_attribute.values[scene_index], dtype=np.int64)
                if raw_value.shape != (5,):
                    raise ValueError(
                        f"Internal attribute {CROP_MASK_BBOX_ATTRIBUTE!r} must contain five values per image; "
                        f"got shape {raw_value.shape} at scene index {scene_index}"
                    )
                raw_x0, raw_y0, raw_x1, raw_y1, valid_count = (int(value) for value in raw_value)
            else:
                mask = _load_binary_mask(image_meta.mask_path)
                if mask.shape != full_image_size:
                    raise ValueError(
                        f"Mask shape {mask.shape} does not match camera image size {full_image_size} for scene index "
                        f"{scene_index}: {image_meta.mask_path}"
                    )
                if self.mask_rasterization_mode == "bbox":
                    raw_x0, raw_y0, raw_x1, raw_y1, valid_count = _raw_mask_bbox(mask)

            if self.mask_rasterization_mode == "bbox":
                if not (0 <= valid_count <= image_height * image_width):
                    raise ValueError(
                        f"Invalid mask pixel count {valid_count} at scene index {scene_index}: {image_meta.mask_path}"
                    )
                if valid_count == 0:
                    if (raw_x0, raw_y0, raw_x1, raw_y1) != (0, 0, 0, 0):
                        raise ValueError(
                            f"Invalid raw mask bbox {(raw_x0, raw_y0, raw_x1, raw_y1)} for an empty mask at "
                            f"scene index {scene_index}: {image_meta.mask_path}"
                        )
                    raster_crop = (
                        0,
                        0,
                        min(self.raster_tile_size, image_width),
                        min(self.raster_tile_size, image_height),
                    )
                    image_loss_scale = 0.0
                else:
                    if not (0 <= raw_x0 < raw_x1 <= image_width and 0 <= raw_y0 < raw_y1 <= image_height):
                        raise ValueError(
                            f"Invalid raw mask bbox {(raw_x0, raw_y0, raw_x1, raw_y1)} for image size "
                            f"{full_image_size} at scene index {scene_index}: {image_meta.mask_path}"
                        )
                    if valid_count > (raw_x1 - raw_x0) * (raw_y1 - raw_y0):
                        raise ValueError(
                            f"Mask pixel count {valid_count} exceeds raw bbox area at scene index {scene_index}: "
                            f"{image_meta.mask_path}"
                        )
                    raster_crop = _aligned_bbox_crop(
                        (raw_x0, raw_y0, raw_x1, raw_y1),
                        full_image_size,
                        self.raster_context_pixels,
                        self.raster_tile_size,
                    )
                    _, _, crop_width, crop_height = raster_crop
                    image_loss_scale = (crop_height * crop_width) / (image_height * image_width)
                raster_tile_mask = None
            else:
                raster_crop = (0, 0, image_width, image_height)
                raster_tile_mask = _dilated_mask_to_tiles(mask, self.raster_context_pixels, self.raster_tile_size)
                image_loss_scale = 1.0 if np.any(raster_tile_mask) else 0.0

            self._mask_rasterization_info[scene_index] = _MaskRasterizationInfo(
                full_image_size=full_image_size,
                raster_crop=raster_crop,
                image_loss_scale=image_loss_scale,
                raster_tile_mask=raster_tile_mask,
            )

    @property
    def sfm_scene(self) -> SfmScene:
        """
        Returns the SfmScene associated with this dataset.

        Returns:
            sfm_scene (SfmScene): The SfmScene associated with this dataset.
        """
        return self._sfm_scene

    @property
    def indices(self) -> np.ndarray:
        """
        Return the indices of the images in the SfmScene used in the dataset.

        Returns:
            np.ndarray: The indices of the images in the SfmScene used in the dataset.
        """
        return self._indices

    @property
    def scene_bbox(self) -> np.ndarray:
        """
        Get the bounding box of the scene.

        The bounding box is defined as a tensor of shape (6,) where the first three elements are the minimum
        corner and the last three elements are the maximum corner of the bounding box.
        _i.e._ [min_x, min_y, min_z, max_x, max_y, max_z].

        Returns:
            torch.Tensor: A tensor of shape (6,) representing the bounding box of the scene.
        """
        return self._sfm_scene.scene_bbox

    @property
    def camera_to_world_matrices(self) -> np.ndarray:
        """
        Get the camera to world matrices for all images in the dataset.

        This returns the camera to world matrices as a numpy array of shape (N, 4, 4) where N is the number of images.

        Returns:
            np.ndarray: An Nx4x4 array of camera to world matrices for the cameras in the dataset.
        """
        return self.sfm_scene.camera_to_world_matrices[self._indices]

    @property
    def projection_matrices(self) -> np.ndarray:
        """
        Get the projection matrices mapping camera to pixel coordinates for all images in the dataset.

        This returns the scene projection matrices as a numpy array of shape (N, 3, 3) where N is the number of images.

        Returns:
            np.ndarray: An Nx3x3 array of projection matrices for the cameras in the dataset.
        """
        return np.stack([self._sfm_scene.images[i].camera_metadata.projection_matrix for i in self._indices], axis=0)

    @property
    def image_sizes(self) -> np.ndarray:
        """
        Get the image sizes for all images in the dataset.

        This returns the image sizes as a numpy array of shape (N, 2) where N is the number of images.
        Each row contains the height and width of the corresponding image.

        Returns:
            np.ndarray: An Nx2 array of image sizes for the cameras in the dataset.
        """
        return np.array(
            [
                (self._sfm_scene.images[i].camera_metadata.height, self._sfm_scene.images[i].camera_metadata.width)
                for i in self._indices
            ],
            dtype=np.int32,
        )

    @property
    def camera_models(self) -> np.ndarray:
        """
        Get the canonical camera model for each image in the dataset.

        Returns:
            np.ndarray: An array of integer-encoded ``fvdb_reality_capture.CameraModel`` values.
        """
        return np.array(
            [int(self._sfm_scene.images[i].camera_metadata.camera_model) for i in self._indices],
            dtype=np.int32,
        )

    @property
    def distortion_coeffs(self) -> np.ndarray:
        """
        Get packed distortion coefficients for each image in the dataset.

        Returns:
            np.ndarray: An ``(N, 12)`` array of packed distortion coefficients, zero-filled for
                camera models without distortion.
        """
        ret = np.zeros((len(self._indices), 12), dtype=np.float32)
        for out_idx, scene_idx in enumerate(self._indices):
            coeffs = self._sfm_scene.images[scene_idx].camera_metadata.distortion_coeffs
            if coeffs.size != 0:
                ret[out_idx] = coeffs
        return ret

    @property
    def points(self) -> np.ndarray:
        """
        Get the 3D points in the scene.
        This returns the points in world coordinates as a numpy array of shape (N, 3) where N is the number of points.

        Returns:
            np.ndarray: An Nx3 array of 3D points in the scene.
        """
        return self.sfm_scene.points

    @property
    def visible_point_indices(self) -> np.ndarray:
        """
        Return the indices of all points that are visible by some camera in the dataset.
        This is useful for filtering points that are not visible in any image.

        Returns:
            np.ndarray: An array of point indices that are visible in at least one image.
        """
        if not self._sfm_scene.has_visible_point_indices:
            return self._sfm_scene.points
        visible_points = set()
        for idx in self._indices:
            image_meta: SfmPosedImageMetadata = self._sfm_scene.images[idx]
            assert image_meta.point_indices is not None, (
                "SfmScene.has_visible_point_indices is True but image has no point indices"
            )
            visible_points.update(image_meta.point_indices.tolist())
        return np.array(list(visible_points))

    @property
    def points_rgb(self) -> np.ndarray:
        """
        Return the RGB colors of the points in the scene as a uint8 numpy array.
        The shape of the array is (N, 3) where N is the number of points.

        Returns:
            np.ndarray: An Nx3 array of uint8 RGB colors for the points in the scene.
        """
        return self._sfm_scene.points_rgb

    def __iter__(self):
        """
        Iterate over the dataset

        Yields:
            The next image in the dataset.
        """
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        """
        Get the number of images in the dataset.
        This is the number of images that will be returned by the dataset iterator.

        Returns:
            int: The number of images in the dataset.
        """
        return len(self._indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        """
        Get a single item from the dataset.

        An item is a dictionary with the following keys:
         - projection: The projection matrix for the camera.
         - camera_to_world: The camera to world transformation matrix.
         - world_to_camera: The world to camera transformation matrix.
         - image: The image tensor.
         - image_id: The global index of the image in the ``SfmScene``.
         - image_path: The file path of the image.
         - points (Optional): The projected points in the image (if return_visible_points is True).
         - sparse_depth (Optional): The depths of the projected points (if return_visible_points is True).
         - sparse_depth_uv (Optional): The pixel (uv) coordinates of the sparse depth points (if return_visible_points is True).
         - mask (Optional): The residual loss mask tensor (if available).
         - mask_path (Optional): The file path of the mask (if available).
         - raster_crop (Optional): Global image crop ``(x, y, width, height)`` for mask-aware rasterization.
         - full_image_size (Optional): Full sensor image size ``(height, width)`` before cropping.
         - image_loss_scale (Optional): Crop-to-full-image area ratio used to preserve image loss magnitude.
         - raster_tile_mask (Optional): Context-expanded compact tile mask for ``"tiles"`` mode.

        Returns:
            Dict[str, Any]: A dictionary containing the image data and metadata.
        """
        index = self._indices[item]

        image_meta: SfmPosedImageMetadata = self._sfm_scene.images[index]
        camera_meta: SfmCameraMetadata = image_meta.camera_metadata

        if image_meta.image_path.endswith(".jpg") or image_meta.image_path.endswith(".jpeg"):
            data = torchvision.io.read_file(image_meta.image_path)
            image = torchvision.io.decode_jpeg(data, device="cpu")
            assert isinstance(image, torch.Tensor)
            image = image.permute(1, 2, 0).numpy()
        elif image_meta.image_path.endswith(".png"):
            data = torchvision.io.read_file(image_meta.image_path)
            image = torchvision.io.decode_png(data).permute(1, 2, 0).numpy()
        else:
            image = cv2.imread(image_meta.image_path, cv2.IMREAD_UNCHANGED)
            assert image is not None, f"Failed to load image: {image_meta.image_path}"
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image.ndim == 2:
            image = image[:, :, None]
        if image_meta.mask_path == "":
            mask = None
        elif self.mask_rasterization_mode == "off":
            mask = _load_legacy_binary_mask(image_meta.mask_path)
        else:
            mask = _load_binary_mask(image_meta.mask_path)
        projection_matrix = camera_meta.projection_matrix.copy()
        camera_to_world_matrix = image_meta.camera_to_world_matrix.copy()
        world_to_camera_matrix = image_meta.world_to_camera_matrix.copy()

        scene_index = int(index)
        rasterization_info = self._mask_rasterization_info.get(scene_index)
        if rasterization_info is not None:
            if image.shape[:2] != rasterization_info.full_image_size:
                raise ValueError(
                    f"Decoded image shape {image.shape[:2]} does not match camera image size "
                    f"{rasterization_info.full_image_size} for scene index {scene_index}: {image_meta.image_path}"
                )
            if mask is None or mask.shape != rasterization_info.full_image_size:
                raise ValueError(
                    f"Mask shape {None if mask is None else mask.shape} does not match camera image size "
                    f"{rasterization_info.full_image_size} for scene index {scene_index}: {image_meta.mask_path}"
                )

        if self.patch_size is not None:
            # Random crop. This path is mutually exclusive with mask-aware rasterization.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            if mask is not None:
                mask = mask[y : y + self.patch_size, x : x + self.patch_size]
            projection_matrix[0, 2] -= x
            projection_matrix[1, 2] -= y
        elif rasterization_info is not None and self.mask_rasterization_mode == "bbox":
            x, y, crop_width, crop_height = rasterization_info.raster_crop
            image = image[y : y + crop_height, x : x + crop_width]
            assert mask is not None
            mask = mask[y : y + crop_height, x : x + crop_width]

        data = {
            "projection": torch.from_numpy(projection_matrix).float(),
            "camera_to_world": torch.from_numpy(camera_to_world_matrix).float(),
            "world_to_camera": torch.from_numpy(world_to_camera_matrix).float(),
            "camera_model": torch.tensor(int(camera_meta.camera_model), dtype=torch.int32),
            "distortion_coeffs": torch.from_numpy(
                camera_meta.distortion_coeffs.copy()
                if camera_meta.distortion_coeffs.size != 0
                else np.zeros((12,), np.float32)
            ).float(),
            "image": image,
            "image_id": index,  # the global index of the image in the SfmScene
            "image_path": image_meta.image_path,
        }

        if mask is not None:
            data["mask_path"] = image_meta.mask_path
            data["mask"] = mask

        if rasterization_info is not None:
            data["raster_crop"] = np.asarray(rasterization_info.raster_crop, dtype=np.int64)
            data["full_image_size"] = np.asarray(rasterization_info.full_image_size, dtype=np.int64)
            data["image_loss_scale"] = np.float32(rasterization_info.image_loss_scale)
            if rasterization_info.raster_tile_mask is not None:
                data["raster_tile_mask"] = rasterization_info.raster_tile_mask.copy()

        # If you asked to load depths, we'll load the depths of visible colmap points
        if self._return_visible_points:
            # projected points to image plane to get depths
            points_world = self._sfm_scene.points[image_meta.point_indices]  # (M, 3)
            points_cam = (world_to_camera_matrix[:3, :3] @ points_world.T + world_to_camera_matrix[:3, 3:4]).T  # (M, 3)
            points_proj = (projection_matrix @ points_cam.T).T  # (M, 3)
            points = points_proj[:, :2] / points_proj[:, 2:3]  # (M, 2)
            depths = points_cam[:, 2]  # (M,)
            if self.patch_size is not None:
                points[:, 0] -= x
                points[:, 1] -= y
            # filter out points outside the image
            selector = (
                (points[:, 0] >= 0)
                & (points[:, 0] < image.shape[1])
                & (points[:, 1] >= 0)
                & (points[:, 1] < image.shape[0])
                & (depths > 0)
            )
            points = points[selector]
            depths = depths[selector]
            median_depth = np.median(depths)
            data["median_depth"] = torch.tensor(median_depth).float()
            data["sparse_depth_uv"] = torch.from_numpy(points).float()
            data["sparse_depth"] = torch.from_numpy(depths).float()

        for attr_name in self._load_attributes:
            attr = self._sfm_scene.get_attribute(attr_name)
            if isinstance(attr, DepthMapAttribute):
                depth_np, valid_np = attr.load_depth(index)
                depth = torch.from_numpy(depth_np).contiguous()
                valid = torch.from_numpy(valid_np).contiguous()
                if self.patch_size is not None:
                    depth = depth[y : y + self.patch_size, x : x + self.patch_size]
                    valid = valid[y : y + self.patch_size, x : x + self.patch_size]
                elif rasterization_info is not None and self.mask_rasterization_mode == "bbox":
                    crop_x, crop_y, crop_width, crop_height = rasterization_info.raster_crop
                    depth = depth[crop_y : crop_y + crop_height, crop_x : crop_x + crop_width]
                    valid = valid[crop_y : crop_y + crop_height, crop_x : crop_x + crop_width]
                data[attr_name] = depth
                data[f"{attr_name}_valid"] = valid
            elif isinstance(attr, PerImageRasterAttribute):
                path = attr.paths[index]
                if path.endswith(".npy"):
                    raster = torch.from_numpy(np.load(path))
                elif path.endswith(".pt"):
                    loaded_pt = torch.load(path, map_location="cpu", weights_only=False)
                    if isinstance(loaded_pt, np.ndarray):
                        raster = torch.from_numpy(loaded_pt)
                    elif isinstance(loaded_pt, torch.Tensor):
                        raster = loaded_pt
                    else:
                        raise TypeError(
                            f"Raster attribute '{attr_name}' loaded from {path} is {type(loaded_pt).__name__}, "
                            f"expected torch.Tensor or numpy.ndarray."
                        )
                else:
                    loaded = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                    if loaded is None:
                        raise FileNotFoundError(f"Failed to load raster attribute '{attr_name}' from {path}")
                    raster = torch.from_numpy(loaded)
                if self.patch_size is not None:
                    raster = raster[y : y + self.patch_size, x : x + self.patch_size]
                elif rasterization_info is not None and self.mask_rasterization_mode == "bbox":
                    crop_x, crop_y, crop_width, crop_height = rasterization_info.raster_crop
                    raster = raster[crop_y : crop_y + crop_height, crop_x : crop_x + crop_width]
                data[attr_name] = raster
            elif isinstance(attr, PerImageValueAttribute):
                data[attr_name] = attr.values[index]
            else:
                raise TypeError(
                    f"Unsupported attribute type for '{attr_name}': {type(attr).__name__}. "
                    f"Supported types are PerImageRasterAttribute and PerImageValueAttribute."
                )

        return data


__all__ = ["SfmDataset"]
