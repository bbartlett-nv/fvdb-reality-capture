# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import pathlib
from typing import Any, Literal

import cv2
import numpy as np
import tqdm

from fvdb_reality_capture.mask_utils import (
    build_scene_cache_fingerprint,
    cache_fingerprint_matches,
    encode_binary_mask,
    load_binary_mask,
    mask_path_is_declared,
)
from fvdb_reality_capture.sfm_scene import SfmCache, SfmPosedImageMetadata, SfmScene
from fvdb_reality_capture.sfm_scene.scene_attribute import CROP_MASK_BBOX_ATTRIBUTE

from .base_transform import BaseTransform, transform

_DOWNSAMPLE_CACHE_MANIFEST_VERSION = 1
_DOWNSAMPLE_CACHE_MANIFEST_FILE = "manifest"
_DOWNSAMPLE_CACHE_FINGERPRINT_KEY = "fingerprint"
_DOWNSAMPLE_ALGORITHM_VERSION = 2


@transform
class DownsampleImages(BaseTransform):
    """
    A :class:`~base_transform.BaseTransform` which downsamples all images in an
    :class:`~fvdb_reality_capture.sfm_scene.SfmScene` by a specified factor and caches the downsampled images
    for future use.

    You can specify the cached downsampled image type (e.g., ``"jpg"`` or ``"png"``),
    the mode for downsampling (e.g., ``cv2.INTER_AREA``), and the rescaled JPEG quality (if using JPEG).

    If the downsampled images already exist in the scene's cache with the correct parameters,
    they will be loaded from the cache instead of being regenerated.

    Example usage:

    .. code-block:: python

        # Example usage:
        from fvdb_reality_capture import transforms
        from fvdb_reality_capture.sfm_scene import SfmScene

        scene_transform = transforms.DownsampleImages(4)
        input_scene: SfmScene = ...  # Load or create an SfmScene

        # The returned scene will have paths pointing to downsampled images by a factor of 4.
        transformed_scene: SfmScene = scene_transform(input_scene)
    """

    version = "1.0.0"

    def __init__(
        self,
        image_downsample_factor: int,
        image_type: Literal["jpg", "png"] = "jpg",
        rescale_sampling_mode: int = cv2.INTER_AREA,
        rescaled_jpeg_quality: int = 98,
    ):
        """
        Create a new :class:`DownsampleImages` transform with the specified downsampling factor
        and image caching parameters (image type, downsampling mode, and quality).

        .. note::
            We use enums from `OpenCV <https://opencv.org/>`_ for the ``rescale_sampling_mode`` parameter,
            e.g., ``cv2.INTER_AREA``, ``cv2.INTER_LINEAR``, ``cv2.INTER_CUBIC``, etc.
            This means if you want to change the resampling mode, you will need to ``import cv2```
            and pass in the appropriate enum value.
            See the `OpenCV documentation <https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121>`
            for more details on valid enum values.

        Args:
            image_downsample_factor (int): The factor by which to downsample the images.
            image_type (str): The type of the cached downsampled images, either "jpg" or "png".
            rescale_sampling_mode (int): The interpolation method to use for rescaling images.
                Note that we use enums from `OpenCV <https://opencv.org/>`_ for this parameter,
                e.g., ``cv2.INTER_AREA``, ``cv2.INTER_LINEAR``, ``cv2.INTER_CUBIC``, etc.
            rescaled_jpeg_quality (int): The quality of the JPEG images when saving them to the cache (1-100).
        """
        super().__init__()
        self._image_downsample_factor = image_downsample_factor
        self._image_type = image_type
        self._rescale_sampling_mode = rescale_sampling_mode
        self._rescaled_jpeg_quality = rescaled_jpeg_quality
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

    def __call__(self, input_scene: SfmScene) -> SfmScene:
        """
        Return a new :class:`~fvdb_reality_capture.sfm_scene.SfmScene` with images downsampled by the specified factor.
        *i.e.* images will be resized to ``(width / image_downsample_factor, height / image_downsample_factor)``.

        Args:
            input_scene (SfmScene): The input scene with images to be downsampled.

        Returns:
            output_scene (SfmScene): The scene with downsampled images.
        """
        if self._image_downsample_factor == 1:
            self._logger.info("Image downsample factor is 1, skipping downsampling.")
            return input_scene

        if len(input_scene.images) == 0:
            self._logger.warning("No images found in the SfmScene. Returning the input scene unchanged.")
            return input_scene
        if len(input_scene.cameras) == 0:
            self._logger.warning("No cameras found in the SfmScene. Returning the input scene unchanged.")
            return input_scene

        image_ids = np.asarray([image.image_id for image in input_scene.images], dtype=np.int64)
        if len(np.unique(image_ids)) != len(image_ids):
            raise ValueError("DownsampleImages requires unique image IDs for deterministic cache filenames")
        expected_fingerprint = build_scene_cache_fingerprint(
            input_scene,
            algorithm="downsample_images",
            algorithm_version=_DOWNSAMPLE_ALGORITHM_VERSION,
            settings={
                "image_downsample_factor": self._image_downsample_factor,
                "image_type": self._image_type,
                "rescale_sampling_mode": self._rescale_sampling_mode,
                "rescaled_jpeg_quality": self._rescaled_jpeg_quality,
                "mask_sampling_mode": cv2.INTER_NEAREST_EXACT,
                "mask_format": "png",
            },
            include_source_images=True,
        )
        expected_manifest = {
            "version": _DOWNSAMPLE_CACHE_MANIFEST_VERSION,
            _DOWNSAMPLE_CACHE_FINGERPRINT_KEY: expected_fingerprint,
        }

        input_cache: SfmCache = input_scene.cache
        cache_prefix = f"downsampled_{self._image_downsample_factor}x_{self._image_type}_q{self._rescaled_jpeg_quality}_m{self._rescale_sampling_mode}"
        output_cache = input_cache.make_folder(
            cache_prefix, description=f"Rescaled images by a factor of {self._image_downsample_factor}"
        )

        new_camera_metadata = {}
        for cam_id, cam_meta in input_scene.cameras.items():
            rescaled_cam_w = int(cam_meta.width / self._image_downsample_factor)
            rescaled_cam_h = int(cam_meta.height / self._image_downsample_factor)
            new_camera_metadata[cam_id] = cam_meta.resize(rescaled_cam_w, rescaled_cam_h)

        self._logger.info(
            f"Rescaling images using downsample factor {self._image_downsample_factor}, "
            f"sampling mode {self._rescale_sampling_mode}, and quality {self._rescaled_jpeg_quality}."
        )

        self._logger.info(f"Attempting to load downsampled images from cache.")
        # How many zeros to pad the image index in the mask file names
        num_zeropad = len(str(len(input_scene.images))) + 2

        new_image_metadata = []

        regenerate_cache = False

        num_masks = sum(mask_path_is_declared(image_meta.mask_path) for image_meta in input_scene.images)

        if output_cache.num_files != input_scene.num_images + num_masks + 1:
            if output_cache.num_files == 0:
                self._logger.info(f"No downsampled images found in the cache.")
            else:
                self._logger.info(
                    f"Inconsistent number of downsampled images in the cache. "
                    f"Expected {input_scene.num_images}, found {output_cache.num_files}. "
                    f"Clearing cache and regenerating downsampled images."
                )
            output_cache.clear_current_folder()
            regenerate_cache = True

        if not regenerate_cache:
            if not output_cache.has_file(_DOWNSAMPLE_CACHE_MANIFEST_FILE):
                self._logger.info("Downsample cache has no versioned manifest; regenerating legacy cache.")
                output_cache.clear_current_folder()
                regenerate_cache = True
            else:
                _, cached_manifest = output_cache.read_file(_DOWNSAMPLE_CACHE_MANIFEST_FILE)
                manifest_matches = (
                    isinstance(cached_manifest, dict)
                    and cached_manifest.get("version") == _DOWNSAMPLE_CACHE_MANIFEST_VERSION
                    and cache_fingerprint_matches(
                        cached_manifest.get(_DOWNSAMPLE_CACHE_FINGERPRINT_KEY),
                        expected_fingerprint,
                    )
                )
                if not manifest_matches:
                    self._logger.info("Downsample cache manifest is malformed or stale; regenerating cache.")
                    output_cache.clear_current_folder()
                    regenerate_cache = True

        for image_meta in input_scene.images:
            if regenerate_cache:
                break

            cache_image_filename = f"image_{image_meta.image_id:0{num_zeropad}}"
            if not output_cache.has_file(cache_image_filename):
                self._logger.info(
                    f"Image {cache_image_filename} not found in the cache. " f"Clearing cache and regenerating."
                )
                output_cache.clear_current_folder()
                regenerate_cache = True
                break

            cache_file_meta = output_cache.get_file_metadata(cache_image_filename)
            value_meta = cache_file_meta["metadata"]
            value_quality = value_meta.get("quality", -1)
            value_mode = value_meta.get("downsample_mode", -1)

            if (
                cache_file_meta.get("data_type", "") != self._image_type
                or value_quality != self._rescaled_jpeg_quality
                or value_mode != self._rescale_sampling_mode
            ):
                self._logger.info(
                    f"Output cache image metadata does not match expected format. "
                    f"Clearing the cache and regenerating downsampled images."
                )
                output_cache.clear_current_folder()
                regenerate_cache = True
                break
            mask_path = ""
            if mask_path_is_declared(image_meta.mask_path):
                cache_mask_filename = f"mask_{image_meta.image_id:0{num_zeropad}}"
                if not output_cache.has_file(cache_mask_filename):
                    self._logger.info(f"Mask {cache_mask_filename} is missing from the cache; regenerating.")
                    output_cache.clear_current_folder()
                    regenerate_cache = True
                    break
                mask_file_meta = output_cache.get_file_metadata(cache_mask_filename)
                mask_path = str(mask_file_meta["path"])
                try:
                    cached_mask = load_binary_mask(mask_path)
                except (FileNotFoundError, TypeError, ValueError) as error:
                    self._logger.info(f"Cached mask {mask_path} is unreadable ({error}); regenerating.")
                    output_cache.clear_current_folder()
                    regenerate_cache = True
                    break
                output_camera = new_camera_metadata[image_meta.camera_id]
                if mask_file_meta.get("data_type") != "png" or cached_mask.shape != (
                    output_camera.height,
                    output_camera.width,
                ):
                    self._logger.info(f"Cached mask {mask_path} has stale format or dimensions; regenerating.")
                    output_cache.clear_current_folder()
                    regenerate_cache = True
                    break
                _, raw_cached_mask = output_cache.read_file(cache_mask_filename)
                if not np.array_equal(raw_cached_mask, encode_binary_mask(cached_mask, "png")):
                    self._logger.info(
                        f"Cached mask {mask_path} does not use canonical 0/255 PNG encoding; regenerating."
                    )
                    output_cache.clear_current_folder()
                    regenerate_cache = True
                    break

            new_image_metadata.append(
                SfmPosedImageMetadata(
                    world_to_camera_matrix=image_meta.world_to_camera_matrix,
                    camera_to_world_matrix=image_meta.camera_to_world_matrix,
                    camera_metadata=new_camera_metadata[image_meta.camera_id],
                    camera_id=image_meta.camera_id,
                    image_path=str(cache_file_meta["path"]),
                    mask_path=mask_path,
                    point_indices=image_meta.point_indices,
                    image_id=image_meta.image_id,
                )
            )

        if regenerate_cache:
            new_image_metadata = []
            self._logger.info(
                f"Generating images downsampled by a factor of {self._image_downsample_factor} and saving to cache."
            )
            pbar = tqdm.tqdm(input_scene.images, unit="imgs")
            for _, image_meta in enumerate(pbar):
                image_filename = pathlib.Path(image_meta.image_path).name
                full_res_image_path = image_meta.image_path
                full_res_img = cv2.imread(full_res_image_path)
                assert full_res_img is not None, f"Failed to load image {full_res_image_path}"
                img_h, img_w = full_res_img.shape[:2]
                rescaled_img_h = int(img_h / self._image_downsample_factor)
                rescaled_img_w = int(img_w / self._image_downsample_factor)
                assert (
                    rescaled_img_w == new_camera_metadata[image_meta.camera_id].width
                ), f"Got mismatched widths {rescaled_img_w} !={new_camera_metadata[image_meta.camera_id].width}"
                assert (
                    rescaled_img_h == new_camera_metadata[image_meta.camera_id].height
                ), f"Got mismatched heights {rescaled_img_h} !={new_camera_metadata[image_meta.camera_id].height}"
                pbar.set_description(
                    f"Rescaling {image_filename} from {img_w} x {img_h} to {rescaled_img_w} x {rescaled_img_h}"
                )
                rescaled_image = cv2.resize(
                    full_res_img, (rescaled_img_w, rescaled_img_h), interpolation=self._rescale_sampling_mode
                )
                assert (
                    rescaled_image.shape[0] == rescaled_img_h and rescaled_image.shape[1] == rescaled_img_w
                ), f"Rescaled image {image_filename} has shape {rescaled_image.shape} but expected {rescaled_img_h, rescaled_img_w}"
                # Save the rescaled image to the cache
                cache_image_filename = f"image_{image_meta.image_id:0{num_zeropad}}"
                cache_file_meta = output_cache.write_file(
                    name=cache_image_filename,
                    data=rescaled_image,
                    data_type=self._image_type,
                    quality=self._rescaled_jpeg_quality,
                    metadata={
                        "quality": self._rescaled_jpeg_quality,
                        "downsample_mode": self._rescale_sampling_mode,
                    },
                )

                mask_path = ""
                if mask_path_is_declared(image_meta.mask_path):
                    full_res_mask = load_binary_mask(image_meta.mask_path)
                    if full_res_mask.shape != (img_h, img_w):
                        raise ValueError(
                            f"Mask shape {full_res_mask.shape} does not match source image shape {(img_h, img_w)} "
                            f"for image {image_meta.image_id}: {image_meta.mask_path}"
                        )
                    pbar.set_description(
                        f"Rescaling mask for {image_filename} from {img_w} x {img_h} "
                        f"to {rescaled_img_w} x {rescaled_img_h}"
                    )
                    rescaled_mask = (
                        cv2.resize(
                            full_res_mask.astype(np.uint8),
                            (rescaled_img_w, rescaled_img_h),
                            interpolation=cv2.INTER_NEAREST_EXACT,
                        )
                        > 0
                    )
                    cache_mask_filename = f"mask_{image_meta.image_id:0{num_zeropad}}"
                    cache_mask_meta = output_cache.write_file(
                        name=cache_mask_filename,
                        data=encode_binary_mask(rescaled_mask, "png"),
                        data_type="png",
                        metadata={"downsample_mode": cv2.INTER_NEAREST_EXACT},
                    )
                    mask_path = str(cache_mask_meta["path"])

                new_image_metadata.append(
                    SfmPosedImageMetadata(
                        world_to_camera_matrix=image_meta.world_to_camera_matrix,
                        camera_to_world_matrix=image_meta.camera_to_world_matrix,
                        camera_metadata=new_camera_metadata[image_meta.camera_id],
                        camera_id=image_meta.camera_id,
                        image_path=str(cache_file_meta["path"]),
                        mask_path=mask_path,
                        point_indices=image_meta.point_indices,
                        image_id=image_meta.image_id,
                    )
                )

            pbar.close()
            output_cache.write_file(
                _DOWNSAMPLE_CACHE_MANIFEST_FILE,
                expected_manifest,
                data_type="pt",
            )

            self._logger.info(
                f"Rescaled {input_scene.num_images} images by a factor of {self._image_downsample_factor} "
                f"and saved to cache with sampling mode {self._rescale_sampling_mode} and quality "
                f"{self._rescaled_jpeg_quality}."
            )

        new_attrs = {}
        for attr_name, attr in input_scene.attributes.items():
            if attr_name == CROP_MASK_BBOX_ATTRIBUTE:
                # These private bounds are in the input image coordinate system. Omit them so consumers
                # recompute bounds from the resized output masks instead of using stale full-resolution values.
                continue
            new_attrs[attr_name] = attr.on_downsample_images(
                attr_name=attr_name,
                downsample_factor=self._image_downsample_factor,
                output_cache=output_cache,
            )

        output_scene = input_scene.replace(
            cameras=new_camera_metadata,
            images=new_image_metadata,
            cache=output_cache,
            attributes=new_attrs,
        )

        return output_scene

    @staticmethod
    def name() -> str:
        """
        Return the name of the :class:`DownsampleImages` transform. **i.e.** ``"DownsampleImages"``.

        Returns:
            str: The name of the :class:`DownsampleImages` transform. **i.e.** ``"DownsampleImages"``.
        """
        return "DownsampleImages"

    def state_dict(self) -> dict[str, Any]:
        """
        Return the state of the :class:`DownsampleImages` transform for serialization.

        You can use this state dictionary to recreate the transform using :meth:`from_state_dict`.

        Returns:
            state_dict (dict[str, Any]): A dictionary containing information to serialize/deserialize the transform.
        """
        return {
            "name": self.name(),
            "version": self.version,
            "image_downsample_factor": self._image_downsample_factor,
            "image_type": self._image_type,
            "rescale_sampling_mode": self._rescale_sampling_mode,
            "rescaled_jpeg_quality": self._rescaled_jpeg_quality,
        }

    @staticmethod
    def from_state_dict(state_dict: dict[str, Any]) -> "DownsampleImages":
        """
        Create a :class:`DownsampleImages` transform from a state dictionary generated with :meth:`state_dict`.

        Args:
            state_dict (dict): The state dictionary for the transform.

        Returns:
            transform (DownsampleImages): An instance of the :class:`DownsampleImages` transform.
        """
        if state_dict["name"] != "DownsampleImages":
            raise ValueError(f"Expected state_dict with name 'DownsampleImages', got {state_dict['name']} instead.")

        return DownsampleImages(
            image_downsample_factor=state_dict["image_downsample_factor"],
            image_type=state_dict["image_type"],
            rescale_sampling_mode=state_dict["rescale_sampling_mode"],
            rescaled_jpeg_quality=state_dict["rescaled_jpeg_quality"],
        )
