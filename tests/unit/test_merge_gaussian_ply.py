# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import pathlib

import numpy as np
import pytest
import torch

from fvdb_reality_capture import GaussianSplat3d
from fvdb_reality_capture.tools import (
    GaussianPlyMergeSource,
    gaussian_center_ownership_mask,
    merge_gaussian_ply_files,
    validate_gaussian_ply_file,
)
from fvdb_reality_capture.tools import _merge_gaussian_ply as merge_module

_BASE_PROPERTY_NAMES = (
    "x",
    "y",
    "z",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
)
_METADATA_MAGIC = "fvdb_ply_af_8198767135"


def _property_names(rest_count: int = 0) -> tuple[str, ...]:
    return (
        _BASE_PROPERTY_NAMES
        + tuple(f"f_dc_{index}" for index in range(3))
        + tuple(f"f_rest_{index}" for index in range(rest_count))
    )


def _make_records(means: np.ndarray, rest_count: int = 0, offset: float = 0.0) -> np.ndarray:
    means = np.asarray(means, dtype=np.float32)
    records = np.arange(means.shape[0] * len(_property_names(rest_count)), dtype=np.float32).reshape(means.shape[0], -1)
    records += offset
    records[:, :3] = means
    return records


def _write_gaussian_ply(
    path: pathlib.Path,
    records: np.ndarray,
    *,
    rest_count: int = 0,
    trailing_tensor_metadata: bool = False,
) -> None:
    comments = ["comment fvdb_gs_ply_version fvdb_ply 1.0.0"]
    trailing_element: list[str] = []
    trailing_body = b""
    if trailing_tensor_metadata:
        comments.append(f"comment {_METADATA_MAGIC}ignored_input_value|tensor|1,2")
        trailing_element = ["element ignored_input_value 2", "property int value"]
        trailing_body = np.asarray([17, 23], dtype="<i4").tobytes()

    lines = [
        "ply",
        "format binary_little_endian 1.0",
        *comments,
        f"element vertex {records.shape[0]}",
        *(f"property float {name}" for name in _property_names(rest_count)),
        *trailing_element,
        "end_header",
    ]
    with path.open("wb") as output_file:
        output_file.write(("\n".join(lines) + "\n").encode("utf-8"))
        output_file.write(np.asarray(records, dtype="<f4").tobytes())
        output_file.write(trailing_body)


def test_ownership_mask_assigns_shared_faces_exactly_once():
    centers = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [np.nan, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    left = gaussian_center_ownership_mask(centers, (0.0, -1.0, -1.0, 1.0, 1.0, 1.0))
    right = gaussian_center_ownership_mask(
        centers,
        (1.0, -1.0, -1.0, 2.0, 1.0, 1.0),
        inclusive_max=(True, True, True),
    )

    np.testing.assert_array_equal(left, [True, False, False, False])
    np.testing.assert_array_equal(right, [False, True, True, False])
    np.testing.assert_array_equal(left.astype(np.int8) + right.astype(np.int8), [1, 1, 1, 0])


def test_ownership_mask_uses_float32_representable_bbox_faces():
    lower_boundary = np.float32(0.7)
    upper_boundary = np.float32(1.1)
    centers = np.asarray(
        [
            [lower_boundary, 0.0, 0.0],
            [upper_boundary, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    owned = gaussian_center_ownership_mask(
        centers,
        (0.7, -1.0, -1.0, 1.1, 1.0, 1.0),
        inclusive_max=(True, True, True),
    )

    np.testing.assert_array_equal(owned, [True, True])


def test_streaming_merge_filters_to_cores_and_round_trips_metadata_on_cpu(tmp_path: pathlib.Path):
    first_path = tmp_path / "first.ply"
    second_path = tmp_path / "second.ply"
    output_path = tmp_path / "merged.ply"
    first_records = _make_records(
        np.asarray([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [np.nan, 0.0, 0.0]])
    )
    second_records = _make_records(
        np.asarray([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
        offset=1_000.0,
    )
    _write_gaussian_ply(first_path, first_records, trailing_tensor_metadata=True)
    _write_gaussian_ply(second_path, second_records)

    metadata = {
        "camera_count": 7,
        "ids": np.asarray([4, 8], dtype=np.int16),
        "metric_scale": 1.25,
        "scene_name": "CPU merge test",
        "transform": torch.arange(6, dtype=torch.float64).reshape(2, 3),
    }
    result = merge_gaussian_ply_files(
        [
            GaussianPlyMergeSource(first_path, (-10.0, -1.0, -1.0, 2.0, 1.0, 1.0)),
            GaussianPlyMergeSource(
                second_path,
                (2.0, -1.0, -1.0, 4.0, 1.0, 1.0),
                inclusive_max=(True, True, True),
            ),
        ],
        output_path,
        metadata,
        records_per_block=2,
    )

    merged, merged_metadata = GaussianSplat3d.from_ply(output_path, device="cpu")
    expected_records = np.concatenate((first_records[:3], second_records))
    torch.testing.assert_close(merged.means, torch.from_numpy(expected_records[:, :3]))
    torch.testing.assert_close(merged.logit_opacities, torch.from_numpy(expected_records[:, 3]))
    torch.testing.assert_close(merged.log_scales, torch.from_numpy(expected_records[:, 4:7]))
    torch.testing.assert_close(merged.quats, torch.from_numpy(expected_records[:, 7:11]))
    torch.testing.assert_close(merged.sh0, torch.from_numpy(expected_records[:, 11:14]).unsqueeze(1))
    assert merged.shN.shape == (6, 0, 3)

    assert "ignored_input_value" not in merged_metadata
    assert merged_metadata["camera_count"] == 7
    assert merged_metadata["metric_scale"] == 1.25
    assert merged_metadata["scene_name"] == "CPU merge test"
    torch.testing.assert_close(merged_metadata["ids"], torch.tensor([4, 8], dtype=torch.int16))
    torch.testing.assert_close(merged_metadata["transform"], metadata["transform"])

    assert result.input_gaussians == 8
    assert result.output_gaussians == 6
    assert result.filtered_gaussians == 2
    assert result.retained_per_source == (3, 3)


def test_merge_preserves_all_records_without_a_core_filter(tmp_path: pathlib.Path):
    first_path = tmp_path / "first.ply"
    empty_path = tmp_path / "empty.ply"
    output_path = tmp_path / "merged.ply"
    records = _make_records(np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    _write_gaussian_ply(first_path, records)
    _write_gaussian_ply(empty_path, np.empty((0, records.shape[1]), dtype=np.float32))

    result = merge_gaussian_ply_files([first_path, empty_path], output_path, records_per_block=1)
    merged, metadata = GaussianSplat3d.from_ply(output_path, device="cpu")

    torch.testing.assert_close(merged.means, torch.from_numpy(records[:, :3]))
    assert metadata == {}
    assert result.retained_per_source == (2, 0)


def test_all_empty_sources_are_rejected_without_creating_an_output(tmp_path: pathlib.Path):
    first_path = tmp_path / "first_empty.ply"
    second_path = tmp_path / "second_empty.ply"
    output_path = tmp_path / "merged_empty.ply"
    empty_records = np.empty((0, len(_property_names())), dtype=np.float32)
    _write_gaussian_ply(first_path, empty_records)
    _write_gaussian_ply(second_path, empty_records)

    with pytest.raises(ValueError, match="retained no Gaussians"):
        merge_gaussian_ply_files([first_path, second_path], output_path)

    assert not output_path.exists()


def test_incompatible_schemas_are_rejected_before_output_is_created(tmp_path: pathlib.Path):
    degree_zero_path = tmp_path / "degree_zero.ply"
    higher_degree_path = tmp_path / "higher_degree.ply"
    output_path = tmp_path / "merged.ply"
    _write_gaussian_ply(degree_zero_path, _make_records(np.asarray([[0.0, 0.0, 0.0]])))
    _write_gaussian_ply(
        higher_degree_path,
        _make_records(np.asarray([[1.0, 0.0, 0.0]]), rest_count=3),
        rest_count=3,
    )

    with pytest.raises(ValueError, match="Incompatible Gaussian vertex schema"):
        merge_gaussian_ply_files([degree_zero_path, higher_degree_path], output_path)

    assert not output_path.exists()


def test_truncated_source_is_rejected(tmp_path: pathlib.Path):
    source_path = tmp_path / "truncated.ply"
    output_path = tmp_path / "merged.ply"
    _write_gaussian_ply(source_path, _make_records(np.asarray([[0.0, 0.0, 0.0]])))
    source_path.write_bytes(source_path.read_bytes()[:-1])

    with pytest.raises(ValueError, match="body size"):
        merge_gaussian_ply_files([source_path], output_path)


def test_validator_checks_complete_body_and_expected_vertex_count(tmp_path: pathlib.Path):
    source_path = tmp_path / "source.ply"
    _write_gaussian_ply(source_path, _make_records(np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])))

    assert validate_gaussian_ply_file(source_path, expected_vertex_count=2) == 2
    with pytest.raises(ValueError, match="vertex count mismatch"):
        validate_gaussian_ply_file(source_path, expected_vertex_count=1)

    source_path.write_bytes(source_path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="body size"):
        validate_gaussian_ply_file(source_path)


def test_existing_output_is_not_overwritten_by_default(tmp_path: pathlib.Path):
    source_path = tmp_path / "source.ply"
    output_path = tmp_path / "existing.ply"
    _write_gaussian_ply(source_path, _make_records(np.asarray([[0.0, 0.0, 0.0]])))
    output_path.write_bytes(b"keep me")

    with pytest.raises(FileExistsError):
        merge_gaussian_ply_files([source_path], output_path)

    assert output_path.read_bytes() == b"keep me"


def test_output_created_during_install_is_not_overwritten(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    source_path = tmp_path / "source.ply"
    output_path = tmp_path / "racing.ply"
    _write_gaussian_ply(source_path, _make_records(np.asarray([[0.0, 0.0, 0.0]])))
    racing_contents = b"created by another process"
    real_link = merge_module.os.link

    def racing_link(source, destination):
        pathlib.Path(destination).write_bytes(racing_contents)
        return real_link(source, destination)

    monkeypatch.setattr(merge_module.os, "link", racing_link)
    with pytest.raises(FileExistsError, match="created while merging"):
        merge_gaussian_ply_files([source_path], output_path)

    assert output_path.read_bytes() == racing_contents
