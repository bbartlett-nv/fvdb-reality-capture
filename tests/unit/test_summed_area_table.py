# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

import torch

from fvdb_reality_capture.radiance_fields import gaussian_splatting
from fvdb_reality_capture.radiance_fields.gaussian_splatting import (
    _cumsum_last_dim_with_1d_scan,
    _tile_mask_to_summed_area_table,
)


class TestSummedAreaTable(unittest.TestCase):
    def test_one_dimensional_scan_matches_row_wise_cumsum(self):
        values = torch.tensor(
            [
                [[1, 0, 1, 1], [0, 1, 1, 0], [1, 1, 0, 0]],
                [[0, 1, 0, 1], [1, 0, 0, 1], [1, 0, 1, 0]],
            ],
            dtype=torch.int32,
        )
        expected = torch.cumsum(values, dim=-1, dtype=torch.int32)
        native_cumsum = torch.cumsum

        def cumsum_1d_only(input_tensor, dim, *, dtype=None, out=None):
            self.assertEqual(input_tensor.ndim, 1)
            self.assertEqual(dim, 0)
            return native_cumsum(input_tensor, dim, dtype=dtype, out=out)

        with patch.object(gaussian_splatting.torch, "cumsum", side_effect=cumsum_1d_only):
            actual = _cumsum_last_dim_with_1d_scan(values)

        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_summed_area_table_is_padded_and_batched(self):
        tile_mask = torch.tensor(
            [
                [[True, False, True], [False, True, True]],
                [[True, True, False], [True, False, True]],
            ],
            dtype=torch.bool,
        )
        expected = torch.tensor(
            [
                [[0, 0, 0, 0], [0, 1, 1, 2], [0, 1, 2, 4]],
                [[0, 0, 0, 0], [0, 1, 2, 2], [0, 2, 3, 4]],
            ],
            dtype=torch.int32,
        )

        actual = _tile_mask_to_summed_area_table(tile_mask)

        self.assertEqual(actual.dtype, torch.int32)
        self.assertEqual(actual.device, tile_mask.device)
        self.assertTrue(actual.is_contiguous())
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
