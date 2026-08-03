# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import dataclasses
import unittest

import numpy as np

from fvdb_reality_capture.spatial_chunking import (
    chunk_ownership_mask,
    plan_spatial_chunks,
    resolve_chunk_domain,
)


class SpatialChunkPlannerTest(unittest.TestCase):
    def test_finite_scene_bbox_takes_precedence_over_point_bbox(self):
        scene_bbox = (-10.0, -20.0, -30.0, 10.0, 20.0, 30.0)
        point_bbox = (-1.0, -2.0, -3.0, 1.0, 2.0, 3.0)

        self.assertEqual(resolve_chunk_domain(scene_bbox), scene_bbox)
        self.assertEqual(resolve_chunk_domain(scene_bbox, point_bbox), scene_bbox)

    def test_unbounded_scene_bbox_falls_back_to_point_bbox(self):
        point_bbox = (-1.0, -2.0, -3.0, 1.0, 2.0, 3.0)

        resolved = resolve_chunk_domain(
            (-np.inf, -np.inf, -np.inf, np.inf, np.inf, np.inf),
            point_bbox,
        )

        self.assertEqual(resolved, point_bbox)

    def test_unbounded_scene_bbox_requires_finite_point_bbox(self):
        with self.assertRaisesRegex(ValueError, "finite point_bbox"):
            resolve_chunk_domain((-np.inf, -np.inf, -np.inf, np.inf, np.inf, np.inf), None)
        with self.assertRaisesRegex(ValueError, "only finite"):
            resolve_chunk_domain(None, (0.0, 0.0, 0.0, 1.0, np.inf, 1.0))

    def test_malformed_explicit_scene_bbox_is_not_silently_replaced(self):
        point_bbox = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)

        with self.assertRaisesRegex(ValueError, "only finite"):
            resolve_chunk_domain((0.0, 0.0, np.nan, 1.0, 1.0, 1.0), point_bbox)
        with self.assertRaisesRegex(ValueError, "strictly less"):
            resolve_chunk_domain((0.0, 0.0, 0.0, 1.0, 0.0, 1.0), point_bbox)

    def test_grid_shape_requires_exactly_three_positive_integers(self):
        domain = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        invalid_shapes = ((1, 1), (1, 1, 1, 1), (0, 1, 1), (-1, 1, 1), (1.0, 1, 1), (True, 1, 1))

        for invalid_shape in invalid_shapes:
            with self.subTest(nchunks=invalid_shape):
                with self.assertRaisesRegex(ValueError, "three positive integers"):
                    plan_spatial_chunks(domain, invalid_shape, 0.1)

    def test_overlap_requires_finite_half_open_unit_interval(self):
        domain = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)

        for invalid_overlap in (-0.1, 1.0, np.inf, -np.inf, np.nan, True, "0.1"):
            with self.subTest(overlap=invalid_overlap):
                with self.assertRaisesRegex(ValueError, r"\[0, 1\)"):
                    plan_spatial_chunks(domain, (1, 1, 1), invalid_overlap)  # type: ignore[arg-type]

    def test_domain_requires_finite_strictly_increasing_bounds(self):
        invalid_domains = (
            (0.0, 0.0, 0.0, 1.0, 1.0),
            (0.0, 0.0, 0.0, 1.0, np.inf, 1.0),
            (0.0, 0.0, 0.0, 1.0, np.nan, 1.0),
            (0.0, 0.0, 0.0, 0.0, 1.0, 1.0),
            (0.0, 2.0, 0.0, 1.0, 1.0, 1.0),
        )

        for invalid_domain in invalid_domains:
            with self.subTest(domain=invalid_domain):
                with self.assertRaises(ValueError):
                    plan_spatial_chunks(invalid_domain, (1, 1, 1), 0.1)

    def test_product_order_indices_and_ids_are_deterministic(self):
        chunks = plan_spatial_chunks((0.0, 0.0, 0.0, 2.0, 2.0, 2.0), (2, 2, 2), 0.0)

        self.assertEqual([chunk.index for chunk in chunks], list(range(8)))
        self.assertEqual([chunk.id for chunk in chunks], [f"chunk_{index:04d}" for index in range(8)])
        self.assertEqual(
            [chunk.grid_index for chunk in chunks],
            [
                (0, 0, 0),
                (0, 0, 1),
                (0, 1, 0),
                (0, 1, 1),
                (1, 0, 0),
                (1, 0, 1),
                (1, 1, 0),
                (1, 1, 1),
            ],
        )

    def test_core_and_halo_bounds_are_clipped_to_domain(self):
        chunks = plan_spatial_chunks((0.0, 0.0, 0.0, 10.0, 20.0, 30.0), (2, 1, 3), 0.2)
        first = chunks[0]
        middle = chunks[1]
        last = chunks[-1]

        self.assertEqual(first.core_bbox, (0.0, 0.0, 0.0, 5.0, 20.0, 10.0))
        self.assertEqual(first.train_bbox, (0.0, 0.0, 0.0, 5.5, 20.0, 11.0))
        self.assertEqual(middle.core_bbox, (0.0, 0.0, 10.0, 5.0, 20.0, 20.0))
        self.assertEqual(middle.train_bbox, (0.0, 0.0, 9.0, 5.5, 20.0, 21.0))
        self.assertEqual(last.core_bbox, (5.0, 0.0, 20.0, 10.0, 20.0, 30.0))
        self.assertEqual(last.train_bbox, (4.5, 0.0, 19.0, 10.0, 20.0, 30.0))

        for chunk in chunks:
            self.assertEqual(chunk.core_bbox[1], chunk.train_bbox[1])
            self.assertEqual(chunk.core_bbox[4], chunk.train_bbox[4])

    def test_single_chunk_axes_receive_no_halo(self):
        chunks = plan_spatial_chunks((-1.0, -2.0, -3.0, 1.0, 2.0, 3.0), (1, 1, 1), 0.9)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].train_bbox, chunks[0].core_bbox)
        self.assertEqual(chunks[0].inclusive_max, (True, True, True))

    def test_internal_max_face_belongs_only_to_following_chunk(self):
        chunks = plan_spatial_chunks((0.0, 0.0, 0.0, 2.0, 1.0, 1.0), (2, 1, 1), 0.2)
        first, second = chunks

        self.assertEqual(first.inclusive_max, (False, True, True))
        self.assertEqual(second.inclusive_max, (True, True, True))
        self.assertTrue(first.owns_center((0.0, 0.0, 0.0)))
        self.assertTrue(first.owns_center((0.999, 1.0, 1.0)))
        self.assertFalse(first.owns_center((1.0, 0.5, 0.5)))
        self.assertTrue(second.owns_center((1.0, 0.5, 0.5)))
        self.assertTrue(second.owns_center((2.0, 1.0, 1.0)))
        self.assertFalse(second.owns_center((2.001, 0.5, 0.5)))
        self.assertFalse(second.owns_center((np.nan, 0.5, 0.5)))

    def test_vectorized_ownership_is_unique_on_internal_and_global_faces(self):
        chunks = plan_spatial_chunks((0.0, 0.0, 0.0, 2.0, 2.0, 2.0), (2, 2, 2), 0.5)
        coordinates = np.array([0.0, 1.0, 2.0])
        centers = np.stack(np.meshgrid(coordinates, coordinates, coordinates, indexing="ij"), axis=-1).reshape(-1, 3)

        ownership_count = sum(chunk_ownership_mask(centers, chunk).astype(np.int32) for chunk in chunks)

        np.testing.assert_array_equal(ownership_count, np.ones(len(centers), dtype=np.int32))

    def test_vectorized_ownership_preserves_fractional_core_bounds_for_integer_centers(self):
        chunks = plan_spatial_chunks((0.0, 0.0, 0.0, 3.0, 1.0, 1.0), (2, 1, 1), 0.0)
        centers = np.array([[1, 0, 0], [2, 0, 0]], dtype=np.int64)

        np.testing.assert_array_equal(chunk_ownership_mask(centers, chunks[0]), np.array([True, False]))
        np.testing.assert_array_equal(chunk_ownership_mask(centers, chunks[1]), np.array([False, True]))

    def test_vectorized_ownership_rejects_non_numeric_centers(self):
        chunk = plan_spatial_chunks((0.0, 0.0, 0.0, 1.0, 1.0, 1.0), (1, 1, 1), 0.0)[0]

        with self.assertRaisesRegex(ValueError, "real numeric"):
            chunk_ownership_mask(np.array([["0", "0", "0"]]), chunk)

    def test_chunk_specs_are_immutable(self):
        chunk = plan_spatial_chunks((0.0, 0.0, 0.0, 1.0, 1.0, 1.0), (1, 1, 1), 0.0)[0]

        with self.assertRaises(dataclasses.FrozenInstanceError):
            chunk.index = 2  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
