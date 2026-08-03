# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import hashlib
import json
import os
import pathlib
import socket
import tempfile
import unittest
from unittest import mock

from fvdb_reality_capture.chunk_manifest import (
    ChunkManifestError,
    ChunkRunLock,
    ChunkRunLockError,
    ChunkRunManifest,
    serialize_chunk_spec,
)
from fvdb_reality_capture.spatial_chunking import plan_spatial_chunks


class ChunkRunManifestTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary_directory.name)
        self.work_dir = self.root / "chunk_work"
        self.signature = {
            "dataset": "/data/scene",
            "config": {"epochs": 200, "features": ["bbox", "random-background"]},
        }
        self.plan = plan_spatial_chunks((0.0, 0.0, 0.0, 2.0, 1.0, 1.0), (2, 1, 1), 0.1)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _open(self, signature=None, plan=None):
        return ChunkRunManifest.open_or_create(
            self.work_dir,
            self.signature if signature is None else signature,
            self.plan if plan is None else plan,
        )

    def _write_artifact(self, name: str, data: bytes = b"ply-data") -> pathlib.Path:
        path = self.work_dir / name
        path.write_bytes(data)
        return path

    def test_initialize_writes_v2_plan_and_pending_states(self):
        input_signature = json.loads(json.dumps(self.signature))
        manifest = self._open(signature=input_signature)
        input_signature["config"]["epochs"] = 1

        document = json.loads(manifest.path.read_text(encoding="utf-8"))
        self.assertEqual(document["kind"], "fvdb-reality-capture-chunk-run")
        self.assertEqual(document["version"], 2)
        self.assertEqual(document["run_signature"], self.signature)
        self.assertEqual(document["chunk_plan"], [serialize_chunk_spec(chunk) for chunk in self.plan])
        self.assertEqual([state["status"] for state in document["chunk_states"]], ["pending", "pending"])
        self.assertEqual(manifest.run_signature, self.signature)
        self.assertEqual(manifest.chunk_plan, self.plan)
        self.assertEqual(manifest.pending_chunks, self.plan)
        self.assertEqual(manifest.completed_chunks, ())
        for state in document["chunk_states"]:
            self.assertIsNone(state["artifact_path"])
            self.assertIsNone(state["artifact_size"])
            self.assertIsNone(state["artifact_sha256"])
            self.assertIsNone(state["initial_count"])
            self.assertIsNone(state["final_count"])

    def test_matching_manifest_resumes_completed_chunks(self):
        manifest = self._open()
        artifact = self._write_artifact("chunk_0000.ply")
        state = manifest.mark_complete("chunk_0000", artifact, initial_count=123, final_count=100)

        reordered_signature = {"config": self.signature["config"], "dataset": self.signature["dataset"]}
        resumed = self._open(signature=reordered_signature)

        self.assertEqual(state.status, "complete")
        self.assertEqual(state.artifact_path, artifact)
        expected_digest = hashlib.sha256(b"ply-data").hexdigest()
        self.assertEqual(state.artifact_size, len(b"ply-data"))
        self.assertEqual(state.artifact_sha256, expected_digest)
        stored_state = json.loads(manifest.path.read_text(encoding="utf-8"))["chunk_states"][0]
        self.assertEqual((stored_state["artifact_size"], stored_state["artifact_sha256"]), (8, expected_digest))
        self.assertEqual(state.initial_count, 123)
        self.assertEqual(state.final_count, 100)
        self.assertEqual(resumed.completed_chunks, (self.plan[0],))
        self.assertEqual(resumed.pending_chunks, (self.plan[1],))
        self.assertEqual(resumed.state("chunk_0000"), state)

    def test_mark_complete_streams_sha256_in_bounded_blocks(self):
        manifest = self._open()
        contents = b"0123456789"
        artifact = self._write_artifact("chunk_0000.ply", contents)
        requested_read_sizes = []
        real_fdopen = os.fdopen
        test_case = self

        class GuardedReader:
            def __init__(self, wrapped):
                self.wrapped = wrapped

            def __enter__(self):
                self.wrapped.__enter__()
                return self

            def __exit__(self, exception_type, exception, traceback):
                return self.wrapped.__exit__(exception_type, exception, traceback)

            def read(self, size=-1):
                test_case.assertGreater(size, 0)
                test_case.assertLessEqual(size, 4)
                requested_read_sizes.append(size)
                return self.wrapped.read(size)

            def fileno(self):
                return self.wrapped.fileno()

        def guarded_fdopen(file_descriptor, mode):
            return GuardedReader(real_fdopen(file_descriptor, mode))

        with (
            mock.patch("fvdb_reality_capture.chunk_manifest._ARTIFACT_HASH_BLOCK_SIZE", 4),
            mock.patch("fvdb_reality_capture.chunk_manifest.os.fdopen", side_effect=guarded_fdopen),
        ):
            state = manifest.mark_complete("chunk_0000", artifact, initial_count=10, final_count=9)

        self.assertEqual(requested_read_sizes, [4, 4, 4, 4])
        self.assertEqual(state.artifact_size, len(contents))
        self.assertEqual(state.artifact_sha256, hashlib.sha256(contents).hexdigest())

    def test_signature_mismatch_rejects_without_modifying_work_dir(self):
        manifest = self._open()
        sentinel = self._write_artifact("keep.txt", b"keep")
        original_manifest = manifest.path.read_bytes()

        with self.assertRaisesRegex(ChunkManifestError, "signature"):
            self._open(signature={"dataset": "/different", "config": self.signature["config"]})

        self.assertEqual(manifest.path.read_bytes(), original_manifest)
        self.assertEqual(sentinel.read_bytes(), b"keep")

    def test_plan_mismatch_rejects_without_modifying_work_dir(self):
        manifest = self._open()
        original_manifest = manifest.path.read_bytes()
        different_plan = plan_spatial_chunks((0.0, 0.0, 0.0, 3.0, 1.0, 1.0), (2, 1, 1), 0.1)

        with self.assertRaisesRegex(ChunkManifestError, "plan"):
            self._open(plan=different_plan)

        self.assertEqual(manifest.path.read_bytes(), original_manifest)

    def test_existing_nonmanifest_directory_is_rejected_without_deletion(self):
        self.work_dir.mkdir()
        sentinel = self._write_artifact("keep.txt", b"keep")

        with self.assertRaisesRegex(ChunkManifestError, "no valid"):
            self._open()

        self.assertEqual(sentinel.read_bytes(), b"keep")
        self.assertFalse((self.work_dir / "chunk_manifest.json").exists())

    def test_existing_nondirectory_work_path_is_rejected_without_deletion(self):
        self.work_dir.write_bytes(b"keep")

        with self.assertRaisesRegex(ChunkManifestError, "not a directory"):
            self._open()

        self.assertEqual(self.work_dir.read_bytes(), b"keep")

    def test_existing_malformed_or_unsupported_manifest_is_not_rewritten(self):
        malformed_documents = (
            b"not-json",
            b"{}",
            json.dumps(
                {
                    "kind": "fvdb-reality-capture-chunk-run",
                    "version": 1,
                    "run_signature": self.signature,
                    "chunk_plan": [],
                    "chunk_states": [],
                }
            ).encode("utf-8"),
            json.dumps(
                {
                    "kind": "fvdb-reality-capture-chunk-run",
                    "version": 3,
                    "run_signature": self.signature,
                    "chunk_plan": [],
                    "chunk_states": [],
                }
            ).encode("utf-8"),
        )

        for index, contents in enumerate(malformed_documents):
            with self.subTest(index=index):
                work_dir = self.root / f"malformed_{index}"
                work_dir.mkdir()
                manifest_path = work_dir / "chunk_manifest.json"
                manifest_path.write_bytes(contents)
                with self.assertRaises(ChunkManifestError):
                    ChunkRunManifest.open_or_create(work_dir, self.signature, self.plan)
                self.assertEqual(manifest_path.read_bytes(), contents)

    def test_complete_state_requires_valid_v2_integrity_fields(self):
        manifest = self._open()
        artifact = self._write_artifact("chunk_0000.ply")
        manifest.mark_complete("chunk_0000", artifact, 10, 9)
        original_manifest = manifest.path.read_bytes()
        base_document = json.loads(original_manifest)

        mutations = (
            ("missing artifact_size", lambda state: state.pop("artifact_size")),
            ("zero artifact_size", lambda state: state.update(artifact_size=0)),
            ("boolean artifact_size", lambda state: state.update(artifact_size=True)),
            ("missing artifact_sha256", lambda state: state.pop("artifact_sha256")),
            ("short artifact_sha256", lambda state: state.update(artifact_sha256="0" * 63)),
            ("uppercase artifact_sha256", lambda state: state.update(artifact_sha256="A" * 64)),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                document = json.loads(json.dumps(base_document))
                mutate(document["chunk_states"][0])
                invalid_manifest = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
                manifest.path.write_bytes(invalid_manifest)

                with self.assertRaises(ChunkManifestError):
                    self._open()

                self.assertEqual(manifest.path.read_bytes(), invalid_manifest)
                manifest.path.write_bytes(original_manifest)

    def test_mark_complete_requires_regular_nonempty_artifact(self):
        manifest = self._open()
        empty_file = self._write_artifact("empty.ply", b"")
        directory = self.work_dir / "directory.ply"
        directory.mkdir()
        target = self._write_artifact("target.ply")
        symlink = self.work_dir / "symlink.ply"
        symlink.symlink_to(target)

        for path in (self.work_dir / "missing.ply", empty_file, directory, symlink):
            with self.subTest(path=path.name):
                with self.assertRaisesRegex(ValueError, "regular nonempty"):
                    manifest.mark_complete("chunk_0000", path, 10, 9)
                self.assertEqual(manifest.state("chunk_0000").status, "pending")

    def test_mark_complete_validates_counts_before_writing(self):
        manifest = self._open()
        artifact = self._write_artifact("chunk_0000.ply")
        original_manifest = manifest.path.read_bytes()

        for initial_count, final_count in ((-1, 1), (1, -1), (True, 1), (1, 1.5)):
            with self.subTest(initial=initial_count, final=final_count):
                with self.assertRaises(ChunkManifestError):
                    manifest.mark_complete("chunk_0000", artifact, initial_count, final_count)
                self.assertEqual(manifest.path.read_bytes(), original_manifest)

    def test_resume_demotes_all_invalid_completed_artifacts_without_deleting_them(self):
        plan = plan_spatial_chunks((0.0, 0.0, 0.0, 4.0, 1.0, 1.0), (4, 1, 1), 0.1)
        manifest = self._open(plan=plan)
        artifacts = []
        for chunk in plan:
            artifact = self._write_artifact(f"{chunk.id}.ply")
            artifacts.append(artifact)
            manifest.mark_complete(chunk.id, artifact, 100 + chunk.index, 80 + chunk.index)

        artifacts[0].unlink()
        artifacts[1].write_bytes(b"")
        artifacts[2].unlink()
        artifacts[2].mkdir()
        artifacts[3].unlink()
        symlink_target = self.root / "outside.ply"
        symlink_target.write_bytes(b"target")
        artifacts[3].symlink_to(symlink_target)

        resumed = self._open(plan=plan)

        self.assertEqual(resumed.pending_chunks, plan)
        for chunk in plan:
            state = resumed.state(chunk.id)
            self.assertEqual(state.status, "pending")
            self.assertIsNone(state.artifact_path)
            self.assertIsNone(state.artifact_size)
            self.assertIsNone(state.artifact_sha256)
            self.assertIsNone(state.initial_count)
            self.assertIsNone(state.final_count)
        self.assertTrue(artifacts[2].is_dir())
        self.assertTrue(artifacts[3].is_symlink())
        self.assertEqual(symlink_target.read_bytes(), b"target")

    def test_validate_artifacts_demotes_stale_state_atomically(self):
        manifest = self._open()
        artifact = self._write_artifact("chunk_0000.ply")
        manifest.mark_complete("chunk_0000", artifact, 10, 9)
        artifact.unlink()

        demoted = manifest.validate_artifacts()

        self.assertEqual(demoted, ("chunk_0000",))
        self.assertEqual(manifest.state("chunk_0000").status, "pending")
        self.assertEqual(manifest.validate_artifacts(), ())

    def test_validate_artifacts_demotes_size_and_digest_mismatches_without_deleting_files(self):
        manifest = self._open()
        digest_mismatch = self._write_artifact("chunk_0000.ply", b"abcdefgh")
        size_mismatch = self._write_artifact("chunk_0001.ply", b"ijklmnop")
        manifest.mark_complete("chunk_0000", digest_mismatch, 10, 9)
        manifest.mark_complete("chunk_0001", size_mismatch, 11, 8)

        digest_mismatch.write_bytes(b"ABCDEFGH")
        size_mismatch.write_bytes(b"size changed")

        self.assertEqual(manifest.validate_artifacts(), ("chunk_0000", "chunk_0001"))
        for chunk_id in ("chunk_0000", "chunk_0001"):
            state = manifest.state(chunk_id)
            self.assertEqual(state.status, "pending")
            self.assertIsNone(state.artifact_path)
            self.assertIsNone(state.artifact_size)
            self.assertIsNone(state.artifact_sha256)
            self.assertIsNone(state.initial_count)
            self.assertIsNone(state.final_count)
        self.assertEqual(digest_mismatch.read_bytes(), b"ABCDEFGH")
        self.assertEqual(size_mismatch.read_bytes(), b"size changed")

        resumed = self._open()
        self.assertEqual(resumed.pending_chunks, self.plan)
        self.assertEqual(resumed.validate_artifacts(), ())

    def test_mark_pending_does_not_delete_artifact(self):
        manifest = self._open()
        artifact = self._write_artifact("chunk_0000.ply")
        manifest.mark_complete("chunk_0000", artifact, 10, 9)

        state = manifest.mark_pending("chunk_0000")

        self.assertEqual(state.status, "pending")
        self.assertIsNone(state.artifact_path)
        self.assertIsNone(state.artifact_size)
        self.assertIsNone(state.artifact_sha256)
        self.assertIsNone(state.initial_count)
        self.assertIsNone(state.final_count)
        self.assertEqual(artifact.read_bytes(), b"ply-data")
        resumed = self._open()
        self.assertEqual(resumed.state("chunk_0000").status, "pending")

    def test_writes_use_same_directory_temporary_file_and_os_replace(self):
        calls = []
        events = []
        real_replace = os.replace

        def checked_replace(source, destination):
            source_path = pathlib.Path(source)
            destination_path = pathlib.Path(destination)
            self.assertEqual(source_path.parent, self.work_dir)
            self.assertEqual(destination_path, self.work_dir / "chunk_manifest.json")
            self.assertTrue(source_path.is_file())
            calls.append((source_path, destination_path))
            real_replace(source, destination)
            events.append("replace")

        def checked_fsync_directory(path):
            self.assertEqual(path, self.work_dir)
            self.assertTrue((self.work_dir / "chunk_manifest.json").is_file())
            events.append("directory fsync")

        with (
            mock.patch("fvdb_reality_capture.chunk_manifest.os.replace", side_effect=checked_replace),
            mock.patch(
                "fvdb_reality_capture.chunk_manifest._fsync_directory", side_effect=checked_fsync_directory
            ) as fsync_directory,
        ):
            self._open()

        self.assertEqual(len(calls), 1)
        fsync_directory.assert_called_once_with(self.work_dir)
        self.assertEqual(events, ["replace", "directory fsync"])
        self.assertEqual(list(self.work_dir.glob(".chunk_manifest.json.*.tmp")), [])

    def test_failed_atomic_replace_preserves_manifest_and_in_memory_state(self):
        manifest = self._open()
        artifact = self._write_artifact("chunk_0000.ply")
        original_manifest = manifest.path.read_bytes()

        with mock.patch("fvdb_reality_capture.chunk_manifest.os.replace", side_effect=OSError("replace failed")):
            with self.assertRaisesRegex(OSError, "replace failed"):
                manifest.mark_complete("chunk_0000", artifact, 10, 9)

        self.assertEqual(manifest.path.read_bytes(), original_manifest)
        self.assertEqual(manifest.state("chunk_0000").status, "pending")
        self.assertEqual(list(self.work_dir.glob(".chunk_manifest.json.*.tmp")), [])

    def test_unknown_chunk_id_is_rejected_without_writing(self):
        manifest = self._open()
        artifact = self._write_artifact("chunk.ply")
        original_manifest = manifest.path.read_bytes()

        with self.assertRaises(KeyError):
            manifest.mark_complete("unknown", artifact, 10, 9)

        self.assertEqual(manifest.path.read_bytes(), original_manifest)


class ChunkRunLockTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary_directory.name)
        self.work_dir = self.root / "nested" / "chunk_work"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_lock_is_atomic_and_records_actionable_owner_metadata(self):
        lock = ChunkRunLock(self.work_dir)
        contender = ChunkRunLock(self.work_dir)

        with lock as acquired:
            self.assertIs(acquired, lock)
            self.assertTrue(lock.path.is_file())
            self.assertFalse(self.work_dir.exists())
            metadata = json.loads(lock.path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["kind"], "fvdb-reality-capture-chunk-run-lock")
            self.assertEqual(metadata["version"], 1)
            self.assertEqual(metadata["pid"], os.getpid())
            self.assertEqual(metadata["hostname"], socket.gethostname())
            self.assertEqual(metadata["work_dir"], str(self.work_dir.resolve()))
            self.assertTrue(metadata["created_at_utc"])
            self.assertEqual(lock.metadata, metadata)
            original_lock = lock.path.read_bytes()

            with self.assertRaises(ChunkRunLockError) as raised:
                contender.acquire()

            message = str(raised.exception)
            self.assertIn(str(lock.path), message)
            self.assertIn(f"pid={os.getpid()}", message)
            self.assertIn(f"hostname={socket.gethostname()}", message)
            self.assertIn("remove the lock file manually", message)
            self.assertEqual(lock.path.read_bytes(), original_lock)

        self.assertFalse(lock.path.exists())
        self.assertIsNone(lock.metadata)
        with contender:
            self.assertTrue(contender.path.is_file())
        self.assertFalse(contender.path.exists())

    def test_lock_can_be_acquired_before_creating_manifest(self):
        plan = plan_spatial_chunks((0.0, 0.0, 0.0, 1.0, 1.0, 1.0), (1, 1, 1), 0.0)
        lock = ChunkRunLock(self.work_dir)

        with lock:
            self.assertFalse(self.work_dir.exists())
            manifest = ChunkRunManifest.open_or_create(self.work_dir, {"dataset": "/data/scene"}, plan)
            self.assertTrue(lock.path.is_file())
            self.assertTrue(manifest.path.is_file())

        self.assertFalse(lock.path.exists())
        self.assertTrue(manifest.path.is_file())

    def test_context_releases_lock_when_body_raises(self):
        lock = ChunkRunLock(self.work_dir)

        with self.assertRaisesRegex(RuntimeError, "training failed"):
            with lock:
                self.assertTrue(lock.path.is_file())
                raise RuntimeError("training failed")

        self.assertFalse(lock.path.exists())
        with ChunkRunLock(self.work_dir):
            pass

    def test_unknown_existing_lock_is_never_silently_stolen(self):
        lock = ChunkRunLock(self.work_dir)
        lock.path.parent.mkdir(parents=True)
        lock.path.write_bytes(b"not lock metadata")
        original_lock = lock.path.read_bytes()

        with self.assertRaises(ChunkRunLockError) as raised:
            lock.acquire()

        message = str(raised.exception)
        self.assertIn("metadata unavailable", message)
        self.assertIn("Another reconstruction may still be active", message)
        self.assertIn("remove the lock file manually", message)
        self.assertEqual(lock.path.read_bytes(), original_lock)

    def test_release_does_not_remove_a_replacement_lock(self):
        lock = ChunkRunLock(self.work_dir).acquire()
        lock.path.unlink()
        lock.path.write_bytes(b"replacement lock")

        with self.assertRaisesRegex(ChunkRunLockError, "changed while held"):
            lock.release()

        self.assertEqual(lock.path.read_bytes(), b"replacement lock")

    def test_failed_metadata_write_removes_new_lock(self):
        lock = ChunkRunLock(self.work_dir)

        with mock.patch("fvdb_reality_capture.chunk_manifest.os.fsync", side_effect=OSError("fsync failed")):
            with self.assertRaisesRegex(OSError, "fsync failed"):
                lock.acquire()

        self.assertFalse(lock.path.exists())
        self.assertIsNone(lock.metadata)


if __name__ == "__main__":
    unittest.main()
