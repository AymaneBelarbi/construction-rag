import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rag.providers.embed import hashing_embed
from rag.records.chunker import Chunk
from rag.storage.store import build_index, chunks_fingerprint, load_index, load_or_build, save_index

SAMPLE = Path("data/records/samples/library-renovation")


class CountingEmbedder:
    """Wraps hashing_embed and counts how many texts it was asked to embed."""

    def __init__(self) -> None:
        self.texts_embedded = 0

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.texts_embedded += len(texts)
        return hashing_embed(texts)


class StoreTests(unittest.TestCase):
    def test_every_row_becomes_one_chunk_with_one_vector(self) -> None:
        index = build_index(SAMPLE, hashing_embed, "hashing")
        self.assertEqual(len(index.chunks), 14 + 11 + 7 + 32 + 24)
        self.assertEqual(len(index.vectors), len(index.chunks))

    def test_get_returns_all_revisions_of_a_submittal(self) -> None:
        index = build_index(SAMPLE, hashing_embed, "hashing")
        self.assertEqual(len(index.get("RFI-003")), 1)
        self.assertEqual(len(index.get("S-001")), 2)
        self.assertEqual(index.get("RFI-999"), [])

    def test_rfi_and_change_event_are_linked_both_ways(self) -> None:
        index = build_index(SAMPLE, hashing_embed, "hashing")
        self.assertIn("CE-001", [c.record_id for c in index.linked("RFI-003")])
        self.assertIn("RFI-003", [c.record_id for c in index.linked("CE-001")])

    def test_second_load_reuses_cache_without_embedding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "index.json"
            first = CountingEmbedder()
            load_or_build(SAMPLE, first, "counting", cache)
            self.assertEqual(first.texts_embedded, 88)
            second = CountingEmbedder()
            index = load_or_build(SAMPLE, second, "counting", cache)
            self.assertEqual(second.texts_embedded, 0)
            self.assertEqual(len(index.chunks), 88)

    def test_changed_csv_triggers_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "project"
            shutil.copytree(SAMPLE, folder)
            cache = Path(tmp) / "index.json"
            load_or_build(folder, hashing_embed, "hashing", cache)
            with open(folder / "rfis.csv", "a", encoding="utf-8") as f:
                f.write("P-2041,RFI-099,New,Question?,,,2026-09-01,,,,open,,,none,none,Me\n")
            embedder = CountingEmbedder()
            index = load_or_build(folder, embedder, "hashing", cache)
            self.assertEqual(embedder.texts_embedded, 89)
            self.assertEqual(len(index.get("RFI-099")), 1)

    def test_only_the_newest_submittal_revision_is_latest(self) -> None:
        index = build_index(SAMPLE, hashing_embed, "hashing")
        flags = {c.text.split(" revision ")[1][:1]: c.metadata["is_latest"] for c in index.get("S-001")}
        self.assertEqual(flags, {"0": "no", "1": "yes"})

    def test_missing_folder_and_empty_folder_are_errors_not_empty_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                build_index(Path(tmp) / "typo", hashing_embed, "hashing")
            with self.assertRaises(ValueError):
                build_index(Path(tmp), hashing_embed, "hashing")

    def test_fingerprint_changes_when_only_metadata_changes(self) -> None:
        a = Chunk("rfi", "RFI-001", "same text", {"status": "open"})
        b = Chunk("rfi", "RFI-001", "same text", {"status": "closed"})
        self.assertNotEqual(chunks_fingerprint([a]), chunks_fingerprint([b]))

    def test_links_are_rebuilt_from_chunks_when_loading_a_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "index.json"
            load_or_build(SAMPLE, hashing_embed, "hashing", cache)
            data = json.loads(cache.read_text(encoding="utf-8"))
            data["links"] = {"STALE": ["X"]}  # an old cache with outdated links
            cache.write_text(json.dumps(data), encoding="utf-8")
            index = load_index(cache)
            self.assertIn("CE-001", [c.record_id for c in index.linked("RFI-003")])
            self.assertEqual(index.linked("STALE"), [])

    def test_cache_with_wrong_vector_count_is_rebuilt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "index.json"
            load_or_build(SAMPLE, hashing_embed, "hashing", cache)
            data = json.loads(cache.read_text(encoding="utf-8"))
            data["vectors"].pop()
            cache.write_text(json.dumps(data), encoding="utf-8")
            embedder = CountingEmbedder()
            index = load_or_build(SAMPLE, embedder, "hashing", cache)
            self.assertEqual(embedder.texts_embedded, 88)
            self.assertEqual(len(index.vectors), 88)

    def test_failure_before_the_temp_file_exists_leaves_the_old_cache_intact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "index.json"
            good = build_index(SAMPLE, hashing_embed, "hashing")
            save_index(good, cache)
            before = cache.read_bytes()
            with mock.patch("rag.storage.store.json.dumps", side_effect=RuntimeError("cannot serialize")):
                with self.assertRaises(RuntimeError):
                    save_index(good, cache)
            self.assertEqual(cache.read_bytes(), before)
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["index.json"])

    def test_failure_after_the_temp_file_exists_removes_it_and_keeps_the_old_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "index.json"
            good = build_index(SAMPLE, hashing_embed, "hashing")
            save_index(good, cache)
            before = cache.read_bytes()
            with mock.patch("rag.storage.store.os.replace", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    save_index(good, cache)
            self.assertEqual(cache.read_bytes(), before)
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["index.json"])  # temp file cleaned up

    def test_different_embedder_id_triggers_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "index.json"
            load_or_build(SAMPLE, hashing_embed, "model-a", cache)
            embedder = CountingEmbedder()
            load_or_build(SAMPLE, embedder, "model-b", cache)
            self.assertEqual(embedder.texts_embedded, 88)


if __name__ == "__main__":
    unittest.main()
