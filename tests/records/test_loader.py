import tempfile
import unittest
from pathlib import Path

from rag.records.loader import load_csv


class LoadCsvTests(unittest.TestCase):
    def write(self, content: bytes) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "rfis.csv"
        path.write_bytes(content)
        return path

    def test_rows_come_back_as_dicts_keyed_by_the_header(self) -> None:
        rows = load_csv(self.write(b"project_id,rfi_number\nP-1,RFI-001\n"))
        self.assertEqual(rows, [{"project_id": "P-1", "rfi_number": "RFI-001"}])

    def test_a_byte_order_mark_from_a_spreadsheet_export_does_not_rename_the_first_column(self) -> None:
        rows = load_csv(self.write("\ufeffproject_id,rfi_number\nP-1,RFI-001\n".encode("utf-8")))
        self.assertEqual(list(rows[0]), ["project_id", "rfi_number"])


if __name__ == "__main__":
    unittest.main()
