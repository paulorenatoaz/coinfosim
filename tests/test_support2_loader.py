import csv
import hashlib
from pathlib import Path


RAW_DIR = Path("data/raw/support2")
RAW_PATH = RAW_DIR / "support2.csv"
EXPECTED_SHA256 = "79621945edf2a5c8dc36359684ff356d3c6025e773ba4fefac26f865f7894c78"
CHANNELS = ("meanbp", "hrt", "resp", "temp", "wblc", "crea", "sod")


def test_canonical_support2_file_structure_and_hash():
    assert RAW_PATH.exists()
    assert hashlib.sha256(RAW_PATH.read_bytes()).hexdigest() == EXPECTED_SHA256

    with RAW_PATH.open(newline="", encoding="utf-8") as handle:
        rows = csv.reader(handle)
        header = next(rows)
        data_rows = list(rows)

    assert len(header) == 47
    assert len(set(header)) == 47
    assert len(data_rows) == 9_105
    assert {len(row) for row in data_rows} == {48}
    assert {"death", "d.time", "dzgroup", *CHANNELS}.issubset(header)
