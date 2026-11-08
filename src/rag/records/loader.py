import csv
from pathlib import Path


def load_csv(path:Path) -> list[dict[str,str]]:
    with open(path,newline="",encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)
