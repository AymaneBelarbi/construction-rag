"""The exact evidence exposed to the answer model by a tool call."""
import hashlib
import json
from dataclasses import dataclass


def _identifier(values: tuple[object, ...]) -> str:
    return hashlib.sha256(json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RecordEvidence:
    record_id: str
    record_type: str
    revision: str
    dataset_version: str | None
    text: str
    complete: bool

    @property
    def evidence_id(self) -> str:
        return _identifier((self.dataset_version, self.record_type, self.record_id, self.revision, self.text, self.complete))


@dataclass(frozen=True)
class ToolEvidence:
    tool: str
    dataset_version: str | None
    text: str
    truncated: bool
    records: tuple[RecordEvidence, ...] = ()
    computed: str = ""
    computed_id: str | None = None

    @property
    def evidence_id(self) -> str:
        return _identifier((self.dataset_version, self.tool, self.text, self.truncated))
