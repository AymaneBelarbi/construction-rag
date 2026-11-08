import unittest
from pathlib import Path

from tests.support.db_support import skip_unless

SHA = "7e0e2ac4ed7e5c54c34ff24bd19506f0316ca7ac"


class SkipUnlessTests(unittest.TestCase):
    """CI must fail, not quietly skip, when the database or the sample data is missing."""

    def test_a_missing_requirement_skips_the_test_in_ordinary_runs(self) -> None:
        @skip_unless(False, "no database", strict=False)
        class Wrapped(unittest.TestCase):
            def test_it(self) -> None: ...

        self.assertTrue(getattr(Wrapped, "__unittest_skip__", False))
        self.assertEqual(Wrapped.__unittest_skip_why__, "no database")  # type: ignore[attr-defined]

    def test_a_met_requirement_never_skips(self) -> None:
        @skip_unless(True, "unused", strict=False)
        class Wrapped(unittest.TestCase):
            def test_it(self) -> None: ...

        self.assertFalse(getattr(Wrapped, "__unittest_skip__", False))

    def test_in_strict_mode_a_missing_requirement_is_not_hidden_by_a_skip(self) -> None:
        @skip_unless(False, "no database", strict=True)
        class Wrapped(unittest.TestCase):
            def test_it(self) -> None: ...

        self.assertFalse(getattr(Wrapped, "__unittest_skip__", False))  # it runs, and fails on the missing database


class ContinuousIntegrationSetupTests(unittest.TestCase):
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    def test_ci_runs_in_strict_mode(self) -> None:
        self.assertIn('RAG_REQUIRE_FULL: "1"', self.workflow)

    def test_ci_and_the_readme_use_the_same_pinned_commit_of_the_records_repo(self) -> None:
        self.assertIn(f"ref: {SHA}", self.workflow)
        self.assertIn(SHA, Path("README.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
