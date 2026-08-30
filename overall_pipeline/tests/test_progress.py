from __future__ import annotations

import io
import unittest

from text2sql.core.progress import ProgressReporter


class ProgressReporterTests(unittest.TestCase):
    def test_plain_progress_is_periodic_and_finishes_at_total(self) -> None:
        stream = io.StringIO()
        current_time = [0.0]
        reporter = ProgressReporter(
            enabled=True,
            interval_seconds=5.0,
            stream=stream,
            clock=lambda: current_time[0],
            force_plain=True,
        )
        reporter.start("generation", "LLM generation", 10, initial=2)
        current_time[0] = 1.0
        reporter.update(3)
        current_time[0] = 5.0
        reporter.update(4)
        reporter.message("worker-00 is ready")
        current_time[0] = 10.0
        reporter.update(10)
        reporter.finish()

        output = stream.getvalue()
        self.assertIn("LLM generation: 2/10 (20.0%)", output)
        self.assertNotIn("LLM generation: 3/10", output)
        self.assertIn("LLM generation: 4/10 (40.0%)", output)
        self.assertIn("worker-00 is ready", output)
        self.assertEqual(output.count("LLM generation: 10/10"), 1)

    def test_disabled_progress_is_silent(self) -> None:
        stream = io.StringIO()
        reporter = ProgressReporter(enabled=False, stream=stream)
        reporter.start("sql", "SQL execution", 2)
        reporter.update(2)
        reporter.message("ignored")
        reporter.finish()
        self.assertEqual(stream.getvalue(), "")

    def test_progress_rejects_decreasing_updates(self) -> None:
        reporter = ProgressReporter(enabled=False)
        reporter.start("official", "Official evaluation", 8, initial=4)
        with self.assertRaises(ValueError):
            reporter.update(3)


if __name__ == "__main__":
    unittest.main()
