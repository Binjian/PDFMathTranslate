"""Tests for GUI translation adapters."""

import unittest
from unittest.mock import MagicMock

from pdf2zh.gui_progress import update_gui_progress


class TestGuiProgress(unittest.TestCase):
    def test_precise_progress_event(self):
        progress = MagicMock()

        update_gui_progress(
            progress,
            {"stage": "Translating", "overall_progress": 25.0},
        )

        progress.assert_called_once_with(0.25, desc="Translating")

    def test_legacy_tqdm_progress(self):
        progress = MagicMock()
        event = MagicMock(n=2, total=4, desc="")

        update_gui_progress(progress, event)

        progress.assert_called_once_with(0.5, desc="Translating...")

    def test_progress_is_clamped(self):
        progress = MagicMock()

        update_gui_progress(progress, {"overall_progress": 125.0})

        progress.assert_called_once_with(1.0, desc="Translating...")


if __name__ == "__main__":
    unittest.main()
