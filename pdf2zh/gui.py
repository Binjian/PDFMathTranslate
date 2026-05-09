"""FastHTML GUI entry point."""

from pdf2zh.gui_fasthtml import *  # noqa: F401,F403


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    setup_gui()
