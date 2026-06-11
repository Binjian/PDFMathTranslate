import logging
from pathlib import Path

log = logging.getLogger(__name__)

try:
    from dotenv import find_dotenv, load_dotenv
except ModuleNotFoundError:
    load_dotenv = None
else:
    load_dotenv(find_dotenv(usecwd=True), override=False)
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

__version__ = "1.9.11"
__author__ = "Byaidu"
__all__ = ["translate", "translate_stream"]


def __getattr__(name):
    if name in {"translate", "translate_stream"}:
        from pdf2zh.high_level import translate, translate_stream

        return {"translate": translate, "translate_stream": translate_stream}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
