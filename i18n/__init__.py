"""
Locale loader for i18n support.

Usage:
    import i18n
    i18n.init("en")          # call once at startup

    from i18n import T
    print(T.BOT_NAME)        # "Xiao You"

Default language is Chinese (zh).
"""
import importlib
from types import ModuleType

T: ModuleType | None = None


def init(lang: str = "zh") -> None:
    """Load the locale module for the given language code."""
    global T
    T = importlib.import_module(f"i18n.{lang}")
