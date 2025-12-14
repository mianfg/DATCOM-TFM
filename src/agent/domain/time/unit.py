import re
from enum import StrEnum


class TimeUnit(StrEnum):
    us = "us"  # microseconds
    ms = "ms"
    s = "s"
    m = "m"
    h = "h"
    d = "d"
    y = "y"

    @classmethod
    def get_regex_pattern(cls) -> str:
        return "|".join(re.escape(unit.value) for unit in cls)
