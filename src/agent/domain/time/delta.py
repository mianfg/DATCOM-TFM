import re
from datetime import timedelta
from typing import Any, Self

from pydantic_core import CoreSchema, core_schema

from src.agent.domain.time.unit import TimeUnit


class TimeDelta:
    __REGEX_PATTERN = rf"(\d+(?:\.\d+)?)({TimeUnit.get_regex_pattern()})"

    delta: timedelta

    def __init__(self, delta: str | timedelta, strict: bool = False) -> None:
        if isinstance(delta, str):
            self.delta = self.__class__.__parse_delta_str(delta, strict)
        else:
            self.delta = delta

    @classmethod
    def __parse_delta_str(cls, delta_str: str, strict: bool = False) -> timedelta:
        delta_str = delta_str.strip()

        if delta_str.startswith("-"):
            negative = True
            delta_str = delta_str[1:]
        else:
            negative = False

        delta_str = delta_str.strip()

        if not delta_str or delta_str == "0":
            return timedelta()

        matches = re.findall(cls.__REGEX_PATTERN, delta_str)

        if not matches:
            raise ValueError(f"Invalid time point format: {delta_str}")

        if strict:
            consumed = "".join(f"{value}{unit}" for value, unit in matches)
            if consumed != delta_str:
                raise ValueError(f"Invalid characters in time specification: {delta_str}")

        delta = timedelta()

        for value, unit in matches:
            value = float(value)
            if unit == "us":
                delta += timedelta(microseconds=value)
            elif unit == "ms":
                delta += timedelta(milliseconds=value)
            elif unit == "s":
                delta += timedelta(seconds=value)
            elif unit == "m":
                delta += timedelta(minutes=value)
            elif unit == "h":
                delta += timedelta(hours=value)
            elif unit == "d":
                delta += timedelta(days=value)
            elif unit == "y":
                delta += timedelta(days=value * 365)

        return (-1 if negative else 1) * delta

    def delta_to_units(self) -> dict[TimeUnit, int]:
        total_seconds = abs(int(self.delta.total_seconds()))
        total_microseconds = abs(int(self.delta.microseconds))

        return {
            TimeUnit.y: total_seconds // 31536000,
            TimeUnit.d: (total_seconds // 86400) % 365,
            TimeUnit.h: (total_seconds // 3600) % 24,
            TimeUnit.m: (total_seconds // 60) % 60,
            TimeUnit.s: total_seconds % 60,
            TimeUnit.ms: total_microseconds // 1000,
            TimeUnit.us: total_microseconds % 1000,
        }

    def __delta_to_str(self) -> str:
        return "".join(f"{value}{unit}" for unit, value in self.delta_to_units().items() if value > 0)

    def __str__(self) -> str:
        if delta_str := self.__delta_to_str():
            return f"{'-' if self.delta < timedelta() else ''}{delta_str}"
        else:
            return "0"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__str__()})"

    def __add__(self, other: Self) -> Self:
        return self.__class__(self.delta + other.delta)

    def __sub__(self, other: Self) -> Self:
        return self.__class__(self.delta - other.delta)

    def __ge__(self, other: Self) -> bool:
        return self.delta >= other.delta

    def __gt__(self, other: Self) -> bool:
        return self.delta > other.delta

    def __le__(self, other: Self) -> bool:
        return self.delta <= other.delta

    def __lt__(self, other: Self) -> bool:
        return self.delta < other.delta

    def __eq__(self, other: Self) -> bool:
        return self.delta == other.delta

    def __ne__(self, other: Self) -> bool:
        return self.delta != other.delta

    def __hash__(self) -> int:
        return hash(self.delta)

    def __bool__(self) -> bool:
        return self.delta != timedelta()

    def __int__(self) -> int:
        return int(self.delta.total_seconds())

    def __float__(self) -> float:
        return float(self.delta.total_seconds())

    def __round__(self, n: int = 0) -> Self:
        return self.__class__(self.delta.total_seconds())

    def __floor__(self) -> Self:
        return self.__class__(self.delta.total_seconds())

    def __ceil__(self) -> Self:
        return self.__class__(self.delta.total_seconds())

    def __abs__(self) -> Self:
        return self.__class__(abs(self.delta))

    def __neg__(self) -> Self:
        return self.__class__(-self.delta)

    def __pos__(self) -> Self:
        return self

    def __invert__(self) -> Self:
        return self.__class__(-self.delta)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> CoreSchema:
        return core_schema.no_info_after_validator_function(
            cls._validate,
            core_schema.union_schema(
                [
                    core_schema.is_instance_schema(cls),
                    core_schema.str_schema(),
                    core_schema.is_instance_schema(timedelta),
                ]
            ),
            serialization=core_schema.to_string_ser_schema(),
        )

    @classmethod
    def _validate(cls, value: Any) -> Self:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(value)
        if isinstance(value, timedelta):
            return cls(value)
        raise ValueError(f"Cannot convert {type(value)} to TimeDelta")
