"""Time-of-day formatting helpers shared by the GUI and the analytics
exporter. The simulation stores ``time_of_day`` as a float in [0, 24)
representing hours since midnight; presentation uses 12-hour am/pm.

Examples (24h -> formatted):
    0.0   -> "12am"
    0.5   -> "12:30am"
    9.0   -> "9am"
    11.99 -> "11:59am"      (minute rounding never spills past 11:59 here)
    12.0  -> "12pm"
    13.25 -> "1:15pm"
    23.0  -> "11pm"
    23.99 -> "11:59pm"

We round minutes to the nearest integer and carry over to the next hour
when rounding hits 60, so 23.999h prints as ``"12am"`` (next day's midnight).
"""

from __future__ import annotations


def format_time_ampm(hours: float, *, with_minutes: bool | None = None) -> str:
    """Render an hours-since-midnight float in 12-hour am/pm style.

    Parameters
    ----------
    hours:
        Float in [0, 24); values outside the range are wrapped mod 24.
    with_minutes:
        Force inclusion / omission of the ``:MM`` part. Default ``None``
        means "include only when the time isn't on the hour", which keeps
        labels short while still showing 11:30am etc.
    """
    h = hours % 24
    whole = int(h)
    minutes = int(round((h - whole) * 60))
    if minutes == 60:
        whole = (whole + 1) % 24
        minutes = 0

    suffix = "am" if whole < 12 else "pm"
    h12 = whole % 12
    if h12 == 0:
        h12 = 12

    show_minutes = (
        with_minutes
        if with_minutes is not None
        else minutes != 0
    )
    if show_minutes:
        return f"{h12}:{minutes:02d}{suffix}"
    return f"{h12}{suffix}"
