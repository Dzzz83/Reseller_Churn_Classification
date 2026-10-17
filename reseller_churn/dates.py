"""Calendar-date parsing shared by source validation and prediction stages."""

from datetime import date as CalendarDate
from datetime import datetime

import numpy as np
import pandas as pd

DateLike = str | CalendarDate | datetime | np.datetime64 | pd.Timestamp


def parse_calendar_date(value: DateLike) -> pd.Timestamp:
    """Parse a midnight calendar date without a timezone."""
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is not None or result != result.normalize():
        raise ValueError("Dates must be non-missing, timezone-naive calendar dates.")
    return result
