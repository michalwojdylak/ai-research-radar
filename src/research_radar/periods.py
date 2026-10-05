from collections.abc import Iterator
from datetime import UTC, date, datetime, time, timedelta

from research_radar.models.paper import RadarModel


def utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class WeekPeriod(RadarModel):
    id: str
    year: int
    week: int
    start_date: date
    end_date: date

    @property
    def start(self) -> datetime:
        return datetime.combine(self.start_date, time.min, UTC)

    @property
    def end(self) -> datetime:
        return datetime.combine(self.end_date + timedelta(days=1), time.min, UTC)

    @property
    def month_key(self) -> tuple[int, int]:
        thursday = self.start_date + timedelta(days=3)
        return thursday.year, thursday.month


def week_for(value: date | datetime) -> WeekPeriod:
    day = utc(value).date() if isinstance(value, datetime) else value
    year, week, _ = day.isocalendar()
    return WeekPeriod(
        id=f"{year:04}-W{week:02}",
        year=year,
        week=week,
        start_date=date.fromisocalendar(year, week, 1),
        end_date=date.fromisocalendar(year, week, 7),
    )


def parse_week(identifier: str) -> WeekPeriod:
    import re

    match = re.fullmatch(r"(\d{4})-W(\d{2})", identifier)
    if match is None:
        raise ValueError("Use an ISO week ID such as 2024-W08")
    return week_for(date.fromisocalendar(int(match[1]), int(match[2]), 1))


def previous_week(now: datetime) -> WeekPeriod:
    return week_for(week_for(now).start_date - timedelta(days=1))


def iter_weeks(start: date, end: date) -> Iterator[WeekPeriod]:
    if start > end:
        raise ValueError("Start date must not follow end date")
    cursor = week_for(start).start_date
    while cursor <= end:
        yield week_for(cursor)
        cursor += timedelta(days=7)
