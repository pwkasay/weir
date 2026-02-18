"""Example: Air Quality Data Ingestion Pipeline

A realistic pipeline that demonstrates weir's core features:
- Concurrent API fetching with rate limiting via concurrency control
- Validation with error routing
- Simulated database persistence
- Automatic metrics and observability

This uses simulated data so it runs without any external dependencies,
but the structure mirrors a real ingestion pipeline.

Run with:
    python -m examples.air_quality
"""

import asyncio
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timezone

from weir import DeadLetterCollector, Pipeline, stage


# ── Domain Models ──


@dataclass
class StationConfig:
    station_id: str
    name: str
    lat: float
    lon: float


@dataclass
class RawReading:
    station_id: str
    pm25: float
    pm10: float
    ozone: float
    timestamp: datetime


@dataclass
class ValidatedReading:
    station_id: str
    pm25: float
    pm10: float
    ozone: float
    timestamp: datetime
    aqi_category: str


# ── Custom Errors ──


class InvalidReading(Exception):
    """A sensor reading that fails validation."""

    pass


class APIError(Exception):
    """Simulated API error (transient)."""

    pass


# ── Pipeline Stages ──


@stage(concurrency=5, retries=3, timeout=10, retry_base_delay=0.1, retryable_errors=(APIError,))
async def fetch_reading(station: StationConfig) -> RawReading:
    """Fetch a reading from the air quality API.

    Simulates network latency and occasional transient failures.
    In a real pipeline, this would hit an actual API endpoint.
    """
    # Simulate network latency
    await asyncio.sleep(random.uniform(0.05, 0.2))

    # Simulate occasional transient API errors (~10% failure rate)
    if random.random() < 0.10:
        raise APIError(f"Timeout fetching station {station.station_id}")

    # Generate simulated sensor data
    return RawReading(
        station_id=station.station_id,
        pm25=random.uniform(-5, 300),   # Intentionally can be negative (bad sensor)
        pm10=random.uniform(0, 500),
        ozone=random.uniform(0, 200),
        timestamp=datetime.now(timezone.utc),
    )


@stage(concurrency=10)
async def validate(reading: RawReading) -> ValidatedReading:
    """Validate and categorize a sensor reading.

    Rejects physically impossible values and assigns AQI categories.
    Invalid readings raise InvalidReading and get routed to dead letters.
    """
    # Physical validation
    if reading.pm25 < 0:
        raise InvalidReading(
            f"Negative PM2.5 ({reading.pm25}) from station {reading.station_id}"
        )
    if reading.pm10 < 0:
        raise InvalidReading(
            f"Negative PM10 ({reading.pm10}) from station {reading.station_id}"
        )

    # AQI categorization (simplified EPA breakpoints for PM2.5)
    if reading.pm25 <= 12:
        category = "Good"
    elif reading.pm25 <= 35.4:
        category = "Moderate"
    elif reading.pm25 <= 55.4:
        category = "Unhealthy (Sensitive)"
    elif reading.pm25 <= 150.4:
        category = "Unhealthy"
    elif reading.pm25 <= 250.4:
        category = "Very Unhealthy"
    else:
        category = "Hazardous"

    return ValidatedReading(
        station_id=reading.station_id,
        pm25=reading.pm25,
        pm10=reading.pm10,
        ozone=reading.ozone,
        timestamp=reading.timestamp,
        aqi_category=category,
    )


@stage(concurrency=2)
async def persist(reading: ValidatedReading) -> None:
    """Persist a validated reading to the database.

    Simulates a slow database write. Low concurrency (2) prevents
    overwhelming the DB connection pool — this is where backpressure
    naturally kicks in.
    """
    # Simulate database write latency
    await asyncio.sleep(random.uniform(0.02, 0.08))


# ── Source Data ──


def generate_stations(n: int = 50) -> list[StationConfig]:
    """Generate simulated monitoring stations."""
    return [
        StationConfig(
            station_id=f"AQ-{i:04d}",
            name=f"Station {i}",
            lat=40.0 + random.uniform(-2, 2),
            lon=-74.0 + random.uniform(-2, 2),
        )
        for i in range(n)
    ]


# ── Main ──


async def main() -> None:
    dead_letters = DeadLetterCollector()

    stations = generate_stations(200)

    pipe = (
        Pipeline(
            "air-quality-ingest",
            channel_capacity=32,
            drain_timeout=10.0,
            log_level=logging.INFO,
        )
        .source(stations)
        .then(fetch_reading)
        .then(validate)
        .then(persist)
        .on_error(InvalidReading, dead_letters)
        .build()
    )

    print(f"\nTopology:\n{pipe.topology}\n")

    result = await pipe.run()

    print(f"\n{result.summary()}\n")

    if dead_letters.items:
        print(f"Dead letters ({dead_letters.count}):")
        for dl in dead_letters.items[:5]:
            print(f"  {dl.stage_name}: {dl.error}")
        if dead_letters.count > 5:
            print(f"  ... and {dead_letters.count - 5} more")


if __name__ == "__main__":
    asyncio.run(main())
