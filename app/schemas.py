"""Pydantic request/response models for the ParkCast SF API."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class BlocksRequest(BaseModel):
    lat: float = Field(..., description="Destination latitude")
    lon: float = Field(..., description="Destination longitude")
    radius_meters: int = Field(500, ge=100, le=2000)
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    month: int = Field(..., ge=1, le=12)
    is_raining: int = Field(0, ge=0, le=1)
    is_holiday: int = Field(0, ge=0, le=1)
    is_school_day: int = Field(1, ge=0, le=1)
    temperature: float = Field(60.0)
    event_intensity: float = Field(0.0, ge=0.0, le=1.0)
    minutes_away: int = Field(0, ge=0, le=120)

    class Config:
        json_schema_extra = {
            "example": {
                "lat": 37.7816,
                "lon": -122.3975,
                "radius_meters": 500,
                "hour": 19,
                "day_of_week": 4,
                "month": 4,
                "is_raining": 0,
                "is_holiday": 0,
                "is_school_day": 1,
                "temperature": 62.5,
                "event_intensity": 0.6,
                "minutes_away": 20,
            }
        }


class BlockPrediction(BaseModel):
    lat: float
    lon: float
    street: Optional[str] = None
    neighborhood: Optional[str] = None
    total_spaces: int
    distance_meters: int
    predicted_occupancy_pct: float
    available_spaces_estimate: int
    demand_level: str
    color: str
    # "metered"  — block has SFpark training labels, prediction is direct
    # "inferred" — non-metered blockface, prediction via neighborhood baseline
    coverage: str = "metered"


class BlocksResponse(BaseModel):
    destination_lat: float
    destination_lon: float
    radius_meters: int
    predicted_at_hour: int
    minutes_away: int
    total_blocks_found: int
    blocks: List[BlockPrediction]


class ParkingInput(BaseModel):
    """Aggregate neighborhood-level prediction (backward-compat with v2)."""

    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    month: int = Field(..., ge=1, le=12)
    neighborhood: str = Field(...)
    total_spaces: int = Field(40, ge=1)
    is_raining: int = Field(0, ge=0, le=1)
    is_holiday: int = Field(0, ge=0, le=1)
    is_school_day: int = Field(1, ge=0, le=1)
    temperature: float = Field(60.0)
    event_intensity: float = Field(0.0, ge=0.0, le=1.0)


class ParkingPrediction(BaseModel):
    neighborhood: str
    hour: int
    day_of_week: int
    predicted_occupancy_pct: float
    available_spaces_estimate: int
    demand_level: str
    recommendation: str
    blocks_aggregated: int

    class Config:
        protected_namespaces = ()
