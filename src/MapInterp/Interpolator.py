from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
from datetime import datetime
from typing import Any

from OceanDB.AlongTrack import AlongTrack, SLA_Geographic, SLA_Projected

from .InterpQueryPoints import InterpQueryPoints
from .InterpolatedPoints import InterpolatedPoints


@dataclass
class GeographicPoint:
    latitude: float
    longitude: float
    datetime: datetime

    @classmethod
    def from_query_points(cls, query_points: InterpQueryPoints, i: int):
        return cls(
            query_points.lat.item(i), query_points.lon.item(i), query_points.dates[i]
        )


class Interpolator(ABC):
    @abstractmethod
    def interp(
        self, data: SLA_Geographic, point: GeographicPoint
    ) -> np.float64 | float:
        """
        Interpolate to point, given SLA data
        """


class NearestNeighborInterpolator(Interpolator):
    """
    Base class for interpolation methods that use nearest neighbor(s)
    """

    def interp(self, data, point):
        return data.sla_filtered[0]


class GeographicWindowInterpolator(Interpolator, ABC):
    """
    Abstract base class for interpolation methods that use windows of geographic
    (lat/lon) points within some great circle distance.
    """


class ProjectedWindowInterpolator(Interpolator, ABC):
    """
    Abstract base class for interpolation methods that use windows of projected
    points (transverse Mercator by default).
    """

    def interp(self, data: SLA_Projected, point: GeographicPoint): ...


class GeographicGaussianKernelInterpolator(GeographicWindowInterpolator):
    def __init__(self, length_scale: float):
        """
        Interpolation method using a gaussian spreading kernel

        length_scale: spreading distance, in meters
        earth_radius: Earth's radius, in meters
        """

        self.length_scale = length_scale
        self.needs_distance = True

    def interp(self, data, point):
        r = data.distance
        weights = np.exp(-1 / 2 * (r / self.length_scale) ** 2)
        weights /= weights.sum()  # normalize
        return np.dot(data.sla_filtered, weights)


class ProjectedGaussianKernelInterpolator(ProjectedWindowInterpolator):
    def __init__(self, length_scale: float):
        """
        Interpolation method using a gaussian spreading kernel

        length_scale: spreading distance, in meters
        earth_radius: Earth's radius, in meters
        """

        self.length_scale = length_scale
        self.needs_distance = True

    def interp(self, data, point):
        x2 = data.delta_x**2 + data.delta_y**2
        weights = np.exp(-1 / 2 * x2 / self.length_scale**2)
        if weights.sum() == 0 or np.isnan(weights.sum()):
            print("got none or nan")
            print(data)
            print(weights)
        weights /= weights.sum()  # normalize
        return np.dot(data.sla_filtered, weights)


def interpolate_using_atdb(
    queryPoints: InterpQueryPoints,
    interpolator: Interpolator,
    atdb: AlongTrack = AlongTrack(),
    distances: float | npt.NDArray[np.float64] = 150000,
    missions: list[str] | None = None,
) -> InterpolatedPoints:

    # initialize output array
    sla = np.empty_like(queryPoints.lat)
    # any requested points not on the ocean will be filled with NaN.
    sla[:] = np.nan

    # restrict to only ocean points
    basin_mask = atdb.basin_mask(queryPoints.lat, queryPoints.lon)
    ocean_indices = (basin_mask > 0) & (basin_mask < 1000)
    lat_ocean = queryPoints.lat[ocean_indices]
    lon_ocean = queryPoints.lon[ocean_indices]
    sla_ocean = sla[ocean_indices]

    if isinstance(interpolator, NearestNeighborInterpolator):
        data = atdb.geographic_nearest_neighbors_dt(
            lat_ocean, lon_ocean, queryPoints.dates, missions=missions
        )
    elif isinstance(interpolator, GeographicWindowInterpolator):
        data = atdb.geographic_points_in_r_dt(
            lat_ocean,
            lon_ocean,
            queryPoints.dates,
            distances=distances,
            missions=missions,
        )
    elif isinstance(interpolator, ProjectedWindowInterpolator):
        data = atdb.projected_points_in_r_dt(
            lat_ocean,
            lon_ocean,
            queryPoints.dates,
            distances=distances,
            missions=missions,
        )
    else:
        raise ValueError("Unrecognized Interpolator type")

    nones = 0
    for i, data in enumerate(data):
        if data is None:
            sla_ocean[i] = np.nan
            nones += 1
        else:
            point = GeographicPoint.from_query_points(queryPoints, i)
            sla_ocean[i] = interpolator.interp(data, point)

    sla[ocean_indices] = sla_ocean
    sla_obj = InterpolatedPoints(sla, queryPoints)
    return sla_obj
