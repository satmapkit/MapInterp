from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from datetime import datetime, timedelta
from typing import Any, Mapping, cast

from OceanDB.data_access.along_track import AlongTrack
from OceanDB.utils.projections import latitude_longitude_to_spherical_transverse_mercator

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
    def interp(self, data: Mapping[Any, Any], point: GeographicPoint) -> np.float64 | float:
        """
        Interpolate to point, given SLA data
        """


class NearestNeighborInterpolator(Interpolator):
    """
    Base class for interpolation methods that use nearest neighbor(s)
    """

    def interp(self, data, point):
        return data["sla_filtered"][0]


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

    @abstractmethod
    def interp(
        self, data: Mapping[Any, Any], point: GeographicPoint
    ) -> np.float64 | float:
        raise NotImplementedError


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
        r = data["distance"]
        weights = np.exp(-1 / 2 * (r / self.length_scale) ** 2)
        weights /= weights.sum()  # normalize
        return np.dot(data["sla_filtered"], weights)


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
        x, y = latitude_longitude_to_spherical_transverse_mercator(
            data["latitude"], data["longitude"], point.longitude
        )
        x0, y0 = latitude_longitude_to_spherical_transverse_mercator(
            point.latitude, point.longitude, point.longitude
        )
        x2 = (x - x0) ** 2 + (y - y0) ** 2
        weights = np.exp(-1 / 2 * x2 / self.length_scale**2)
        if weights.sum() == 0 or np.isnan(weights.sum()):
            print("got none or nan")
            print(data)
            print(weights)
        weights /= weights.sum()  # normalize
        return np.dot(data["sla_filtered"], weights)


def interpolate_using_atdb(
    queryPoints: InterpQueryPoints,
    interpolator: Interpolator,
    atdb: AlongTrack = AlongTrack(),
    distances: float | npt.NDArray[np.float64] = 150000,
    missions: list[str] | None = None,
) -> InterpolatedPoints:

    def query_fields() -> list[Any]:
        if isinstance(interpolator, NearestNeighborInterpolator):
            return ["sla_filtered", "distance"]
        if isinstance(interpolator, GeographicWindowInterpolator):
            fields = ["sla_filtered"]
            if getattr(interpolator, "needs_distance", False):
                fields.append("distance")
            return fields
        if isinstance(interpolator, ProjectedWindowInterpolator):
            return ["latitude", "longitude", "sla_filtered"]
        raise ValueError("Unrecognized Interpolator type")

    def point_distance(i: int) -> float:
        if not isinstance(distances, np.ndarray):
            return float(cast(Any, distances))
        return float(distances.astype(float, copy=False).reshape(-1)[i])

    # initialize output array
    sla = np.empty_like(queryPoints.lat)
    # any requested points not on the ocean will be filled with NaN.
    sla[:] = np.nan

    # restrict to only ocean points
    basin_mask = atdb.basin_mask(queryPoints.lat, queryPoints.lon)
    ocean_indices = (basin_mask > 0) & (basin_mask < 1000)
    sla_ocean = sla[ocean_indices]
    ocean_flat_indices = np.flatnonzero(ocean_indices.reshape(-1))
    fields = query_fields()
    query_missions = cast(Any, missions if missions is not None else atdb.all_missions)
    time_window = timedelta(days=10)

    for i, flat_index in enumerate(ocean_flat_indices):
        point = GeographicPoint.from_query_points(queryPoints, flat_index)

        if isinstance(interpolator, NearestNeighborInterpolator):
            data = atdb.geographic_nearest_neighbors(
                fields=fields,
                latitude=point.latitude,
                longitude=point.longitude,
                date=point.datetime,
                time_window=time_window,
                missions=query_missions,
            )
        elif isinstance(interpolator, GeographicWindowInterpolator | ProjectedWindowInterpolator):
            data = atdb.geographic_point_in_r_dt(
                fields=fields,
                latitude=point.latitude,
                longitude=point.longitude,
                date=point.datetime,
                radius=point_distance(flat_index),
                time_window=time_window,
                missions=query_missions,
            )
        else:
            raise ValueError("Unrecognized Interpolator type")

        if data is None:
            sla_ocean[i] = np.nan
        else:
            sla_ocean[i] = interpolator.interp(data, point)

    sla[ocean_indices] = sla_ocean
    sla_obj = InterpolatedPoints(sla, queryPoints)
    return sla_obj
