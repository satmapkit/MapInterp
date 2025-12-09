from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
from datetime import datetime
from typing import Any

from OceanDB.AlongTrack import AlongTrack, SLA_Geographic, SLA_Projected

@dataclass
class GeographicPoint:
    latitude: float
    longitude: float
    datetime: datetime


class Interpolator(ABC):
    @abstractmethod
    def interp(self, data: SLA_Geographic, point: GeographicPoint) -> np.float64|float:
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
    def interp(self, data: SLA_Projected, point: GeographicPoint):
        ...


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
        weights = np.exp(-1/2 * (r/self.length_scale)**2)
        weights /= weights.sum() # normalize
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
        x2 = data.delta_x **2 + data.delta_y **2
        weights = np.exp(-1/2 * x2/self.length_scale**2)
        if weights.sum() == 0 or np.isnan(weights.sum()):
            print("got none or nan")
            print(data)
            print(weights)
        weights /= weights.sum() # normalize
        return np.dot(data.sla_filtered, weights)


def interpolate_using_atdb(
    lats: npt.NDArray[np.floating[Any]],
    lons: npt.NDArray[np.floating[Any]],
    dates: list[datetime]|datetime,
    interpolator: Interpolator,
    atdb: AlongTrack = AlongTrack(),
    distances: float | npt.NDArray[np.float64] = 150000,
    missions: list[str] | None = None,
):

    # initialize output array
    sla = np.empty_like(lats)
    # any requested points not on the ocean will be filled with NaN.
    sla[:] = np.nan

    # restrict to only ocean points
    basin_mask = atdb.basin_mask(lats, lons)
    ocean_indices = (basin_mask > 0) & (basin_mask < 1000)
    lat_ocean = lats[ocean_indices]
    lon_ocean = lons[ocean_indices]
    sla_ocean = sla[ocean_indices]

    if isinstance(dates, datetime):
        dates = [dates] * lat_ocean.size

    if isinstance(interpolator, NearestNeighborInterpolator):
        data = atdb.geographic_nearest_neighbors_dt(
                lat_ocean,
                lon_ocean,
                dates,
                missions=missions
                )
    elif isinstance(interpolator, GeographicWindowInterpolator):
        data = atdb.geographic_points_in_r_dt(
                lat_ocean,
                lon_ocean,
                dates,
                distances=distances,
                missions=missions
                )
    elif isinstance(interpolator, ProjectedWindowInterpolator):
        data = atdb.projected_points_in_r_dt(
                lat_ocean,
                lon_ocean,
                dates,
                distances=distances,
                missions=missions
                )
    else:
        raise ValueError("Unrecognized Interpolator type")

    nones = 0
    for i, (data, lat_point, lon_point, date) in enumerate(zip(data, lat_ocean, lon_ocean, dates)):
        if data is None:
            sla_ocean[i] = np.nan
            nones += 1
        else:
            sla_ocean[i] = interpolator.interp(data, GeographicPoint(lat_point, lon_point, date))

    sla[ocean_indices] = sla_ocean
    return sla

