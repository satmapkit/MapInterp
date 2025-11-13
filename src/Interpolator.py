from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from datetime import datetime
from typing import Any

from OceanDB.AlongTrack import AlongTrack, SLA_Geographic, SLA_Projected

class SLA_Geographic:
    """
    Make SLA data type explict
    longitude,
     latitude,
	 sla_filtered,
	 EXTRACT(EPOCH FROM ({central_date_time} - date_time)) AS time_difference_secs

    """
    latitude: npt.NDArray
    longitude: npt.NDArray
    sla_filtered: npt.NDArray
    delta_t: npt.NDArray

@dataclass
class GeographicPoint:
    latitude: float
    longitude: float
    datetime: datetime

@dataclass
class ProjectedPoint(GeographicPoint):
    x: float
    y: float


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
    def interp(self, data: SLA_Projected, point: ProjectedPoint):
        ...


class GeographicGaussianKernelInterpolator(GeographicWindowInterpolator):
    def __init__(self, sigma: float, earth_radius = 6.357 * 10**6):
        """
        Interpolation method using a gaussian spreading kernel

        sigma: spreading distance, in meters
        earth_radius: Earth's radius, in meters
        """

        self.sigma = sigma
        self.earth_radius = earth_radius

    def distance(self,
                 lats: npt.NDArray[np.floating],
                 lons: npt.NDArray[np.floating],
                 ref_lat: float,
                 ref_lon: float) -> npt.NDArray[np.floating]:
        """
        Finds the great-circle distance from each point in points to the reference point ref_point,
        approximating the earth as a sphere

        lats: n-array of latitudes
        lons: n-array of longitudes
        ref_point: single reference point (lat, lon)
        """

        # convert from degrees to rad
        lats = lats * np.pi / 180
        lons = lons * np.pi / 180
        ref_lat = ref_lat * np.pi / 180
        ref_lon = ref_lon * np.pi / 180

        return np.arccos(np.sin(lats) * np.sin(ref_lat) +
                         np.cos(lats) * np.cos(ref_lat) * np.cos(lons - ref_lon)
                         ) * self.earth_radius

    def gaussian(self, x2: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """
        Return the gaussian kernal function evaluated at given distances.
        x2: Square distances from center
        """
        return np.exp(-x2 / (2  * self.sigma * self.sigma)) / (self.sigma * np.sqrt(2 * np.pi))

    def interp(self, data, point):
        x2 = self.distance(data.latitude, data.longitude, point.latitude, point.longitude) ** 2
        weights = self.gaussian(x2)
        print(weights.sum())
        weights /= weights.sum() # normalize
        return np.dot(data.sla_filtered, weights)


def interpolate_using_atdb(
    lats: npt.NDArray[np.floating[Any]],
    lons: npt.NDArray[np.floating[Any]],
    dates: list[datetime]|datetime,
    interpolator: Interpolator,
    atdb: AlongTrack,
    distances: float | npt.NDArray[np.float64] = 500000,
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
    # print(len(dates), lat_ocean.shape)

    if isinstance(interpolator, NearestNeighborInterpolator):
        for i, (data, lat_point, lon_point, date) in enumerate(
            zip(atdb.geographic_nearest_neighbors(lat_ocean, lon_ocean, dates, missions=missions),
                lat_ocean, lon_ocean, dates
                )):
            sla_ocean[i] = interpolator.interp(data, GeographicPoint(lat_point, lon_point, date))
    elif isinstance(interpolator, GeographicWindowInterpolator):
        for i, (data, lat_point, lon_point, date) in enumerate(
            zip(atdb.geographic_points_in_spatialtemporal_windows(lat_ocean, lon_ocean, dates, distances=distances, missions=missions),
                lat_ocean, lon_ocean, dates
                )):
            sla_ocean[i] = interpolator.interp(data, GeographicPoint(lat_point, lon_point, date))
    sla[ocean_indices] = sla_ocean
    return sla

