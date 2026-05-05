from abc import ABC, abstractmethod
import numpy as np
from datetime import timedelta
from typing import Any, TypedDict, TypeVar, Literal, Mapping, Generic, cast
import time

from OceanDB.data_access.along_track import AlongTrack, along_track_fields, Mission
from OceanDB.utils.projections import latitude_longitude_to_spherical_transverse_mercator
from OceanDB.ocean_data.dataset import Dataset

from .InterpQueryPoints import InterpQueryPoints, InterpQueryPoint
from .InterpolatedPoints import InterpolatedPoints

class InterpArgs(TypedDict):
    fields: list[along_track_fields]

class NearestNeighborInterpArgs(InterpArgs):
    time_window: timedelta
    missions: list[Mission]

class GeographicWindowArgs(InterpArgs):
    radius: float
    time_window: timedelta
    missions: list[Mission]

K = TypeVar('K', bound=str)
V = TypeVar('V', bound=InterpArgs)
class Interpolator(Generic[K, V], ABC):
    @abstractmethod
    def query_params(self) -> Mapping[K, V]:
        """
        Returns
        -------
        A dictionary mapping from used queries to the parameters that should be used for that query
        """

    @abstractmethod
    def interp(
            self,
            data: dict[K, Dataset[along_track_fields, Any]|None],
            point: InterpQueryPoint
            ) -> float:
        """
        Interpolate to point, given SLA data
        """



class GeographicGaussianKernelInterpolator(Interpolator[Literal['geographic_window'], GeographicWindowArgs]):
    def __init__(
            self,
            length_scale: float,
            radius: float = 500_000.0,
            time_window: timedelta = timedelta(days=10)
            ):
        """
        Interpolation method using a gaussian spreading kernel

        Arguments
        ---------
        length_scale: spreading distance, in meters
        radius: maximum distance from interp point to query point
        time_window: maximum time difference (+ or -) from interp point to query point
        """

        self.length_scale = length_scale
        self.radius = radius
        self.time_window = time_window

    def query_params(self):
        args : Mapping[Literal["geographic_window"], GeographicWindowArgs] = {
            "geographic_window": {
                "fields": ["sla_filtered", "distance"],
                "missions": ["al"],
                "radius": self.radius,
                "time_window": self.time_window,
            }
        }
        return args


    def interp(self, data, point):
        data = data["geographic_window"]
        if data is None:
            print("No data found in range")
            return 0
        r = data["distance"]
        weights = np.exp(-1 / 2 * (r / self.length_scale) ** 2)
        weights /= weights.sum()  # normalize
        return np.dot(data["sla_filtered"], weights)


class ProjectedGaussianKernelInterpolator(Interpolator[Literal['geographic_window'], GeographicWindowArgs]):
    def __init__(
            self,
            length_scale: float,
            radius: float = 500_000.0,
            time_window: timedelta = timedelta(days=10)
            ):
        """
        Interpolation method using a gaussian spreading kernel

        Arguments
        ---------
        length_scale: spreading distance, in meters
        radius: maximum distance from interp point to query point
        time_window: maximum time difference (+ or -) from interp point to query point
        """

        self.length_scale = length_scale
        self.radius = radius
        self.time_window = time_window

    def query_params(self):
        args : Mapping[Literal["geographic_window"], GeographicWindowArgs] = {
            "geographic_window": {
                "fields": ["latitude", "longitude", "sla_filtered"],
                "missions": ["al"],
                "radius": self.radius,
                "time_window": self.time_window,
            }
        }
        return args


    def interp(self, data, point):
        data = data["geographic_window"]
        if data is None:
            print("No data found in range")
            return 0
        x, y = latitude_longitude_to_spherical_transverse_mercator(
            data["latitude"], data["longitude"], point.lon
        )
        x0, y0 = latitude_longitude_to_spherical_transverse_mercator(
            point.lat, point.lon, point.lon
        )
        x2 = (x - x0) ** 2 + (y - y0) ** 2
        weights = np.exp(-1 / 2 * x2 / self.length_scale**2)
        if weights.sum() == 0 or np.isnan(weights.sum()):
            print("got none or nan")
            print(data)
            print(weights)
        weights /= weights.sum()  # normalize
        return np.dot(data["sla_filtered"], weights)


class NearestNeighborInterpolator(Interpolator[Literal['nearest_neighbor'], NearestNeighborInterpArgs]):
    def __init__(
            self,
            time_window: timedelta = timedelta(days=10)
            ):
        """
        Interpolation method using a gaussian spreading kernel

        Arguments
        ---------
        length_scale: spreading distance, in meters
        radius: maximum distance from interp point to query point
        time_window: maximum time difference (+ or -) from interp point to query point
        """

        self.time_window = time_window

    def query_params(self):
        args : Mapping[Literal["nearest_neighbor"], NearestNeighborInterpArgs] = {
            "nearest_neighbor": {
                "fields": ["sla_filtered", "mission"],
                "missions": ["al"],
                "time_window": self.time_window,
            }
        }
        return args


    def interp(self, data, point):
        data = data["nearest_neighbor"]
        if data is None:
            print("No data found in range")
            return 0
        return data["sla_filtered"][0]




def interpolate_using_atdb(
    queryPoints: InterpQueryPoints,
    interpolator: Interpolator[K, V],
    atdb: AlongTrack = AlongTrack(),
) -> InterpolatedPoints:

    # initialize output array
    sla = np.empty_like(queryPoints.lat)
    # any requested points not on the ocean will be filled with NaN.
    sla[:] = np.nan

    # restrict to only ocean points
    basin_mask = atdb.basin_mask_lookup.lookup(queryPoints.lat, queryPoints.lon)
    ocean_indices = (basin_mask > 0) & (basin_mask < 1000)
    sla_ocean = sla[ocean_indices]
    ocean_flat_indices = np.flatnonzero(ocean_indices.reshape(-1))

    query_params = interpolator.query_params()

    # query data in batch before interpolating each point
    nearest_neighbor_all = None
    if "nearest_neighbor" in query_params.keys():
        # extract the parameters
        nearest_neighbor_params = query_params["nearest_neighbor"]
        ocean_points = [queryPoints[flat_index] for flat_index in ocean_flat_indices]

        # fetch data
        nearest_neighbor_all = atdb.geographic_nearest_neighbors_batch(
            fields=nearest_neighbor_params["fields"],
            latitudes=[point.lat for point in ocean_points],
            longitudes=[point.lon for point in ocean_points],
            dates=[point.date for point in ocean_points],
            missions=nearest_neighbor_params["missions"],
            time_window=nearest_neighbor_params["time_window"],
        )

    # query data in batch before interpolating each point
    geographic_window_all = None
    if "geographic_window" in query_params.keys():
        # TODO: find a better fix for typecheck errors
        q = cast(Mapping[Literal["geographic_window"], NearestNeighborInterpArgs], query_params)

        # extract the parameters
        geographic_window_params = q["geographic_window"]
        ocean_points = [queryPoints[flat_index] for flat_index in ocean_flat_indices]

        # fetch data
        geographic_window_all = atdb.geographic_point_in_r_dt_batch(
            fields=geographic_window_params["fields"],
            latitudes=[point.lat for point in ocean_points],
            longitudes=[point.lon for point in ocean_points],
            dates=[point.date for point in ocean_points],
            missions=geographic_window_params["missions"],
            time_window=geographic_window_params["time_window"],
        )

    # interpolate using the batched query results
    for (i, flat_index) in enumerate(ocean_flat_indices):
        point = queryPoints[flat_index]
        data : dict[K, Dataset[along_track_fields, Any]|None]= {method: None for method in query_params.keys()}
        if nearest_neighbor_all is not None:
            data["nearest_neighbor"] = next(nearest_neighbor_all)
        if geographic_window_all is not None:
            data["geographic_window"] = next(geographic_window_all)

        # do the interpolation
        sla_ocean[i] = interpolator.interp(data, point)

    sla[ocean_indices] = sla_ocean
    sla_obj = InterpolatedPoints(sla, queryPoints)
    return sla_obj
