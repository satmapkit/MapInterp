from abc import ABC, abstractmethod
from OceanDB.AlongTrack import AlongTrack
import numpy as np
import numpy.typing as npt
from datetime import datetime
from typing import Any


class Interpolator(ABC):
    @abstractmethod
    def interp(self, data: dict[str, npt.NDArray[np.float64]]) -> np.float64: ...


class NearestNeighborInterpolator(Interpolator):
    """
    Base class for interpolation methods that use nearest neighbor(s)
    """

    def interp(self, data):
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


def interpolate_using_atdb(
    lat: npt.NDArray[np.floating[Any]],
    lon: npt.NDArray[np.floating[Any]],
    date: datetime,
    interpolator: Interpolator,
    atdb: AlongTrack,
    distance: float | npt.NDArray[np.float64] = 500000,
    missions: list[str] | None = None,
):

    # initialize output array
    sla = np.empty_like(lat)
    # any requested points not on the ocean will be filled with NaN.
    sla[:] = np.nan

    # restrict to only ocean points
    basin_mask = atdb.basin_mask(lat, lon)
    ocean_indices = (basin_mask > 0) & (basin_mask < 1000)
    lat_ocean = lat[ocean_indices]
    lon_ocean = lon[ocean_indices]
    sla_ocean = sla[ocean_indices]

    if isinstance(interpolator, NearestNeighborInterpolator):
        for i, data in enumerate(
            atdb.geographic_nearest_neighbors(
                lat_ocean, lon_ocean, date, missions=missions
            )
        ):
            sla_ocean[i] = interpolator.interp(data)
    if isinstance(interpolator, GeographicWindowInterpolator):
        for i, data in enumerate(
            atdb.geographic_points_in_spatialtemporal_windows(
                lat_ocean, lon_ocean, date, distance=distance, missions=missions
            )
        ):
            sla_ocean[i] = interpolator.interp(data)
    sla[ocean_indices] = sla_ocean
    return sla




# class GaussianKernel(ProjectedWindowInterpolator):
#     def __init__(self, spread_distance):
#         self.spread_distance = spread_distance
#
#     def area_element(self, latitude: float) -> float:
#         return (50e3 + 250e3 * (900 / (latitude**2 + 900))) / 3.34
#
#     def kernel(self, x2, sigma):
#         return np.exp(-0.5 * x2 / (sigma * sigma)) / (sigma * np.sqrt(2 * np.pi))
#
#     def interp(args):
#         gaussian_kernel_smoother(x2, data["sla_filtered"], L[i])


# class AlongTrackInterp(AlongTrack):
#     """ """
#
#     def interp(self, grid: Grid, interp_method: str): ...
#
#     def interp(self, lat: float, lon: float, time: DateTime, interp_method: str): ...
#
#     def interpn(self, lats: np.ndarray, lon: np.ndarray, interp_method: str): ...
#
#     def interp_grid(self, min_lat, max_lat, min_lon, max_lon, interp_method: str): ...
#
#
# lat, lon, time = grid(args)
# interp_points = atdb.interp(lat, lon, time)
# dataframe = pd.DataFrame({"lat": lat, "lon": lon, "time": time, "points": inter_points})
# dataframe.save_netcdf("filename")
# # file has 4 columns
#
# g = Grid(args)
# g.interp()
# g.save_min_size_netcdf("filename")
# # file has 1 column, but with args as metadata
#
#
# # use cases:
# # points along ship trajectories
# # points along float trajectories
# # grid in lat/lon (geographic coords)
# # grid in projected coords
#
#
# ship_trajectory = pd.from_csv("trajectory.csv")
# atdb.interp(ship_trajectory, "nearest_neighbor")
#
# grid = make_me_a_grid()
# atdb.interp(grid, "nearest_neighbor")
#
# atdb = AlongTrackInterp()
# grid_data = atdb.interp_grid(-60, 60, 40, 50, "nearest_neighbor")
