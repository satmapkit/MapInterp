from abc import ABC
from OceanDB.AlongTrack import AlongTrack
import numpy as np


class Interpolator(ABC):
    """
    Abstract base class for interpolation methods.
    """

    @abstractmethod
    def interp(args): ...



class NearestNeighbor(Interpolator): ...

class GaussianKernel(Interpolator):
    def __init__(self, spread_distance):
        self.L = (50e3 + 250e3 * (900 / (lat_ocean ** 2 + 900))) / 3.34


    def kernel(self, x2, sigma):
        return np.exp(-0.5 * x2 / (sigma*sigma)) / (sigma * np.sqrt(2 * np.pi))

    def interp(args):
        gaussian_kernel_smoother(x2, data["sla_filtered"], L[i])


class AlongTrackInterp(AlongTrack):
    """ """


    def interp(self, grid: Grid, interp_method: str): ...

    def interp(self, lat: float, lon: float, time: DateTime, interp_method: str): ...

    def interpn(self, lats: np.ndarray, lon: np.ndarray, interp_method: str): ...


    def interp_grid(self, min_lat, max_lat, min_lon, max_lon, interp_method: str): ...

lat,lon,time = grid(args)
interp_points = atdb.interp(lat,lon,time)
dataframe = pd.DataFrame({"lat":lat, "lon":lon, "time":time,"points":inter_points})
dataframe.save_netcdf("filename")
# file has 4 columns

g = Grid(args)
g.interp()
g.save_min_size_netcdf("filename")
# file has 1 column, but with args as metadata


# use cases: 
# points along ship trajectories
# points along float trajectories
# grid in lat/lon (geographic coords)
# grid in projected coords


ship_trajectory = pd.from_csv("trajectory.csv")
atdb.interp(ship_trajectory, "nearest_neighbor")

grid = make_me_a_grid()
atdb.interp(grid, "nearest_neighbor")

atdb = AlongTrackInterp()
grid_data = atdb.interp_grid(-60, 60, 40, 50, "nearest_neighbor")
