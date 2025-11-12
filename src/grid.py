from OceanDB.AlongTrack import AlongTrack
import numpy as np
import numpy.typing as npt
from typing import Any


def geographic_grid_using_atdb(
    lat: npt.NDArray[np.floating[Any]],
    lon: npt.NDArray[np.floating[Any]],
    atdb: AlongTrack | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if atdb is None:
        atdb = AlongTrack()
    lat_world, lon_world = np.meshgrid(lat, lon)
    lat_world = lat_world.reshape(-1)
    lon_world = lon_world.reshape(-1)
    basin_mask = atdb.basin_mask(lat_world, lon_world)
    ocean_indices = (basin_mask > 0) & (basin_mask < 1000)
    lat_ocean = lat_world[ocean_indices]
    lon_ocean = lon_world[ocean_indices]
    return lat_ocean, lon_ocean
