from OceanDB.AlongTrack import AlongTrack
import numpy as np
import numpy.typing as npt
from typing import Any
from xarray import DataArray


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


def save_netcdf_grid(
    x: npt.NDArray[np.floating[Any]],
    y: npt.NDArray[np.floating[Any]],
    sla: npt.NDArray[np.floating[Any]],
    filename: str,
    x_name: str = "lat",
    y_name: str = "lon",
    attrs: dict[str, Any] = {},
):
    """
    Save sea level anomalies on a grid

    x: n-array of x axis values
    y: m-array of y axis values
    sla: n by m-array of sea level anomalies
    x_name: name of x-axis
    y_name: name of y-axis
    attrs: additional metadata
    """
    df = DataArray(sla, coords=[(x_name, x), (y_name, y)], name="sla", attrs=attrs)
    df.to_netcdf(filename)
