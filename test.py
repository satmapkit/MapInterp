from src.Interpolator import interpolate_using_atdb, NearestNeighborInterpolator
from OceanDB.AlongTrack import AlongTrack

import numpy as np
from datetime import datetime
import xarray as xr
from matplotlib import colors
import matplotlib.pyplot as plt

date = datetime(year=2008, month=10, day=29, hour=3)

# choose lat/lon grid values
resolution = 2.5
lon_dim = np.arange(-180, 180 - resolution, resolution) + resolution / 2
lat_dim = np.arange(-70, 70 - resolution, resolution) + resolution / 2
lon, lat = np.meshgrid(lon_dim, lat_dim)

# get sla
print("setting up db")
atdb = AlongTrack(config_dir="/home/rowedaniel/work/nwra/OceanDB/")
print("setting up interpolator")
interpolator = NearestNeighborInterpolator()

print("doing the interpolation")
sla = interpolate_using_atdb(lat, lon, date, interpolator, atdb)

print("reformatting for graphing")
sla_grid = sla.reshape(lat.shape)
print('sla_grid.shape:', sla_grid.shape)
print('lon.shape:', lon.shape)
sla_map_nn = xr.DataArray(sla_grid,
                          coords={'latitude': lat_dim, 'longitude': lon_dim},
                          dims=["latitude", "longitude"])

plt.figure()
# ax = sla_map.plot.contourf(levels=100, norm=norm, cmap='RdBu_r')
# plt.pcolormesh(lat, lon, sla_grid, cmap='RdBu_r')
norm = colors.Normalize(vmin=-0.5, vmax=0.5)
ax = sla_map_nn.plot.pcolormesh(norm=norm, cmap='RdBu_r')
plt.title('Nearest neighbor map')
plt.show()
