from src.Interpolator import interpolate_using_atdb, NearestNeighborInterpolator, GeographicGaussianKernelInterpolator
from OceanDB.AlongTrack import AlongTrack

import numpy as np
from datetime import datetime
from matplotlib import colors
import matplotlib.pyplot as plt
import time

date = datetime(year=2013, month=12, day=31, hour=23)

# choose lat/lon grid values
resolution = 2.5
lon_dim = np.arange(-180, 180 - resolution, resolution) + resolution / 2
lat_dim = np.arange(-70, 70 - resolution, resolution) + resolution / 2
lon, lat = np.meshgrid(lon_dim, lat_dim)

print("setting up db")
atdb = AlongTrack()


print("setting up interpolators")
nnInterp = NearestNeighborInterpolator()
gaInterp = GeographicGaussianKernelInterpolator(200 * 10**3)

print("doing the interpolation")

print("starting nearest neighbor")
t1 = time.time()
sla_nn = interpolate_using_atdb(lat, lon, date, nnInterp, atdb)
print(f"finished nearest neighbor. Took {time.time()-t1} seconds")

print("starting gaussian")
t1 = time.time()
sla_ga = interpolate_using_atdb(lat, lon, date, gaInterp, atdb)
print(f"finished gaussian. Took {time.time()-t1} seconds")

print("reformatting for graphing")
sla_nn_grid = sla_nn.reshape(lat.shape)
sla_ga_grid = sla_ga.reshape(lat.shape)

norm = colors.Normalize(vmin=-0.5, vmax=0.5)

# nearest neighbor
plt.figure()
plt.pcolormesh(lon, lat, sla_nn_grid, cmap='RdBu_r', norm=norm)
plt.title('Nearest neighbor map')
plt.savefig("/app/data/output_nn.png")

# geographic gaussian
plt.figure()
plt.pcolormesh(lon, lat, sla_ga_grid, cmap='RdBu_r', norm=norm)
plt.title('Geographic gaussian map')
plt.savefig("/app/data/output_ga.png")

# diff
plt.figure()
plt.pcolormesh(lon, lat, sla_nn_grid - sla_ga_grid, cmap='RdBu_r', norm=norm)
plt.title('difference nn - ga')
plt.savefig("/app/data/output_diff.png")
