from src.Interpolator import interpolate_using_atdb, NearestNeighborInterpolator, GeographicGaussianKernelInterpolator, ProjectedGaussianKernelInterpolator
from src.Grid import InterpGrid

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
grid = InterpGrid(lat_dim, lon_dim, date, 'equirectangular')

interpolators = [
        (NearestNeighborInterpolator(), "Nearest neighbor"),
        (GeographicGaussianKernelInterpolator(50 * 10**3), "Geographic gaussian kernel"),
        (ProjectedGaussianKernelInterpolator(50 * 10**3), "Projected gaussian kernel"),
        ]
for interp, title in interpolators:
    print(f"doing the interpolation using {title.lower()}")
    t1 = time.time()
    sla_obj = interpolate_using_atdb(grid, interp)
    print(f"finished interpolation. Took {time.time()-t1}s")

    # make a plot

    norm = colors.Normalize(vmin=-0.5, vmax=0.5)
    plt.figure()
    plt.pcolormesh(sla_obj.queryPoints.y, sla_obj.queryPoints.x, sla_obj.sla, cmap='RdBu_r', norm=norm)
    plt.title(f'{title}, resolution={resolution} degrees, {str(date)}')
    filename = "figs/output_" + ''.join(title.lower().strip().split(' '))
    plt.savefig(filename + ".png", dpi=400)

    # save to netcdf
    sla_obj.to_netcdf(filename + ".nc")
plt.show()
