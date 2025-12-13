from src.Interpolator import interpolate_using_atdb, NearestNeighborInterpolator, GeographicGaussianKernelInterpolator, ProjectedGaussianKernelInterpolator

import numpy as np
from datetime import datetime
from matplotlib import colors
import matplotlib.pyplot as plt
import time

date = datetime(year=2013, month=12, day=31, hour=23)

# choose lat/lon grid values
resolution = 5
lon_dim = np.arange(-180, 180 - resolution, resolution) + resolution / 2
lat_dim = np.arange(-70, 70 - resolution, resolution) + resolution / 2
lon, lat = np.meshgrid(lon_dim, lat_dim)

interpolators = [
        (NearestNeighborInterpolator(), "Nearest neighbor"),
        (GeographicGaussianKernelInterpolator(100 * 10**3), "Geographic gaussian kernel"),
        (ProjectedGaussianKernelInterpolator(100 * 10**3), "Projected gaussian kernel"),
        ]
all_sla = []
for interp, title in interpolators:
    print(f"doing the interpolation using {title.lower()}")
    t1 = time.time()
    all_sla.append(interpolate_using_atdb(lat, lon, date, interp))
    print(f"finished interpolation. Took {time.time()-t1}s")

    # make a plot

norm = colors.Normalize(vmin=-0.5, vmax=0.5)
for i,(sla, (_,title)) in enumerate(zip(all_sla, interpolators)):
    plt.figure()
    plt.pcolormesh(lon, lat, sla, cmap='RdBu_r', norm=norm)
    plt.title(f'{title}')
    filename = "/app/data/output_" + ''.join(title.lower().strip().split(' ')) + ".png"
    plt.savefig(filename, dpi=400)

    for sla2, (_,title2) in zip(all_sla[i+1:], interpolators[i+1:]):
        plt.figure()
        plt.pcolormesh(lon, lat, sla-sla2, cmap='RdBu_r')
        plt.title(f'{title}')
        filename = "/app/data/output_" + \
                   ''.join(title.lower().strip().split(' ')) + \
                   "_minus_" + \
                   ''.join(title2.lower().strip().split(' ')) + \
                   ".png"
        plt.savefig(filename, dpi=400)
