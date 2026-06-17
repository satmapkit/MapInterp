from MapInterp.Interpolator import (
    GeographicGaussianKernelInterpolator,
    NearestNeighborInterpolator,
    ProjectedGaussianKernelInterpolator,
    interpolate_using_atdb,
)
from MapInterp.Grid import InterpGrid

import numpy as np
from datetime import datetime, timedelta
from matplotlib import colors
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import time

# date = datetime(year=2013, month=5, day=21, hour=23)
date = datetime(year=2021, month=7, day=4, hour=23)

# choose lat/lon grid values
resolution = 10
lon_dim = np.arange(-180, 180 - resolution, resolution) + resolution / 2
lat_dim = np.arange(-70, 70 - resolution, resolution) + resolution / 2
grid = InterpGrid(lat_dim, lon_dim, date, 'equirectangular')

interpolators = [
        (NearestNeighborInterpolator(time_window=timedelta(days=10/2)), "Nearest neighbor"),
        # (GeographicGaussianKernelInterpolator(50 * 10**3), "Geographic gaussian kernel"),
        # (ProjectedGaussianKernelInterpolator(50 * 10**3), "Projected gaussian kernel"),
        ]
for interp, title in interpolators:
    print(f"doing the interpolation using {title.lower()}")
    t1 = time.time()
    sla_obj = interpolate_using_atdb(grid, interp)
    print(f"finished interpolation. Took {time.time()-t1}s")

    # make a plot

    # norm = colors.Normalize(vmin=-0.5, vmax=0.5)
    map_crs = ccrs.PlateCarree(central_longitude=-60)
    data_crs = ccrs.PlateCarree()
    fig, ax = plt.subplots(
        figsize=(12, 6),
        constrained_layout=True,
        subplot_kw={"projection": map_crs},
    )
    # plt.pcolormesh(sla_obj.queryPoints.y, sla_obj.queryPoints.x, sla_obj.sla, cmap='RdBu_r', norm=norm)
    mesh = ax.pcolormesh(
        sla_obj.queryPoints.y,
        sla_obj.queryPoints.x,
        sla_obj.sla,
        cmap='RdBu_r',
        transform=data_crs,
    )
    ax.add_feature(
        cfeature.LAND.with_scale("50m"),
        facecolor="0.5",
        edgecolor="none",
        zorder=2,
    )
    # ax.coastlines(resolution="50m", linewidth=0.6, color="black", zorder=3)
    ax.set_extent([120, 480, -70, 70], crs=data_crs)
    ax.set_xticks(np.arange(120, 481, 30), crs=data_crs)
    ax.set_yticks(np.arange(-60, 61, 20), crs=data_crs)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_title(
        f'{title}, resolution={resolution} degrees, {str(date)}',
        fontsize=12,
    )
    mesh.set_clim(-1, 1)
    plt.colorbar(mesh, ax=ax, pad=0.03, location="bottom")
    filename = "figs/output_" + ''.join(title.lower().strip().split(' '))
    plt.savefig(filename + ".png", dpi=400)

    # save to netcdf
    sla_obj.to_netcdf(filename + ".nc")
plt.show()
