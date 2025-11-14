


#############################################################
# verbose version
#############################################################


class GeographicGaussianKernelInterpolator(
        GeographicWindowInterpolator
    ):
    def __init__(self, length_scale: float):
        """
        Interpolation method using a gaussian spreading kernel

        length_scale: spreading distance, in meters
        earth_radius: Earth's radius, in meters
        """
        self.length_scale = length_scale

    def interp(self, data, point):
        r = data.distance
        weights = np.exp(-1/2 * (r/self.length_scale)**2)
        weights /= weights.sum() # normalize
        return np.dot(data.sla_filtered, weights)



date = datetime(year=2013, month=12, day=31, hour=23)
lon_dim = np.arange(-180, 180, 2.5) # resolution of 2.5 deg
lat_dim = np.arange(-70, 70, 2.5)
lon, lat = np.meshgrid(lon_dim, lat_dim)

# interpolate over a length scale of 50 km
interp = GeographicGaussianKernelInterpolator(50 * 10**3)
sla = interpolate_using_atdb(lat, lon, date, interp)

plt.figure()
plt.pcolormesh(lon, lat, sla, cmap='RdBu_r')
plt.title('Sea surface anomaly (m)')








#############################################################
# skinnier version
#############################################################


class GeographicGaussianKernelInterpolator(
        GeographicWindowInterpolator
    ):
    def __init__(self, length_scale: float):
        self.length_scale = length_scale

    def interp(self, data, point):
        r = data.distance
        weights = np.exp(-1/2 * (r/self.length_scale)**2)
        weights /= weights.sum() # normalize
        return np.dot(data.sla_filtered, weights)


date = datetime(year=2013, month=12, day=31, hour=23)
lon_dim = np.arange(-180, 180, 2.5) # resolution of 2.5 deg
lat_dim = np.arange(-70, 70, 2.5)
lon, lat = np.meshgrid(lon_dim, lat_dim)

interp = GeographicGaussianKernelInterpolator(50 * 10**3)
sla = interpolate_using_atdb(lat, lon, date, interp)

plt.figure()
plt.pcolormesh(lon, lat, sla, cmap='RdBu_r')
plt.title('Sea surface anomaly (m)')

