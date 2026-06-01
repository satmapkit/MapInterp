import sys
import unittest
from datetime import datetime, timedelta
import inspect
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OCEANDB_SRC = ROOT.parent / "OceanDB" / "src"
MAPINTERP_SRC = ROOT / "src"

for path in (str(OCEANDB_SRC), str(MAPINTERP_SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)


from OceanDB.data_access.along_track import AlongTrack
from OceanDB.data_access.eddy import Eddy
from MapInterp.Grid import InterpGrid
from MapInterp.Interpolator import (
    GeographicGaussianKernelInterpolator,
    NearestNeighborInterpolator,
    ProjectedGaussianKernelInterpolator,
    interpolate_using_atdb,
)


class FakeBasinMaskLookup:
    def lookup(self, lat, lon):
        return np.ones_like(lat, dtype=int) if isinstance(lat, np.ndarray) else 1


class FakeDataset(dict):
    pass


class FakeAlongTrack:
    def __init__(self):
        self.basin_mask_lookup = FakeBasinMaskLookup()
        self.nearest_neighbor_calls = []
        self.geographic_window_calls = []

    def geographic_nearest_neighbors_batch(
        self, *, fields, latitudes, longitudes, dates, missions, time_window
    ):
        self.nearest_neighbor_calls.append(
            {
                "fields": fields,
                "latitudes": latitudes,
                "longitudes": longitudes,
                "dates": dates,
                "missions": missions,
                "time_window": time_window,
            }
        )

        for latitude in latitudes:
            yield FakeDataset(
                {
                    "sla_filtered": np.array([latitude], dtype=float),
                    "mission": np.array(["al"]),
                }
            )

    def geographic_point_in_r_dt_batch(
        self,
        *,
        fields,
        latitudes,
        longitudes,
        dates,
        radius,
        missions,
        time_window,
    ):
        self.geographic_window_calls.append(
            {
                "fields": fields,
                "latitudes": latitudes,
                "longitudes": longitudes,
                "dates": dates,
                "radius": radius,
                "missions": missions,
                "time_window": time_window,
            }
        )

        for latitude, longitude in zip(latitudes, longitudes, strict=True):
            yield FakeDataset(
                {
                    "latitude": np.array([latitude, latitude + 0.1], dtype=float),
                    "longitude": np.array([longitude, longitude + 0.1], dtype=float),
                    "sla_filtered": np.array([1.0, 0.5], dtype=float),
                    "distance": np.array([10.0, 20.0], dtype=float),
                }
            )


class OceanDBCompatibilityTests(unittest.TestCase):
    def test_current_oceandb_api_surface_exists(self):
        self.assertTrue(hasattr(AlongTrack, "geographic_nearest_neighbors"))
        self.assertTrue(hasattr(AlongTrack, "geographic_nearest_neighbors_batch"))
        self.assertTrue(hasattr(AlongTrack, "geographic_point_in_r_dt"))
        self.assertTrue(hasattr(AlongTrack, "geographic_point_in_r_dt_batch"))
        self.assertTrue(hasattr(Eddy, "get_eddy_tracks_from_times"))
        self.assertTrue(hasattr(Eddy, "eddy_with_track_id"))
        self.assertTrue(hasattr(Eddy, "eddy_envelope_query"))
        self.assertTrue(hasattr(Eddy, "along_track_points_near_eddy"))

    def test_along_track_method_signatures_match_expected_contract(self):
        nearest_params = inspect.signature(
            AlongTrack.geographic_nearest_neighbors
        ).parameters
        self.assertEqual(
            list(nearest_params),
            ["self", "fields", "latitude", "longitude", "date", "time_window", "missions"],
        )

        nearest_batch_params = inspect.signature(
            AlongTrack.geographic_nearest_neighbors_batch
        ).parameters
        self.assertEqual(
            list(nearest_batch_params),
            ["self", "fields", "latitudes", "longitudes", "dates", "time_window", "missions"],
        )

        window_params = inspect.signature(AlongTrack.geographic_point_in_r_dt).parameters
        self.assertEqual(
            list(window_params),
            ["self", "fields", "latitude", "longitude", "date", "radius", "time_window", "missions"],
        )

        window_batch_params = inspect.signature(
            AlongTrack.geographic_point_in_r_dt_batch
        ).parameters
        self.assertEqual(
            list(window_batch_params),
            ["self", "fields", "latitudes", "longitudes", "dates", "radius", "time_window", "missions"],
        )

    def test_eddy_method_signatures_match_expected_contract(self):
        self.assertEqual(
            list(inspect.signature(Eddy.get_eddy_tracks_from_times).parameters),
            ["self", "start_date", "end_date"],
        )
        self.assertEqual(
            list(inspect.signature(Eddy.eddy_with_track_id).parameters),
            ["self", "fields", "track_id"],
        )
        self.assertEqual(
            list(inspect.signature(Eddy.eddy_envelope_query).parameters),
            ["self", "track_id"],
        )
        self.assertEqual(
            list(inspect.signature(Eddy.along_track_points_near_eddy).parameters),
            ["self", "track_id", "fields"],
        )

    def test_nearest_neighbor_interpolator_uses_current_batch_query(self):
        grid = InterpGrid(
            np.array([-69.0, -68.0]),
            np.array([28.1]),
            datetime(2013, 3, 14, 23),
            "equirectangular",
        )
        fake_atdb = FakeAlongTrack()

        result = interpolate_using_atdb(
            grid,
            NearestNeighborInterpolator(time_window=timedelta(days=5)),
            atdb=fake_atdb,
        )

        self.assertEqual(len(fake_atdb.nearest_neighbor_calls), 1)
        call = fake_atdb.nearest_neighbor_calls[0]
        self.assertEqual(call["fields"], ["sla_filtered", "mission"])
        self.assertEqual(call["missions"], ["al"])
        self.assertEqual(call["time_window"], timedelta(days=5))
        self.assertEqual(call["latitudes"], [-69.0, -68.0])
        self.assertEqual(call["longitudes"], [28.1, 28.1])
        self.assertEqual(call["dates"], [datetime(2013, 3, 14, 23)] * 2)
        np.testing.assert_allclose(result.sla, np.array([[-69.0], [-68.0]]))

    def test_geographic_gaussian_interpolator_uses_radius_batch_query(self):
        grid = InterpGrid(
            np.array([-69.0]),
            np.array([28.1, 28.2]),
            datetime(2013, 3, 14, 23),
            "equirectangular",
        )
        fake_atdb = FakeAlongTrack()

        result = interpolate_using_atdb(
            grid,
            GeographicGaussianKernelInterpolator(
                length_scale=50_000.0,
                radius=123_456.0,
                time_window=timedelta(days=7),
            ),
            atdb=fake_atdb,
        )

        self.assertEqual(len(fake_atdb.geographic_window_calls), 1)
        call = fake_atdb.geographic_window_calls[0]
        self.assertEqual(call["fields"], ["sla_filtered", "distance"])
        self.assertEqual(call["radius"], 123_456.0)
        self.assertEqual(call["missions"], ["al"])
        self.assertEqual(call["time_window"], timedelta(days=7))
        self.assertEqual(call["latitudes"], [-69.0, -69.0])
        self.assertEqual(call["longitudes"], [28.1, 28.2])
        self.assertEqual(call["dates"], [datetime(2013, 3, 14, 23)] * 2)
        self.assertEqual(result.sla.shape, (1, 2))
        self.assertTrue(np.isfinite(result.sla).all())

    def test_projected_gaussian_interpolator_uses_projected_fields(self):
        grid = InterpGrid(
            np.array([-69.0]),
            np.array([28.1]),
            datetime(2013, 3, 14, 23),
            "equirectangular",
        )
        fake_atdb = FakeAlongTrack()

        result = interpolate_using_atdb(
            grid,
            ProjectedGaussianKernelInterpolator(
                length_scale=50_000.0,
                radius=200_000.0,
                time_window=timedelta(days=3),
            ),
            atdb=fake_atdb,
        )

        call = fake_atdb.geographic_window_calls[0]
        self.assertEqual(call["fields"], ["latitude", "longitude", "sla_filtered"])
        self.assertEqual(call["radius"], 200_000.0)
        self.assertEqual(result.sla.shape, (1, 1))
        self.assertTrue(np.isfinite(result.sla).all())


if __name__ == "__main__":
    unittest.main()
