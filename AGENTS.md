# AGENTS_MapInterp.md

## Purpose

MapInterp is a query-only consumer of OceanDB.

It should read along-track and eddy data from an existing OceanDB PostgreSQL/PostGIS database. It should not handle ingestion, downloading, database initialization, or schema changes.

## Main OceanDB Query Endpoints

MapInterp should primarily use these query classes:

- `OceanDB.data_access.along_track.AlongTrack`
- `OceanDB.data_access.eddy.Eddy`

### Along-track queries

`AlongTrack.geographic_point_in_r_dt(...)`

- Fetch along-track points within a spatial radius and time window around a target point.
- Best for: nearby observations for interpolation or map selection.

`AlongTrack.geographic_nearest_neighbors(...)`

- Fetch the nearest along-track observations to a target point within a time window.
- Best for: click-to-inspect or nearest-sample lookup.

### Eddy queries

`Eddy.get_eddy_tracks_from_times(...)`

- Return eddy track ids observed within a time range.
- Best for: discovering candidate eddies for a selected time window.

`Eddy.eddy_with_track_id(...)`

- Fetch all observations for a single eddy track.
- Best for: plotting or inspecting one eddy over time.

`Eddy.eddy_envelope_query(...)`

- Return a compact summary for one eddy track, including time bounds and basin ids.
- Best for: metadata lookup before downstream queries.

`Eddy.along_track_points_near_eddy(...)`

- Fetch along-track observations associated with a given eddy track.
- Best for: comparing eddy structure with nearby altimetry measurements.

## Usage Examples

```python
from datetime import datetime, timedelta

from OceanDB.data_access.along_track import AlongTrack
from OceanDB.data_access.eddy import Eddy

along = AlongTrack()
eddy = Eddy()

nearby_points = along.geographic_point_in_r_dt(
    fields=["latitude", "longitude", "date_time", "sla_filtered"],
    latitude=-39.1,
    longitude=54.7,
    date=datetime(2013, 1, 4, 23),
    radius=500_000.0,
    time_window=timedelta(days=10),
)

nearest_points = along.geographic_nearest_neighbors(
    fields=["latitude", "longitude", "date_time", "sla_filtered", "distance"],
    latitude=-69.0,
    longitude=28.1,
    date=datetime(2013, 1, 4, 23),
    time_window=timedelta(days=10),
)

track_ids = eddy.get_eddy_tracks_from_times(
    start_date=datetime(2013, 1, 1),
    end_date=datetime(2013, 2, 1),
)

if track_ids:
    eddy_track = eddy.eddy_with_track_id(
        fields=["latitude", "longitude", "date_time", "track", "cyclonic_type"],
        track_id=track_ids[0],
    )

    nearby_along_track = eddy.along_track_points_near_eddy(
        track_id=track_ids[0],
        fields=["latitude", "longitude", "date_time", "sla_filtered"],
    )
```

## Guidance

- Prefer OceanDB's `data_access` query classes over raw SQL.
- Expect query results as OceanDB `Dataset` objects.
- Keep MapInterp read-only with respect to the OceanDB database.
