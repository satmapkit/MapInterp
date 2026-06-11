from datetime import datetime, timedelta
from typing import (
    Generator,
    TypeVar,
    Generic,
    Any,
    TypedDict,
    get_args,
    Mapping,
    Optional,
    Callable,
    Iterable,
    Literal
)
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from numpy.typing import NDArray
import numpy as np

class QueryParams: ...

@dataclass
class NearestNeighborParams[K](QueryParams):
    fields: list[K]
    time_window: timedelta


@dataclass
class GeographicWindowParams[K](QueryParams):
    fields: list[K]
    radius: float
    time_window: timedelta

@dataclass
class QuerySpec:
    params: QueryParams
    name: str

QueryOutputSingle = Any
QueryOutput = Generator[QueryOutputSingle]


@dataclass
class InterpPoint:
    latitude: float
    longigude: float
    date: datetime


@dataclass
class Points:
    """
    Points for a data source query
    """

    latitudes: Iterable[float]
    longitudes: Iterable[float]
    dates: Iterable[datetime]

    def to_interp_points(self) -> Generator[InterpPoint]:
        for lat, lon, date in zip(self.latitudes, self.longitudes, self.dates):
            yield InterpPoint(
                lat,
                lon,
                date,
            )


DataSourceOutputSingle = dict[str, QueryOutputSingle]

@dataclass
class DataSourceOutput:
    output: dict[str, QueryOutput]

    def batches(self) -> Generator[DataSourceOutputSingle]:
        keys = self.output.keys()
        for values in zip(*self.output.values()):
            yield dict(zip(keys, values))



class DataSource(ABC):
    def fetch_one[P](
        self, params: P | None, points: Points, func: Callable[[P, Points], QueryOutput]
    ) -> QueryOutput | None:
        if params is None:
            return None
        return func(params, points)

    def fetch(self, specs: list[QuerySpec], points: Points) -> DataSourceOutput:
        output = []
        for spec in specs:
            if spec.name == "nearest_neighbor":
                output.append(self.nearest_neighbor(spec.params, points))
            elif spec.name == "geographic_window":
                output.append(self.nearest_neighbor(spec.params))
        )

    @abstractmethod
    def nearest_neighbor(
        self, params: NearestNeighborParams, points: Points
    ) -> QueryOutput: ...

    @abstractmethod
    def geographic_window(
        self, params: GeographicWindowParams, points: Points
    ) -> QueryOutput: ...


class Interpolator[K1, K2, V](ABC):
    @abstractmethod
    def data_sources(self) -> DataSourceParams[K1, K2]: ...

    @abstractmethod
    def interpolate(
        self, data: DataSourceOutputSingle, interpPoint: InterpPoint
    ) -> V: ...

    def fetch_and_interpolate(self, source: DataSource, points: Points) -> Generator[V]:
        data_all = source.fetch(self.data_sources(), points)
        for data_batch, interpPoint in zip(
            data_all.batches(), points.to_interp_points()
        ):
            yield self.interpolate(data_batch, interpPoint)


from OceanDB.data_access.along_track import AlongTrack, along_track_fields, Mission


class AlongTrackSource(DataSource[along_track_fields, along_track_fields]):
    def __init__(
        self, atdb: AlongTrack = AlongTrack(), missions: Optional[list[Mission]] = None
    ):
        self.atdb = atdb
        self.missions = missions

    def nearest_neighbor(self, params: NearestNeighborParams, points):
        args = {**asdict(params), **asdict(points)}
        if self.missions is None:
            return self.atdb.geographic_nearest_neighbors_batch(**args)
        else:
            return self.atdb.geographic_nearest_neighbors_batch(
                **args, missions=self.missions
            )

    def geographic_window(self, params: GeographicWindowParams, points):
        args = {**asdict(params), **asdict(points)}
        if self.missions is None:
            return self.atdb.geographic_point_in_r_dt_batch(**args)
        else:
            return self.atdb.geographic_point_in_r_dt_batch(
                **args, missions=self.missions
            )


class NearestNeighborInterpolator(Interpolator):
    def data_sources(self) -> DataSourceParams[along_track_fields, along_track_fields]:
        fields : list[along_track_fields] = ["sla_filtered"]
        return {
            "nearest_neighbor": NearestNeighborParams(
                fields, timedelta(days=10)
            )
        }

    def interpolate(self, interpPoint) -> float: ...


my_interpolator = NearestNeighborInterpolator()
my_source = noaaSource("pH")

result = interpolate(my_source, my_interpolator)
