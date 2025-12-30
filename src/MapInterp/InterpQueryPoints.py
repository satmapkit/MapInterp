import numpy as np
import numpy.typing as npt
from typing import Any, Literal
from datetime import datetime

from OceanDB.utils.projections import (
    spherical_transverse_mercator_to_latitude_longitude,
)


class InterpQueryPoints:
    """
    Base class for interpolated query points
    """

    def __init__(
        self,
        x: npt.NDArray[np.floating[Any]],
        y: npt.NDArray[np.floating[Any]],
        dates: list[datetime],
        projection: Literal[
            "equirectangular", "transversemercator"
        ] = "equirectangular",
        projection_params: dict[str, float] = {},
    ):
        """
        x: n by m array of query x location
        y: n by m array of query y location
        dates: n*m list of query datetime

        Each query point is uniquely identified by a latitude, longitude, and date.
        """
        if x.shape != y.shape or y.size != len(dates):
            raise ValueError("Inputs must be the same shape")
        self.x = x
        self.y = y
        self.dates = dates
        self.projection = projection

        if projection == "equirectangular":
            lat, lon = x, y
        elif projection == "transversemercator":
            lat, lon = spherical_transverse_mercator_to_latitude_longitude(
                x, y, **projection_params
            )
        else:
            raise ValueError("Unrecognized projection")

        self.lat = lat
        self.lon = lon

    def params(self) -> dict[str, npt.NDArray[np.floating[Any]] | str]:
        """
        Format output metadata
        """
        return {
            "x": self.x,
            "y": self.y,
            "t": self.dates,
            "projection": self.projection,
        }
