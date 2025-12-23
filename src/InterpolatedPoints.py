import numpy as np
import numpy.typing as npt
from typing import Any
from xarray import DataArray, Dataset

from .InterpQueryPoints import InterpQueryPoints
from .Grid import InterpGrid


class InterpolatedPoints:
    def __init__(
        self, sla: npt.NDArray[np.floating[Any]], queryPoints: InterpQueryPoints
    ):
        self.queryPoints = queryPoints
        self.sla = sla

    def to_netcdf(self, filename: str):
        if isinstance(self.queryPoints, InterpGrid):
            df = DataArray(
                self.sla,
                coords=[("x", self.queryPoints.x_dim), ("y", self.queryPoints.y_dim)],
                name="sla",
                attrs={"projection": self.queryPoints.projection},
            )
        else:
            df = Dataset(
                {
                    "sla": self.sla,
                    "x": self.queryPoints.x,
                    "y": self.queryPoints.y,
                    "projection": self.queryPoints.projection,
                }
            )
        df.to_netcdf(filename)
