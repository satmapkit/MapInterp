from InterpQueryPoints import InterpQueryPoints


import numpy as np
import numpy.typing as npt
from typing import Any, Literal
from datetime import datetime


class InterpGrid(InterpQueryPoints):
    """
    Grid of interpolated query points
    """

    def __init__(
        self,
        x_dim: npt.NDArray[np.floating[Any]],
        y_dim: npt.NDArray[np.floating[Any]],
        date: datetime,
        projection: Literal[
            "equirectangular", "transversemercator"
        ] = "equirectangular",
        projection_params: dict[str, float] = {},
    ):
        """
        x: n-array of x axis values
        y: m-array of y axis values
        date: central date around which to search
        projection: which projection to use.
        projection_params: parameters for the projection of choice.
          - 'lon0' for transverse mercator
        """
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.date = date

        y, x = np.meshgrid(y_dim, x_dim)
        super().__init__(x, y, [date] * x.size, projection, projection_params)

    def params(self):
        return {
            "x": self.x_dim,
            "y": self.y_dim,
            "t": self.date,
            "projection": self.projection,
        }
