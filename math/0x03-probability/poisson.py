#!/usr/bin/env python3
"""This module defines the Poisson class that represents a
poisson distribution."""


class Poisson():
    """This class calculates the poisson distribution of a given data."""

    def __init__(self, data=None, lambtha=1.):
        """Initialize poisson distribution."""
        if lambtha < 0 or lambtha == 0:
            raise ValueError("lambtha must be a positive value")

        if data is None:
            self.lambtha = lambtha

        else:
            if type(data) != list:
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.lambtha = float(sum(data) / len(data))
