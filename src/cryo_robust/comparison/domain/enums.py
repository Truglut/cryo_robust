from enum import Enum


class AggregationStrategy(str, Enum):
    MEAN = "mean"
    ENERGY = "energy"

    @property
    def label(self):
        return {
            AggregationStrategy.MEAN: "Mean aggregation",
            AggregationStrategy.ENERGY: "Energy aggregation",
        }[self]

    def __str__(self):
        return self.value
