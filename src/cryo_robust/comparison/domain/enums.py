from enum import Enum


class ImageSpace(str, Enum):
    # Pass both the raw value and the pretty label to the constructor
    REAL = ("real", "Real Space")
    FOURIER_REAL = ("fourier_real", "Fourier Space (Real Part)")
    FOURIER_IMAG = ("fourier_imag", "Fourier Space (Imaginary Part)")
    FOURIER_COMPLEX = ("fourier_complex", "Fourier Space")

    def __new__(cls, value, label):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.label = label
        return obj

    def __str__(self):
        return self.label


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
