from enum import Enum, auto


class Space(Enum):
    REAL = auto()
    FOURIER_REAL = auto()
    FOURIER_IMAG = auto()

    @property
    def label(self):
        return{
            Space.REAL: "Real Space",
            Space.FOURIER_REAL: "Fourier Space (Real Part)",
            Space.FOURIER_IMAG: "Fourier Space (Imaginary Part)"
        }[self]

    def __str__(self):
        return self.label
    

class AggregationStrategy(str, Enum):
    MEAN = "mean"
    ENERGY = "energy"