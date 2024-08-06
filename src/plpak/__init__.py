from .makePlasmaKinetics import *
from .plasmaKineticSystem import System
from .plasmaSystems import PlasmaSolver3T, N2Plasma_54Rxn, N2Plasma_54Rxn_2, N2PlasmaAleks, AirPlasmaRxn
from .integrator import Solution, updateFromYsp

# Date : 2023-07-025
from .mecWrapper import PlasmaMechanism, PlasmaSolver
from .extraModels import LaserModel, CombinedModels, photoIonizePPT

# define the version of the package
# __version__ = '0.1.0' 

## Change log ----------------------------------------------
# PlasmaMechanism: Reads the custom mechanism file and creates the plasma mechanism
# Custom mechanism file can take any rate expression and can also specify the energy exchange in multi-temperature model
# PlasmaSolver: Solves the plasma kinetics only from the plasma mechanism
# PlasmaSolver3T: Solves the plasma kinetics along with 3T model ( Tv for N2)
__version__ = '0.2.0'

## Change log ----------------------------------------------
# Updated: 2023-08-01