import sympy as sp
from dataclasses import dataclass
from typing import List, Union


@dataclass
class ChemicalElement:
    name: str
    atomic_number: float
    atomic_mass: float
    melting_temperature: float
    boiling_temperature: float
    latent_heat_of_fusion: float
    latent_heat_of_vaporization: float


# Utility functions for element property interpolation
def interpolate(values: List[float], composition: List[float]) -> Union[float, sp.Expr]:
    """
    Interpolates a property based on its values and composition.

    Args:
        values (list): List of property values.
        composition (list): List of composition percentages.

    Returns:
        float: Interpolated property value.
    """
    '''c = np.asarray(composition)
    v = np.asarray(values)
    return float(np.sum(c * v))'''
    result = 0.
    for v, c in zip(values, composition):
        result += v * c
    return result


def interpolate_atomic_number(elements: List[ChemicalElement], composition: List[float]) -> float:
    """
    Interpolates the atomic number based on the elements and their composition.

    Args:
        elements (list[ChemicalElement]): List of elements.
        composition (list[float]): List of composition percentages.

    Returns:
        float: Interpolated atomic number.
    """
    values = [element.atomic_number for element in elements]
    return interpolate(values, composition)


def interpolate_atomic_mass(elements: List[ChemicalElement], composition: List[float]) -> float:
    """
    Interpolates the atomic mass based on the elements and their composition.

    Args:
        elements (list[ChemicalElement]): List of elements.
        composition (list[float]): List of composition percentages.

    Returns:
        float: Interpolated atomic mass.
    """
    values = [element.atomic_mass for element in elements]
    return interpolate(values, composition)


def interpolate_melting_temperature(elements: List[ChemicalElement], composition: List[float]) -> float:
    """
    Interpolates the melting temperature based on the elements and their composition.
    """
    values = [element.melting_temperature for element in elements]
    return interpolate(values, composition)


def interpolate_boiling_temperature(elements: List[ChemicalElement], composition: List[float]) -> float:
    """
    Interpolates the boiling temperature based on the elements and their composition.

    Args:
        elements (list[ChemicalElement]): List of elements.
        composition (list[float]): List of composition percentages.

    Returns:
        float: Interpolated boiling temperature.
    """
    values = [element.boiling_temperature for element in elements]
    return interpolate(values, composition)
