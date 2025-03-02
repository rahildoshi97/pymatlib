from dataclasses import dataclass


@dataclass
class Constants:
    """
    A dataclass to store fundamental constants.

    Attributes:
        temperature_room (float): Room temperature in Kelvin.
        N_a (float): Avogadro's number, the number of constituent particles (usually atoms or molecules) in one mole of a given substance.
        u (float): Atomic mass unit in kilograms.
        e (float): Elementary charge, the electric charge carried by a single proton or the magnitude of the electric charge carried by a single electron.
        speed_of_light (float): Speed of light in vacuum in meters per second.
    """
    temperature_room: float = 298.15  # Room temperature in Kelvin
    N_a: float = 6.022141e23  # Avogadro's number, /mol
    u: float = 1.660538e-27  # Atomic mass unit, kg
    e: float = 1.60217657e-19  # Elementary charge, C
    speed_of_light: float = 0.299792458e9  # Speed of light, m/s
