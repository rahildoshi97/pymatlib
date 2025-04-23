from dataclasses import dataclass

@dataclass
class Constants:
    """
    A dataclass to store fundamental constants.
    Attributes:
        ROOM_TEMPERATURE (float): Room temperature in Kelvin.
        N_A (float): Avogadro's number, the number of constituent particles (usually atoms or molecules) in one mole of a given substance.
        AMU (float): Atomic mass unit in kilograms.
        E (float): Elementary charge, the electric charge carried by a single proton or the magnitude of the electric charge carried by a single electron.
        SPEED_OF_LIGHT (float): Speed of light in vacuum in meters per second.
    """
    ROOM_TEMPERATURE: float = 298.15  # Room temperature in Kelvin
    N_A: float = 6.022141e23  # Avogadro's number, /mol
    AMU: float = 1.660538e-27  # Atomic mass unit, kg
    E: float = 1.60217657e-19  # Elementary charge, C
    SPEED_OF_LIGHT: float = 0.299792458e9  # Speed of light, m/s
