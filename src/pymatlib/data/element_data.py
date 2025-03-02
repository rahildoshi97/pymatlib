from pymatlib.data.constants import Constants
from pymatlib.core.elements import ChemicalElement

# NIST: National Institute of Standards and Technology
# RSC: Royal Society of Chemistry
# CRC: CRC Handbook of Chemistry and Physics

Ti = ChemicalElement(
    name="Titanium",
    atomic_number=22,  # Atomic number = 22 / Source: Periodic Table
    atomic_mass=47.867 * Constants.u,  # Atomic mass = 47.867 u / Source: NIST
    temperature_melt=1941,  # Melting temperature = 1941 K / Source: RSC
    temperature_boil=3560,  # Boiling temperature = 3560 K / Source: RSC
    latent_heat_of_fusion=18700,  # Latent heat of fusion = 18700 J/kg / Source: CRC
    latent_heat_of_vaporization=427000  # Latent heat of vaporization = 427000 J/kg / Source: CRC
)

Al = ChemicalElement(
    name="Aluminium",
    atomic_number=13,  # Atomic number = 13 / Source: Periodic Table
    atomic_mass=26.9815384 * Constants.u,  # Atomic mass = 26.9815384 u / Source: NIST
    temperature_melt=933.35,  # Melting temperature = 933.35 K / Source: RSC
    temperature_boil=2743,  # Boiling temperature = 2743 K / Source: RSC
    latent_heat_of_fusion=10700,  # Latent heat of fusion = 10700 J/kg / Source: CRC
    latent_heat_of_vaporization=284000  # Latent heat of vaporization = 284000 J/kg / Source: CRC
)

V = ChemicalElement(
    name="Vanadium",
    atomic_number=23,  # Atomic number = 23 / Source: Periodic Table
    atomic_mass=50.9415 * Constants.u,  # Atomic mass = 50.9415 u / Source: NIST
    temperature_melt=2183,  # Melting temperature = 2183 K / Source: RSC
    temperature_boil=3680,  # Boiling temperature = 3680 K / Source: RSC
    latent_heat_of_fusion=21500,  # Latent heat of fusion = 21500 J/kg / Source: CRC
    latent_heat_of_vaporization=444000  # Latent heat of vaporization = 444000 J/kg / Source: CRC
)

Fe = ChemicalElement(
    name="Iron",
    atomic_number=26,  # Atomic number = 26 / Source: Periodic Table
    atomic_mass=55.845 * Constants.u,  # Atomic mass = 55.845 u / Source: NIST
    temperature_melt=1809,  # Melting temperature = 1809 K / Source: RSC
    temperature_boil=3134,  # Boiling temperature = 3134 K / Source: RSC
    latent_heat_of_fusion=13800,  # Latent heat of fusion = 13800 J/kg / Source: CRC
    latent_heat_of_vaporization=340000  # Latent heat of vaporization = 340000 J/kg / Source: CRC
)

Cr = ChemicalElement(
    name="Chromium",
    atomic_number=24,  # Atomic number = 24 / Source: Periodic Table
    atomic_mass=51.9961 * Constants.u,  # Atomic mass = 51.9961 u / Source: NIST
    temperature_melt=2180,  # Melting temperature = 2180 K / Source: RSC
    temperature_boil=2944,  # Boiling temperature = 2944 K / Source: RSC
    latent_heat_of_fusion=16500,  # Latent heat of fusion = 16500 J/kg / Source: CRC
    latent_heat_of_vaporization=344000  # Latent heat of vaporization = 344000 J/kg / Source: CRC
)

Mn = ChemicalElement(
    name="Manganese",
    atomic_number=25,  # Atomic number = 25 / Source: Periodic Table
    atomic_mass=54.938045 * Constants.u,  # Atomic mass = 54.938045 u / Source: NIST
    temperature_melt=1519,  # Melting temperature = 1519 K / Source: RSC
    temperature_boil=2334,  # Boiling temperature = 2334 K / Source: RSC
    latent_heat_of_fusion=12500,  # Latent heat of fusion = 12500 J/kg / Source: CRC
    latent_heat_of_vaporization=220000  # Latent heat of vaporization = 220000 J/kg / Source: CRC
)

Ni = ChemicalElement(
    name="Nickel",
    atomic_number=28,  # Atomic number = 28 / Source: Periodic Table
    atomic_mass=58.6934 * Constants.u,  # Atomic mass = 58.6934 u / Source: NIST
    temperature_melt=1728,  # Melting temperature = 1728 K / Source: RSC
    temperature_boil=3186,  # Boiling temperature = 3186 K / Source: RSC
    latent_heat_of_fusion=17200,  # Latent heat of fusion = 17200 J/kg / Source: CRC
    latent_heat_of_vaporization=377000  # Latent heat of vaporization = 377000 J/kg / Source: CRC
)

Mo = ChemicalElement(
    name="Molybdenum",
    atomic_number=42,
    atomic_mass=95.96 * Constants.u,
    temperature_melt=2896,  # Melting temperature = 2896K (2623°C)
    temperature_boil=4912,  # Boiling temperature = 4912K (4639°C)
    latent_heat_of_fusion=37480,  # Latent heat of fusion = 37.48 kJ/mol
    latent_heat_of_vaporization=598000  # Latent heat of vaporization = 598 kJ/mol
)

# This dictionary maps chemical symbols (strings) to their corresponding ChemicalElement instances,
# allowing the parser to convert composition keys from the YAML file (like 'Fe': 0.675) to actual ChemicalElement objects needed by the Alloy class.
element_map = {
    'Ti': Ti,
    'Al': Al,
    'V': V,
    'Fe': Fe,
    'Cr': Cr,
    'Mn': Mn,
    'Ni': Ni,
    'Mo': Mo
}
