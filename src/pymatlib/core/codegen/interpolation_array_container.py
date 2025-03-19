import re
import numpy as np
from pystencils.types import PsCustomType
from pystencilssfg import SfgComposer
from pystencilssfg.composer.custom import CustomGenerator
from pymatlib.core.interpolators import prepare_interpolation_arrays


class InterpolationArrayContainer(CustomGenerator):
    """Container for energy-temperature interpolation arrays and methods.

    This class stores temperature and energy density arrays and generates C++ code
    for efficient bilateral conversion between these properties. It supports both
    binary search interpolation (O(log n)) and double lookup interpolation (O(1))
    with automatic method selection based on data characteristics.

    Attributes:
        name (str): Name for the generated C++ class.
        T_array (np.ndarray): Array of temperature values (must be monotonically increasing).
        E_array (np.ndarray): Array of energy density values corresponding to T_array.
        method (str): Interpolation method selected ("binary_search" or "double_lookup").
        T_bs (np.ndarray): Temperature array prepared for binary search.
        E_bs (np.ndarray): Energy array prepared for binary search.
        has_double_lookup (bool): Whether double lookup interpolation is available.

    If has_double_lookup is True, the following attributes are also available:
        T_eq (np.ndarray): Equidistant temperature array for double lookup.
        E_neq (np.ndarray): Non-equidistant energy array for double lookup.
        E_eq (np.ndarray): Equidistant energy array for double lookup.
        inv_delta_E_eq (float): Inverse of the energy step size for double lookup.
        idx_map (np.ndarray): Index mapping array for double lookup.

    Examples:
        >>> import numpy as np
        >>> from pystencilssfg import SfgComposer
        >>> from pymatlib.core.codegen.interpolation_array_container import InterpolationArrayContainer
        >>>
        >>> # Create temperature and energy arrays
        >>> T = np.array([300, 600, 900, 1200], dtype=np.float64)
        >>> E = np.array([1e9, 2e9, 3e9, 4e9], dtype=np.float64)
        >>>
        >>> # Create and generate the container
        >>> with SfgComposer() as sfg:
        >>>     container = InterpolationArrayContainer("MyMaterial", T, E)
        >>>     sfg.generate(container)
    """
    def __init__(self, name: str, temperature_array: np.ndarray, energy_density_array: np.ndarray):
        """Initialize the interpolation container.
        Args:
            name (str): Name for the generated C++ class.
            temperature_array (np.ndarray): Array of temperature values (K).
                Must be monotonically increasing.
            energy_density_array (np.ndarray): Array of energy density values (J/m³)
                corresponding to temperature_array.
        Raises:
            ValueError: If arrays are empty, have different lengths, or are not monotonic.
            TypeError: If name is not a string or arrays are not numpy arrays.
        """
        super().__init__()

        # Validate inputs
        if not isinstance(name, str) or not name:
            raise TypeError("Name must be a non-empty string")

        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
            raise ValueError(f"'{name}' is not a valid C++ class name")

        if not isinstance(temperature_array, np.ndarray) or not isinstance(energy_density_array, np.ndarray):
            raise TypeError("Temperature and energy arrays must be numpy arrays")

        self.name = name
        self.T_array = temperature_array
        self.E_array = energy_density_array

        # Prepare arrays and determine best method
        try:
            self.data = prepare_interpolation_arrays(T_array=self.T_array, E_array=self.E_array, verbose=False)
            self.method = self.data["method"]

            # Store arrays for binary search (always available)
            self.T_bs = self.data["T_bs"]
            self.E_bs = self.data["E_bs"]

            # Store arrays for double lookup if available
            if self.method == "double_lookup":
                self.T_eq = self.data["T_eq"]
                self.E_neq = self.data["E_neq"]
                self.E_eq = self.data["E_eq"]
                self.inv_delta_E_eq = self.data["inv_delta_E_eq"]
                self.idx_map = self.data["idx_map"]
                self.has_double_lookup = True
            else:
                self.has_double_lookup = False
        except Exception as e:
            raise ValueError(f"Failed to prepare interpolation arrays: {e}") from e

    @classmethod
    def from_material(cls, name: str, material):
        """Create an interpolation container from a material object.
        Args:
            name (str): Name for the generated C++ class.
            material: Material object with temperature and energy properties.
                Must have energy_density_temperature_array and energy_density_array attributes.
        Returns:
            InterpolationArrayContainer: Container with arrays for interpolation.
        """
        return cls(name, material.energy_density_temperature_array, material.energy_density_array)

    def _generate_binary_search(self, sfg: SfgComposer):
        """Generate code for binary search interpolation.
        Args:
            sfg (SfgComposer): Source file generator composer.
        Returns:
            list: List of public members for the C++ class.
        """
        T_bs_arr_values = ", ".join(str(v) for v in self.T_bs)
        E_bs_arr_values = ", ".join(str(v) for v in self.E_bs)

        E_target = sfg.var("E_target", "double")

        return [
            # Binary search arrays
            f"static constexpr std::array< double, {self.T_bs.shape[0]} > T_bs {{ {T_bs_arr_values} }}; \n"
            f"static constexpr std::array< double, {self.E_bs.shape[0]} > E_bs {{ {E_bs_arr_values} }}; \n",

            # Binary search method
            sfg.method("interpolateBS", returns=PsCustomType("[[nodiscard]] double"), inline=True, const=True)(
                sfg.expr("return interpolate_binary_search_cpp({}, *this);", E_target)
            )
        ]

    def _generate_double_lookup(self, sfg: SfgComposer):
        """Generate code for double lookup interpolation.
        Args:
            sfg (SfgComposer): Source file generator composer.
        Returns:
            list: List of public members for the C++ class.
        """
        if not self.has_double_lookup:
            return []

        T_eq_arr_values = ", ".join(str(v) for v in self.T_eq)
        E_neq_arr_values = ", ".join(str(v) for v in self.E_neq)
        E_eq_arr_values = ", ".join(str(v) for v in self.E_eq)
        idx_mapping_arr_values = ", ".join(str(v) for v in self.idx_map)

        E_target = sfg.var("E_target", "double")

        return [
            # Double lookup arrays
            f"static constexpr std::array< double, {self.T_eq.shape[0]} > T_eq {{ {T_eq_arr_values} }}; \n"
            f"static constexpr std::array< double, {self.E_neq.shape[0]} > E_neq {{ {E_neq_arr_values} }}; \n"
            f"static constexpr std::array< double, {self.E_eq.shape[0]} > E_eq {{ {E_eq_arr_values} }}; \n"
            f"static constexpr double inv_delta_E_eq = {self.inv_delta_E_eq}; \n"
            f"static constexpr std::array< int, {self.idx_map.shape[0]} > idx_map {{ {idx_mapping_arr_values} }}; \n",

            # Double lookup method
            sfg.method("interpolateDL", returns=PsCustomType("[[nodiscard]] double"), inline=True, const=True)(
                sfg.expr("return interpolate_double_lookup_cpp({}, *this);", E_target)
            )
        ]

    def generate(self, sfg: SfgComposer):
        """Generate C++ code for the interpolation container.
        This method generates a C++ class with the necessary arrays and methods
        for temperature-energy interpolation.
        Args:
            sfg (SfgComposer): Source file generator composer.
        """
        sfg.include("<array>")
        sfg.include("pymatlib_interpolators/interpolate_binary_search_cpp.h")

        public_members = self._generate_binary_search(sfg)

        # Add double lookup if available
        if self.has_double_lookup:
            sfg.include("pymatlib_interpolators/interpolate_double_lookup_cpp.h")
            public_members.extend(self._generate_double_lookup(sfg))

        # Add interpolate method that uses recommended approach
        E_target = sfg.var("E_target", "double")
        if self.has_double_lookup:
            public_members.append(
                sfg.method("interpolate", returns=PsCustomType("[[nodiscard]] double"), inline=True, const=True)(
                    sfg.expr("return interpolate_double_lookup_cpp({}, *this);", E_target)
                )
            )
        else:
            public_members.append(
                sfg.method("interpolate", returns=PsCustomType("[[nodiscard]] double"), inline=True, const=True)(
                    sfg.expr("return interpolate_binary_search_cpp({}, *this);", E_target)
                )
            )

        # Generate the class
        sfg.klass(self.name)(
            sfg.public(*public_members)
        )
