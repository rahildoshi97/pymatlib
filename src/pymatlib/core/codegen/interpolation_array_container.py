import re
import numpy as np
from typing import List, Any
from pystencils.types import PsCustomType
from pystencilssfg import SfgComposer
from pystencilssfg.composer.custom import CustomGenerator
from pymatlib.core.interpolators import prepare_interpolation_arrays


class InterpolationArrayContainer(CustomGenerator):
    """Container for x-y interpolation arrays and methods."""
    def __init__(self, name: str, x_array: np.ndarray, y_array: np.ndarray):
        """Initialize the interpolation container."""
        super().__init__()

        # Validate inputs
        if not isinstance(name, str) or not name:
            raise TypeError("Name must be a non-empty string")

        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
            raise ValueError(f"'{name}' is not a valid C++ class name")

        if not isinstance(x_array, np.ndarray) or not isinstance(y_array, np.ndarray):
            raise TypeError("x_array and y_array must be numpy arrays")

        self.name = name
        self.x_array = x_array
        self.y_array = y_array

        # Prepare arrays and determine best method
        try:
            self.data = prepare_interpolation_arrays(x_array=self.x_array, y_array=self.y_array, verbose=False)
            self.method = self.data["method"]

            # Store arrays for binary search (always available)
            self.x_bs = self.data["x_bs"]
            self.y_bs = self.data["y_bs"]

            # Store arrays for double lookup if available
            if self.method == "double_lookup":
                self.x_eq = self.data["x_eq"]
                self.y_neq = self.data["y_neq"]
                self.y_eq = self.data["y_eq"]
                self.inv_delta_y_eq = self.data["inv_delta_y_eq"]
                self.idx_map = self.data["idx_map"]
                self.has_double_lookup = True
            else:
                self.has_double_lookup = False
        except Exception as e:
            raise ValueError(f"Failed to prepare interpolation arrays: {e}") from e

    def _generate_binary_search(self, sfg: SfgComposer):
        """Generate code for binary search interpolation."""
        x_bs_arr_values = ", ".join(str(v) for v in self.x_bs)
        y_bs_arr_values = ", ".join(str(v) for v in self.y_bs)

        y_target = sfg.var("y_target", "double")

        return [
            # Binary search arrays
            f"static constexpr std::array< double, {self.x_bs.shape[0]} > x_bs {{ {x_bs_arr_values} }}; \n"
            f"static constexpr std::array< double, {self.y_bs.shape[0]} > y_bs {{ {y_bs_arr_values} }}; \n",

            # Binary search method
            sfg.method("interpolateBS", returns=PsCustomType("[[nodiscard]] double"), inline=True, const=True)(
                sfg.expr("return interpolate_binary_search_cpp({}, *this);", y_target)
            )
        ]

    def _generate_double_lookup(self, sfg: SfgComposer) -> List[Any]:
        """Generate code for double lookup interpolation."""
        if not self.has_double_lookup:
            return []

        x_eq_arr_values = ", ".join(str(v) for v in self.x_eq)
        y_neq_arr_values = ", ".join(str(v) for v in self.y_neq)
        y_eq_arr_values = ", ".join(str(v) for v in self.y_eq)
        idx_mapping_arr_values = ", ".join(str(v) for v in self.idx_map)

        y_target = sfg.var("y_target", "double")

        return [
            # Double lookup arrays
            f"static constexpr std::array< double, {self.x_eq.shape[0]} > x_eq {{ {x_eq_arr_values} }}; \n"
            f"static constexpr std::array< double, {self.y_neq.shape[0]} > y_neq {{ {y_neq_arr_values} }}; \n"
            f"static constexpr std::array< double, {self.y_eq.shape[0]} > y_eq {{ {y_eq_arr_values} }}; \n"
            f"static constexpr double inv_delta_y_eq = {self.inv_delta_y_eq}; \n"
            f"static constexpr std::array< int, {self.idx_map.shape[0]} > idx_map {{ {idx_mapping_arr_values} }}; \n",

            # Double lookup method
            sfg.method("interpolateDL", returns=PsCustomType("[[nodiscard]] double"), inline=True, const=True)(
                sfg.expr("return interpolate_double_lookup_cpp({}, *this);", y_target)
            )
        ]

    def generate(self, sfg: SfgComposer) -> None:
        """Generate C++ code for the interpolation container."""
        sfg.include("<array>")
        sfg.include("pymatlib_interpolators/interpolate_binary_search_cpp.h")

        public_members = self._generate_binary_search(sfg)

        # Add interpolate method that uses recommended approach
        y_target = sfg.var("y_target", "double")
        if self.has_double_lookup:
            sfg.include("pymatlib_interpolators/interpolate_double_lookup_cpp.h")
            public_members.extend(self._generate_double_lookup(sfg))

            public_members.append(
                sfg.method("interpolate", returns=PsCustomType("[[nodiscard]] double"), inline=True, const=True)(
                    sfg.expr("return interpolate_double_lookup_cpp({}, *this);", y_target)
                )
            )
        else:
            public_members.append(
                sfg.method("interpolate", returns=PsCustomType("[[nodiscard]] double"), inline=True, const=True)(
                    sfg.expr("return interpolate_binary_search_cpp({}, *this);", y_target)
                )
            )

        # Generate the class
        sfg.klass(self.name)(
            sfg.public(*public_members)
        )
