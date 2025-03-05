import numpy as np
from pystencils import fields
from importlib.resources import files
from pystencilssfg import SourceFileGenerator
from pymatlib.core.yaml_parser import create_alloy_from_yaml
from pymatlib.core.interpolators import InterpolationArrayContainer


with SourceFileGenerator() as sfg:
    u = fields(f"u: double[2D]", layout='fzyx')

    T_bs = np.array([3243.15, 3248.15, 3258.15, 3278.15], dtype=np.float64)
    E_bs = np.array([1.68e10, 1.69e10, 1.70e10, 1.71e10], dtype=np.float64)

    custom_container = InterpolationArrayContainer("BinarySearchTests", T_bs, E_bs)
    sfg.generate(custom_container)

    T_eq = np.array([3243.15, 3253.15, 3263.15, 3273.15], dtype=np.float64)
    E_neq = np.array([1.68e10, 1.69e10, 1.70e10, 1.71e10], dtype=np.float64)

    custom_container = InterpolationArrayContainer("DoubleLookupTests", np.flip(T_eq), np.flip(E_neq))
    sfg.generate(custom_container)

    yaml_path = files('pymatlib.data.alloys.SS304L').joinpath('SS304L.yaml')
    mat = create_alloy_from_yaml(yaml_path, u.center())
    arr_container = InterpolationArrayContainer.from_material("SS304L", mat)
    sfg.generate(arr_container)
