import os
from pycallgraph2 import PyCallGraph, Config
from pycallgraph2.output import GraphvizOutput
from pycallgraph2.globbing_filter import GlobbingFilter

# Define the path to the folder inside the apps directory
apps_directory = '/local/ca00xebo/repos/pymatlib/apps'
image_folder = os.path.join(apps_directory, 'callgraph_images')

# Create the folder if it does not exist
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

# Create configuration to focus on your specific functions
config = Config(min_depth=1, max_depth=10)
config.trace_filter = GlobbingFilter(
    include=[
        'pymatlib.core.codegen.interpolation_array_container.InterpolationArrayContainer.__init__',
        'pymatlib.core.codegen.interpolation_array_container.InterpolationArrayContainer.from_material',
        'pymatlib.core.interpolators',
        'pymatlib.core.interpolators.*',
        'pymatlib.core.interpolators.prepare_interpolation_arrays',
        'pymatlib.core.interpolators.E_eq_from_E_neq',
        'pymatlib.core.interpolators.create_idx_mapping',
        'pymatlib.core.codegen.interpolation_array_container.*',
        'pymatlib.core.data_handler.*',

        'assignment_converter',
        'interpolate_property',
        'check_equidistant',
        'check_strictly_increasing',
        'InterpolationArrayContainer',
        'prepare_interpolation_arrays',
        'E_eq_from_E_neq',
        'create_idx_mapping',
    ],
    exclude=[
        'pycallgraph2.*',
        '__main__.*',
        '*.append',  # Exclude common methods
        '*.join',
        # Exclude standard library functions that add noise
        # '__*__',  # Exclude magic methods
        '*.<module>',
        '*.<listcomp>',  # Exclude list comprehensions
        # '*.<genexpr>',  # Exclude generator expressions
        '*.<dictcomp>',  # Exclude dictionary comprehensions
    ]
)

# Configure the output with the full path
output_file = os.path.join(image_folder, 'array_container_callgraph.svg')

# Configure the output
graphviz = GraphvizOutput(
    output_file=output_file,
    font_name='Verdana',
    font_size=7,
    group_stdlib=True,
    output_type='svg',
    dpi=1200,
    include_timing=True,
)

# Run the specific part of your code with call graph tracking
with PyCallGraph(config=config, output=graphviz):
    import numpy as np
    from pystencils import fields
    from importlib.resources import files
    from pystencilssfg import SourceFileGenerator
    from pymatlib.parsing import create_material
    from pymatlib.core.codegen.interpolation_array_container import InterpolationArrayContainer
    from pymatlib.core.interpolators import prepare_interpolation_arrays

    # Call the specific functions directly to ensure they're traced
    T_eq = np.array([3243.15, 3253.15, 3263.15, 3273.15], dtype=np.float64)
    E_neq = np.array([1.68e10, 1.69e10, 1.70e10, 1.71e10], dtype=np.float64)
    result = prepare_interpolation_arrays(T_eq, E_neq)

    with SourceFileGenerator() as sfg:
        u = fields(f"u: double[2D]", layout='fzyx')

        T_bs = np.array([3243.15, 3248.15, 3258.15, 3278.15], dtype=np.float64)
        E_bs = np.array([1.68e10, 1.69e10, 1.70e10, 1.71e10], dtype=np.float64)

        custom_container = InterpolationArrayContainer("BinarySearchTests", T_bs, E_bs)
        sfg.generate(custom_container)

        custom_container = InterpolationArrayContainer("BinarySearchTests1", np.flip(T_bs), np.flip(E_bs))
        sfg.generate(custom_container)

        T_eq = np.array([3243.15, 3253.15, 3263.15, 3273.15], dtype=np.float64)
        E_neq = np.array([1.68e10, 1.69e10, 1.70e10, 1.71e10], dtype=np.float64)

        custom_container = InterpolationArrayContainer("DoubleLookupTests", T_eq, E_neq)
        sfg.generate(custom_container)

        custom_container = InterpolationArrayContainer("DoubleLookupTests1", np.flip(T_eq), np.flip(E_neq))
        sfg.generate(custom_container)

        yaml_path = files('pymatlib.data.alloys.SS304L').joinpath('SS304L.yaml')
        mat = create_material(yaml_path, u.center())
        arr_container = InterpolationArrayContainer.from_material("SS304L", mat)
        sfg.generate(arr_container)
