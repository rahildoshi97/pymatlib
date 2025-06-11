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
config = Config()
config.trace_filter = GlobbingFilter(
    include=[
        # 'pymatlib.core.alloy.*',
        # 'pymatlib.core.elements.*',
        'pymatlib.core.yaml_parser.*',
        # 'pymatlib.core.assignment_converter.*',
        'pymatlib.core.interpolators.*',
        'pymatlib.core.interpolators.interpolate_property.*',
        'pymatlib.core.typedefs.*',
        'pymatlib.core.models.*',
        'pymatlib.core.data_handler.*',
        'InterpolationArrayContainer.*',

        'create_alloy_from_yaml',
        'assignment_converter',
        'interpolate_property',
        # 'Assignment',
        'ArrayTypes',
        # 'PropertyTypes',
        'MaterialProperty',
        'density_by_thermal_expansion',
        'thermal_diffusivity_by_heat_conductivity',
        'energy_density_standard',
        'energy_density_enthalpy_based',
        'energy_density_total_enthalpy',
        'read_data_from_file',
        # 'wrapper',
        'check_equidistant',
        'check_strictly_increasing',
        'ChemicalElement',
        # 'interpolate_atomic_mass',
        # 'interpolate_atomic_number',
        # 'interpolate_temperature_boil',
        'evalf',
        'pymatlib.core.yaml_parser.<lambda>',  # Specifically include yaml_parser lambda
        '*.<lambda>',
    ],
    exclude=[
        'pycallgraph2.*',
        '*.append',  # Exclude common methods
        '*.join',
        '*.<module>',
        '*.<listcomp>',  # Exclude list comprehensions
        '*.<genexpr>',  # Exclude generator expressions
        '*.<dictcomp>',  # Exclude dictionary comprehensions
        # '*.<*>',  # Exclude all comprehensions
        'pymatlib.core.alloy.<lambda>',
        'pymatlib.core.typedefs.Assignment',
        'pymatlib.core.typedefs.MaterialProperty',
        'pymatlib.core.typedefs.MaterialProperty.__post_init__',
    ]
)

# Configure the output with the full path
output_file = os.path.join(image_folder, 'yaml_parser_callgraph_reduced.svg')

# Configure the output
graphviz = GraphvizOutput(
    output_file=output_file,
    font_name='Verdana',
    font_size=7,
    group_stdlib=True,
    output_type='svg',
    dpi=1200,
    include_timing=False,
)

# Run the specific part of your code with call graph tracking
with PyCallGraph(config=config, output=graphviz):
    # Import necessary modules
    import sympy as sp
    import pystencils as ps
    from importlib.resources import files
    # from pystencilssfg import SourceFileGenerator
    # from pymatlib.core.assignment_converter import assignment_converter
    # from pymatlib.core.codegen.interpolation_array_container import InterpolationArrayContainer
    from pymatlib.core.yaml_parser import create_alloy_from_yaml

    u = ps.fields("u: float64[2D]", layout='fzyx')
    yaml_path = files('pymatlib.data.alloys.SS304L').joinpath('SS304L.yaml')
    mat = create_alloy_from_yaml(yaml_path, u.center())

    yaml_path1 = files('pymatlib.data.alloys.SS304L').joinpath('SS304L.yaml')
    mat1 = create_alloy_from_yaml(yaml_path1, u.center())

    mat2 = create_alloy_from_yaml(yaml_path, 1600.)

    mat3 = create_alloy_from_yaml(yaml_path1, 1600.)

    '''with SourceFileGenerator() as sfg:
        yaml_path = files('pymatlib.data.alloys.SS304L').joinpath('SS304L.yaml')
        mat = create_alloy_from_yaml(yaml_path, u.center())
        
        arr_container = InterpolationArrayContainer.from_material("SS304L", mat)
        sfg.generate(arr_container)

        # Convert assignments to pystencils format
        subexp, subs = assignment_converter(mat.thermal_diffusivity.assignments)
        thermal_diffusivity = sp.Symbol("thermal_diffusivity")
        subexp.append(ps.Assignment(thermal_diffusivity, mat.thermal_diffusivity.expr))'''
