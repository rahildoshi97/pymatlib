import os
import sys
sys.path.append('/local/ca00xebo/repos/pymatlib')

from pycallgraph2 import PyCallGraph, Config
from pycallgraph2.output import GraphvizOutput
from pycallgraph2.globbing_filter import GlobbingFilter

# Import necessary modules once
import sympy as sp
import pystencils as ps
from importlib.resources import files
from pystencilssfg import SourceFileGenerator
from pymatlib.parsing import create_alloy_from_yaml
from pymatlib.core.assignment_converter import assignment_converter
from pymatlib.core.codegen.interpolation_array_container import InterpolationArrayContainer

# Define the path to the folder inside the apps directory
apps_directory = '/local/ca00xebo/repos/pymatlib/apps'
image_folder = os.path.join(apps_directory, 'callgraph_images')

# Create the folder if it does not exist
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

# Function to create a visualization for a specific target
def create_visualization(target_name, include_patterns, output_filename):
    print(f"Generating visualization for {target_name}...")

    # Create the full path for the output file
    output_file = os.path.join(image_folder, output_filename)

    # Create configuration
    config = Config(max_depth=10)
    config.trace_filter = GlobbingFilter(
        include=include_patterns,
        exclude=[
            'pycallgraph2.*',
            '*.append',
            '*.join',
        ]
    )

    # Configure the output
    graphviz = GraphvizOutput(
        output_file=output_file,
        font_name='Verdana',
        font_size=7,
        group_stdlib=True,
        output_type='svg',
        dpi=400
    )

    # Create minimal setup for all visualizations
    u = ps.fields("u: float64[2D]", layout='fzyx')
    yaml_path = files('pymatlib.data.alloys.SS304L').joinpath('SS304L.yaml')

    # Run the appropriate code based on the target
    with PyCallGraph(config=config, output=graphviz):
        if target_name == "create_alloy_from_yaml":
            mat = create_alloy_from_yaml(yaml_path, u.center())

        elif target_name == "InterpolationArrayContainer":
            mat = create_alloy_from_yaml(yaml_path, u.center())
            arr_container = InterpolationArrayContainer.from_material("SS304L", mat)
            with SourceFileGenerator() as sfg:
                sfg.generate(arr_container)

        elif target_name == "assignment_converter":
            mat = create_alloy_from_yaml(yaml_path, u.center())
            thermal_diffusivity = sp.Symbol("thermal_diffusivity")
            subexp, subs = assignment_converter(mat.thermal_diffusivity.assignments)
            subexp.append(ps.Assignment(thermal_diffusivity, mat.thermal_diffusivity.expr))

    print(f"Visualization saved to {output_file}")

# Generate each visualization
create_visualization(
    "create_alloy_from_yaml",
    ['pymatlib.core.parsing.*', 'create_alloy_from_yaml'],
    'create_alloy_callgraph.svg'
)

create_visualization(
    "InterpolationArrayContainer",
    ['pymatlib.core.interpolators.*', 'InterpolationArrayContainer.*'],
    'array_container_callgraph.svg'
)

create_visualization(
    "assignment_converter",
    ['pymatlib.core.assignment_converter.*', 'assignment_converter'],
    'assignment_converter_callgraph.svg'
)

print("All visualizations completed.")
