# Creating Your First Material Simulation

This tutorial will guide you through creating a basic heat equation simulation using [pymatlib](https://i10git.cs.fau.de/rahil.doshi/pymatlib) and [pystencils](https://pycodegen.pages.i10git.cs.fau.de/pystencils/).
It builds upon the existing [waLBerla tutorial for code generation](https://walberla.net/doxygen/tutorial_codegen01.html), adding material property handling with pymatlib.

## Prerequisites

Before starting, ensure you have:
- Completed the [Getting Started](getting_started.md) tutorial
- Installed pystencils and [pystencilssfg](https://pycodegen.pages.i10git.cs.fau.de/pystencils-sfg/)
- Basic understanding of heat transfer and the heat equation
- Created the `simple_steel.yaml` file from the [Getting Started](getting_started.md) tutorial

## Overview

We'll create a 2D heat equation simulation with temperature-dependent material properties.
The main enhancement compared to the original waLBerla tutorial is that we'll use pymatlib to handle material properties that vary with temperature.

The steps are:

1. Define a temperature field
2. Create an alloy with temperature-dependent properties
3. Set up the heat equation with material properties
4. Generate code for the simulation
5. Run the simulation

## Step 1: Define the Temperature Field

First, we'll create a temperature field using pystencils:

```python
import pystencils as ps
from pystencilssfg import SourceFileGenerator

with SourceFileGenerator() as sfg:
    data_type = "float64"

    # Define temperature fields and output field for thermal diffusivity
    u, u_tmp = ps.fields(f"u, u_tmp: {data_type}[2D]", layout='fzyx')
    thermal_diffusivity_out = ps.fields(f"thermal_diffusivity_out: {data_type}[2D]", layout='fzyx')
```

## Step 2: Set Up the Heat Equation

Now we'll set up the heat equation with temperature-dependent material properties:

```python
import sympy as sp

# Create symbolic variables for the equation
dx = sp.Symbol("dx")
dt = sp.Symbol("dt")
thermal_diffusivity = sp.Symbol("thermal_diffusivity")

# Define the heat equation using finite differences
heat_pde = ps.fd.transient(u) - thermal_diffusivity * (ps.fd.diff(u, 0, 0) + ps.fd.diff(u, 1, 1))

# Discretize the PDE
discretize = ps.fd.Discretization2ndOrder(dx=dx, dt=dt)
heat_pde_discretized = discretize(heat_pde)
heat_pde_discretized = heat_pde_discretized.args[1] + heat_pde_discretized.args[0].simplify()
```

## Step 3: Create an Alloy

This is where pymatlib enhances the original tutorial.
We'll load the alloy from a YAML file:

```python
from pymatlib.core.yaml_parser import create_alloy_from_yaml

# Load the SimpleSteel alloy definition we created earlier
simple_steel_alloy = create_alloy_from_yaml("path/to/simple_steel.yaml", u.center())
```

The second parameter `u.center()` links the alloy to the temperature field, making material properties dependent on the local temperature.

## Step 4: Convert Material Property Assignments

We need to convert the material property assignments to a format pystencils can understand:

```python
from pymatlib.core.assignment_converter import assignment_converter

# Access the thermal diffusivity property directly from the material
# The property is already defined in the YAML file or computed automatically

# Convert material property assignments to pystencils format
subexp, subs = assignment_converter(simple_steel_alloy.thermal_diffusivity.assignments)

# Add the thermal diffusivity expression to subexpressions
subexp.append(ps.Assignment(thermal_diffusivity, simple_steel_alloy.thermal_diffusivity.expr))
```

The `assignment_converter` function transforms pymatlib's internal representation of material properties into pystencils assignments. 
It handles type conversions and creates properly typed symbols for the code generation.

## Step 5: Generate Code for the Simulation

Finally, we'll create the assignment collection and generate optimized C++ code for the simulation:

```python
from sfg_walberla import Sweep
from pymatlib.core.interpolators import InterpolationArrayContainer

# Create assignment collection with subexpressions and main assignments
ac = ps.AssignmentCollection(
    subexpressions=subexp,  # Include the subexpressions from pymatlib
    main_assignments=[
        ps.Assignment(u_tmp.center(), heat_pde_discretized),
        ps.Assignment(thermal_diffusivity_out.center(), thermal_diffusivity)
    ])

# Generate the sweep
sweep = Sweep("HeatEquationKernelWithMaterial", ac)
sfg.generate(sweep)

# Create the container for energy-temperature conversion
# The name "SimpleSteel" is just an identifier for the generated class
arr_container = InterpolationArrayContainer.from_material("SimpleSteel", simple_steel_alloy)
sfg.generate(arr_container)
```

## Step 6: Create the Interpolation Container

For energy-temperature conversion, create an interpolation container:

```python
from pymatlib.core.interpolators import InterpolationArrayContainer

# Create the container for energy-temperature conversion
arr_container = InterpolationArrayContainer.from_material("SS304L", alloy)
sfg.generate(arr_container)
```

## Complete Example

Here's the complete example:

```python
import sympy as sp
import pystencils as ps
from pystencilssfg import SourceFileGenerator
from sfg_walberla import Sweep
from pymatlib.core.assignment_converter import assignment_converter
from pymatlib.core.interpolators import InterpolationArrayContainer
from pymatlib.core.yaml_parser import create_alloy_from_yaml

with SourceFileGenerator() as sfg:
    data_type = "float64"

    # Define fields
    u, u_tmp = ps.fields(f"u, u_tmp: {data_type}[2D]", layout='fzyx')
    thermal_diffusivity = sp.Symbol("thermal_diffusivity")
    thermal_diffusivity_out = ps.fields(f"thermal_diffusivity_out: {data_type}[2D]", layout='fzyx')
    dx, dt = sp.Symbol("dx"), sp.Symbol("dt")

    # Define heat equation
    heat_pde = ps.fd.transient(u) - thermal_diffusivity * (ps.fd.diff(u, 0, 0) + ps.fd.diff(u, 1, 1))

    # Discretize the PDE
    discretize = ps.fd.Discretization2ndOrder(dx=dx, dt=dt)
    heat_pde_discretized = discretize(heat_pde)
    heat_pde_discretized = heat_pde_discretized.args[1] + heat_pde_discretized.args[0].simplify()

    # Create alloy with temperature-dependent properties
    mat = create_alloy_from_yaml("simple_steel.yaml", u.center())

    # Convert material property assignments to pystencils format
    subexp, subs = assignment_converter(mat.thermal_diffusivity.assignments)
    subexp.append(ps.Assignment(thermal_diffusivity, mat.thermal_diffusivity.expr))

    # Create assignment collection with the converted subexpressions
    ac = ps.AssignmentCollection(
        subexpressions=subexp,
        main_assignments=[
            ps.Assignment(u_tmp.center(), heat_pde_discretized),
            ps.Assignment(thermal_diffusivity_out.center(), thermal_diffusivity)
        ])

    # Generate the sweep
    sweep = Sweep("HeatEquationKernelWithMaterial", ac)
    sfg.generate(sweep)

    # Generate interpolation container for energy-temperature conversion
    arr_container = InterpolationArrayContainer.from_material("SimpleSteel", mat)
    sfg.generate(arr_container)
```

## Understanding the Assignment Converter

The `assignment_converter` function is a key component of pymatlib that bridges the gap between symbolic material property representations and pystencils code generation. It:

1. Takes a list of `Assignment` objects from pymatlib
2. Converts them to pystencils-compatible assignments with proper typing
3. Returns both the converted assignments and a mapping of symbols

This allows material properties defined in YAML files to be seamlessly integrated into pystencils code generation, enabling temperature-dependent properties in simulations.

The main ways pymatlib enhances the simulation are:

- Material property handling: Properties are defined in YAML and accessed via the material object
- Assignment conversion: The assignment_converter transforms material property assignments to pystencils format
- Energy-temperature conversion: The InterpolationArrayContainer enables efficient conversion between energy density and temperature

## Next Steps

Now that you've created your first simulation with temperature-dependent material properties, you can:
- Learn about [energy-temperature conversion](../how-to/energy_temperature_conversion.md)
- Explore more complex [material properties](../explanation/material_properties.md)
- Understand the [API reference](../reference/api/alloy.md) for advanced usage
- Check the original [waLBerla tutorial](https://walberla.net/doxygen/tutorial_codegen01.html) for details on running the simulation
