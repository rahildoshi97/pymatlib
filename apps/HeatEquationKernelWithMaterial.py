import sympy as sp
import pystencils as ps
from importlib.resources import files
from pystencilssfg import SourceFileGenerator
from sfg_walberla import Sweep
from pymatlib.core.assignment_converter import assignment_converter
from pymatlib.core.interpolators import DoubleLookupArrayContainer
from pymatlib.core.yaml_parser import create_alloy_from_yaml

with SourceFileGenerator() as sfg:
    data_type = "float64"  # if ctx.double_accuracy else "float32"

    u, u_tmp = ps.fields(f"u, u_tmp: {data_type}[2D]", layout='fzyx')
    thermal_diffusivity = sp.Symbol("thermal_diffusivity")
    thermal_diffusivity_out = ps.fields(f"thermal_diffusivity_out: {data_type}[2D]", layout='fzyx')
    dx, dt = sp.Symbol("dx"), sp.Symbol("dt")

    heat_pde = ps.fd.transient(u) - thermal_diffusivity * (ps.fd.diff(u, 0, 0) + ps.fd.diff(u, 1, 1))

    discretize = ps.fd.Discretization2ndOrder(dx=dx, dt=dt)
    heat_pde_discretized = discretize(heat_pde)
    heat_pde_discretized = heat_pde_discretized.args[1] + heat_pde_discretized.args[0].simplify()

    yaml_path = files('pymatlib.data.alloys.SS316L').joinpath('SS304L.yaml')
    mat = create_alloy_from_yaml(yaml_path, u.center())
    # arr_container = DoubleLookupArrayContainer("SS316L", mat.temperature_array, mat.energy_density_array)
    arr_container = DoubleLookupArrayContainer.from_material("SS316L", mat)
    sfg.generate(arr_container)

    # Convert assignments to pystencils format
    subexp, subs = assignment_converter(mat.thermal_diffusivity.assignments)

    subexp.append(ps.Assignment(thermal_diffusivity, mat.thermal_diffusivity.expr))

    ac = ps.AssignmentCollection(
        subexpressions=subexp,
        main_assignments=[
            ps.Assignment(u_tmp.center(), heat_pde_discretized),
            ps.Assignment(thermal_diffusivity_out.center(), thermal_diffusivity)
        ])

    # generate_sweep(ctx, 'HeatEquationKernelWithMaterial', ac, varying_parameters=((data_type, str(thermal_diffusivity)),))
    sweep = Sweep("HeatEquationKernelWithMaterial", ac)
    sfg.generate(sweep)
