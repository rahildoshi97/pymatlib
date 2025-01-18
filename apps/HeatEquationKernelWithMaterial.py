import sympy as sp
import pystencils as ps
# from pystencils_walberla import CodeGeneration, generate_sweep
# script_dir = os.path.dirname(__file__)
# walberla_dir = os.path.join(script_dir, '..', '..', '..')
# sys.path.append(walberla_dir)
from pystencilssfg import SourceFileGenerator
from sfg_walberla import Sweep
from pymatlib.data.alloys.SS316L import SS316L
from pymatlib.core.assignment_converter import assignment_converter

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

    mat = SS316L.create_SS316L(u.center())

    # Convert assignments to pystencils format
    print("Print statements")
    print(mat.thermal_diffusivity)
    print(mat.thermal_diffusivity.expr)
    print(mat.thermal_diffusivity.assignments)
    subexp, subs = assignment_converter(mat.thermal_diffusivity.assignments)
    print(f"subexp\n{subexp}")
    print(f"subs\n{subs}")
    # subexp_ed, _ = assignment_converter(mat.energy_density.assignments)
    # print(f"subexp_ed =\n{subexp_ed}")

    subexp.append(ps.Assignment(thermal_diffusivity, mat.thermal_diffusivity.expr))

    ac = ps.AssignmentCollection(
        subexpressions=subexp,
        main_assignments=[
            ps.Assignment(u_tmp.center(), heat_pde_discretized),
            ps.Assignment(thermal_diffusivity_out.center(), thermal_diffusivity)
        ])

    # ac = ac.new_with_substitutions(subs)

    # ac = ps.simp.simplifications.add_subexpressions_for_divisions(ac)
    # ac = ps.simp.simplifications.add_subexpressions_for_field_reads(ac)
    # ac = ps.simp.sympy_cse(ac)

    print(f"ac\n{ac}")

    # generate_sweep(ctx, 'HeatEquationKernelWithMaterial', ac, varying_parameters=((data_type, str(thermal_diffusivity)),))
    sweep = Sweep("HeatEquationKernelWithMaterial", ac)
    sfg.generate(sweep)
