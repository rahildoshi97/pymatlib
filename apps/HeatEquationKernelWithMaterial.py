import logging
import sympy as sp
import pystencils as ps
from importlib.resources import files
# from pystencils_walberla import CodeGeneration, generate_sweep
# script_dir = os.path.dirname(__file__)
# walberla_dir = os.path.join(script_dir, '..', '..', '..')
# sys.path.append(walberla_dir)
from pystencilssfg import SourceFileGenerator
from walberla.codegen import Sweep
from pymatlib.core.yaml_parser.api import create_material_from_yaml

logging.basicConfig(
    level=logging.INFO,  # DEBUG/INFO/WARNING
    format="%(asctime)s %(levelname)s %(name)s -> %(message)s"
)
# Silence matplotlib and other noisy libraries
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('fontTools').setLevel(logging.WARNING)
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

    from pathlib import Path
    # Relative path to the package
    # yaml_path = Path(__file__).parent.parent / "src" / "pymatlib" / "data" / "alloys" / "SS304L" / "SS304L.yaml"
    # yaml_path = Path(__file__).parent / 'SS304L_HeatEquationKernelWithMaterial.yaml'

    from importlib.resources import files
    # yaml_path = files('pymatlib.data.alloys.SS304L').joinpath('SS304L_comprehensive.yaml')
    yaml_path = files('apps').joinpath('SS304L_HeatEquationKernelWithMaterial.yaml')
    mat = create_material_from_yaml(yaml_path=yaml_path, T=u.center(), enable_plotting=False)
    subexp = [ps.Assignment(thermal_diffusivity, mat.thermal_diffusivity)]

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

    print(f"ac\n{ac}, type = {type(ac)}")

    # generate_sweep(ctx, 'HeatEquationKernelWithMaterial', ac, varying_parameters=((data_type, str(thermal_diffusivity)),))
    sweep = Sweep("HeatEquationKernelWithMaterial", ac)
    sfg.generate(sweep)
