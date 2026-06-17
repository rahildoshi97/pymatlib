"""Test imports work correctly."""

from pathlib import Path
import pytest
import sympy as sp
from materforge import create_material

def test_all_imports():
    """Test that all modules can be imported without circular dependencies."""
    try:
        # Test main package imports
        import materforge
        import materforge.core
        import materforge.algorithms
        import materforge.parsing
        import materforge.visualization
        # Test core imports
        from materforge.core.materials import Material
        from materforge.core.evaluator import MaterialEvaluator
        from materforge import MaterialEvaluator as TopLevelEvaluator
        # Test catalog imports
        from materforge import list_materials, load_material, get_material_path
        # Test CLI imports
        from materforge.cli import main, validate_entry
        # Test parsing imports
        from materforge.parsing.validation.property_validator import validate_monotonic_property
        # Test algorithm imports
        from materforge.algorithms.interpolation import interpolate_value
        from materforge.algorithms.piecewise_builder import PiecewiseBuilder
        # Test visualization imports
        from materforge.visualization.plotters import PropertyVisualizer
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_basic_material_creation():
    """Test basic material creation functionality."""
    T = sp.Symbol('T')
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent
    yaml_path_Al = project_root / "src" / "materforge" / "data" / "materials" / "Al.yaml"
    yaml_path_SS304L = project_root / "src" / "materforge" / "data" / "materials" / "1.4301.yaml"
    # Test aluminum material creation if file exists
    if yaml_path_Al.exists():
        try:
            mat_Al = create_material(yaml_path=yaml_path_Al, dependency=T, enable_plotting=False)
            assert mat_Al is not None
            assert mat_Al.name == "Aluminum"
        except Exception as e:
            pytest.fail(f"Failed to create aluminum material: {e}")
    else:
        pytest.fail(f"Aluminum YAML file not found: {yaml_path_Al}")
    # Test steel material creation if file exists
    if yaml_path_SS304L.exists():
        try:
            mat_SS304L = create_material(yaml_path=yaml_path_SS304L, dependency=T, enable_plotting=False)
            assert mat_SS304L is not None
            assert "Steel" in mat_SS304L.name or "1.4301" in mat_SS304L.name
        except Exception as e:
            pytest.fail(f"Failed to create steel material: {e}")
    else:
        pytest.fail(f"Steel YAML file not found: {yaml_path_SS304L}")

def test_circular_dependencies():
    """Test specifically for circular import dependencies."""
    import sys

    def _drop_materforge_modules():
        for name in [n for n in sys.modules if n.startswith('materforge')]:
            del sys.modules[name]

    # Snapshot the already-imported modules so we can restore them afterwards.
    # Without this, the re-import below replaces every materforge module object
    # in sys.modules, leaving class references bound by other test modules
    # pointing at the old objects (isinstance checks then fail).
    saved = {name: mod for name, mod in sys.modules.items()
             if name.startswith('materforge')}
    try:
        _drop_materforge_modules()
        # Import in different orders to catch circular dependencies
        import materforge.core.materials
        import materforge
        import materforge.algorithms.interpolation
        import materforge.visualization.plotters
        # Import main package last
        import materforge
    except ImportError as e:
        pytest.fail(f"Circular dependency detected: {e}")
    finally:
        _drop_materforge_modules()
        sys.modules.update(saved)

if __name__ == "__main__":
    test_all_imports()
    test_basic_material_creation()
    test_circular_dependencies()
    print("✅ All import tests passed - no circular dependencies detected")
