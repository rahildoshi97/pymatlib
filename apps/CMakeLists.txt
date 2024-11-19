cmake_minimum_required( VERSION 3.24 )

project ( test_pymatlib )

#   waLBerla Options

SET ( WALBERLA_BUILD_TESTS OFF )
SET ( WALBERLA_BUILD_BENCHMARKS OFF )
SET ( WALBERLA_BUILD_TOOLS OFF )
SET ( WALBERLA_BUILD_TUTORIALS OFF )
SET ( WALBERLA_BUILD_SHOWCASES OFF )
SET ( WALBERLA_BUILD_DOC OFF )

SET ( WALBERLA_BUILD_WITH_PYTHON OFF )
SET ( WALBERLA_BUILD_WITH_CODEGEN OFF )
SET ( WALBERLA_BUILD_WITH_OPENMP ON )
SET ( WALBERLA_BUILD_WITH_MPI OFF )

add_subdirectory( walberla )
add_subdirectory( sfg-walberla )

walberla_add_executable ( NAME HeatEquationWithMaterial
        FILES CodegenHeatEquationWithMaterial.cpp
        DEPENDS blockforest core field stencil timeloop vtk pde )

pystencilssfg_generate_target_sources( HeatEquationWithMaterial
        SCRIPTS HeatEquationKernelWithMaterial.py
        FILE_EXTENSIONS .cpp .h
        OUTPUT_MODE standalone
)
