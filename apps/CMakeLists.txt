cmake_minimum_required( VERSION 3.24 )

project ( test_pymatlib )

#   waLBerla Options

SET ( WALBERLA_BUILD_TESTS OFF CACHE BOOL "" )
SET ( WALBERLA_BUILD_BENCHMARKS OFF CACHE BOOL "" )
SET ( WALBERLA_BUILD_TOOLS OFF CACHE BOOL "" )
SET ( WALBERLA_BUILD_TUTORIALS OFF CACHE BOOL "" )
SET ( WALBERLA_BUILD_SHOWCASES OFF CACHE BOOL "" )
SET ( WALBERLA_BUILD_DOC OFF CACHE BOOL "" )

SET ( WALBERLA_BUILD_WITH_PYTHON OFF CACHE BOOL "" )
SET ( WALBERLA_BUILD_WITH_CODEGEN OFF CACHE BOOL "" )
SET ( WALBERLA_BUILD_WITH_OPENMP ON CACHE BOOL "" )
SET ( WALBERLA_BUILD_WITH_MPI OFF CACHE BOOL "" )


add_subdirectory(walberla)
add_subdirectory(sfg-walberla)
add_subdirectory( ../src/pymatlib/core/cpp pymatlib-build-cpp )

walberla_codegen_venv_install(
        -e ${CMAKE_SOURCE_DIR}/..
)

add_executable ( HeatEquationWithMaterial )
target_sources( HeatEquationWithMaterial PRIVATE CodegenHeatEquationWithMaterial.cpp )
target_link_libraries( HeatEquationWithMaterial PRIVATE walberla::blockforest walberla::core walberla::field walberla::stencil walberla::timeloop walberla::vtk walberla::pde )
target_link_libraries( HeatEquationWithMaterial PRIVATE pymatlib_interpolators )

pystencilssfg_generate_target_sources( HeatEquationWithMaterial
        SCRIPTS HeatEquationKernelWithMaterial.py
        FILE_EXTENSIONS .cpp .hpp
        OUTPUT_MODE standalone
)
