cmake_minimum_required( VERSION 3.24 )

project ( test_pymatlib )
enable_language(CUDA)

SET( CMAKE_CXX_STANDARD 20 )
SET( CMAKE_CXX_STANDARD_REQUIRED ON )

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

# GPU-related options
SET ( WALBERLA_BUILD_WITH_GPU_SUPPORT ON CACHE BOOL "" )
SET ( WALBERLA_BUILD_WITH_CUDA ON CACHE BOOL "" )

add_subdirectory(walberla)
add_subdirectory(sfg-walberla)

walberla_codegen_venv_require(
        -e ${CMAKE_SOURCE_DIR}/..
)
walberla_codegen_venv_populate()

add_executable ( HeatEquationWithMaterial )
target_sources( HeatEquationWithMaterial PRIVATE CodegenHeatEquationWithMaterial.cpp )
target_link_libraries( HeatEquationWithMaterial PRIVATE walberla::blockforest walberla::core walberla::field walberla::stencil walberla::timeloop walberla::vtk walberla::pde )

walberla_generate_sources( HeatEquationWithMaterial
        SCRIPTS HeatEquationKernelWithMaterial.py
        FILE_EXTENSIONS .cpp .hpp
)

add_executable ( HeatEquationWithMaterialCPU )
target_sources( HeatEquationWithMaterialCPU PRIVATE CodegenHeatEquationWithMaterialCPU.cpp)
target_link_libraries( HeatEquationWithMaterialCPU PRIVATE walberla::blockforest walberla::core walberla::field walberla::stencil walberla::timeloop walberla::vtk walberla::pde )

walberla_generate_sources( HeatEquationWithMaterialCPU
        SCRIPTS HeatEquationKernelWithMaterialCPU.py
        FILE_EXTENSIONS .cpp .hpp
)

add_executable ( HeatEquationWithMaterialGPU )
target_sources( HeatEquationWithMaterialGPU PRIVATE CodegenHeatEquationWithMaterialGPU.cpp)
target_link_libraries( HeatEquationWithMaterialGPU PRIVATE walberla::blockforest walberla::core walberla::gpu walberla::field walberla::stencil walberla::timeloop walberla::vtk walberla::pde )

# Set CUDA properties
set_property(TARGET HeatEquationWithMaterialGPU PROPERTY CUDA_SEPARABLE_COMPILATION ON)

walberla_generate_sources( HeatEquationWithMaterialGPU
        SCRIPTS HeatEquationKernelWithMaterialGPU.py
        FILE_EXTENSIONS .cu .hpp
)

set_source_files_properties(
        ${CMAKE_CURRENT_BINARY_DIR}/_gen/HeatEquationWithMaterialGPU/gen/HeatEquationKernelWithMaterialGPU.cu
        PROPERTIES LANGUAGE CUDA
)
