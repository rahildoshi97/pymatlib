add_subdirectory(walberla)
add_subdirectory(sfg-walberla)

walberla_add_executable ( NAME HeatEquationWithMaterial
        FILES CodegenHeatEquationWithMaterial.cpp
        DEPENDS blockforest core field stencil timeloop vtk pde )

pystencilssfg_generate_target_sources( HeatEquationWithMaterial
        SCRIPTS HeatEquationKernelWithMaterial.py
        FILE_EXTENSIONS .cpp .h
        OUTPUT_MODE standalone
)