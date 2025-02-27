//======================================================================================================================
//
//! \file CodegenHeatEquationWithMaterial.cpp
//! \author Rahil Doshi <rahil.doshi@fau.de>
//
//======================================================================================================================

#include "blockforest/Initialization.h"
#include "blockforest/communication/UniformBufferedScheme.h"

#include "core/Environment.h"
#include "core/math/Constants.h"

#include "field/AddToStorage.h"
#include "field/communication/PackInfo.h"
#include "field/vtk/VTKWriter.h"

#include "pde/boundary/Neumann.h"

#include "stencil/D2Q9.h"

#include "timeloop/SweepTimeloop.h"

#include "gen/HeatEquationKernelWithMaterial.hpp"
#include "interpolate_binary_search_cpp.h"

namespace walberla
{
typedef GhostLayerField< real_t, 1 > ScalarField;

void swapFields(StructuredBlockForest& blocks, BlockDataID uID, BlockDataID uTmpID)
{
   for (auto block = blocks.begin(); block != blocks.end(); ++block)
   {
      ScalarField* u     = block->getData< ScalarField >(uID);
      ScalarField* u_tmp = block->getData< ScalarField >(uTmpID);

      u->swapDataPointers(u_tmp);
   }
}

void initDirichletBoundaryNorth(shared_ptr< StructuredBlockForest > blocks, BlockDataID uId, BlockDataID uTmpId)
{
   for (auto block = blocks->begin(); block != blocks->end(); ++block)
   {
      if (blocks->atDomainYMaxBorder(*block))
      {
         ScalarField* u     = block->getData< ScalarField >(uId);
         ScalarField* u_tmp = block->getData< ScalarField >(uTmpId);

         CellInterval xyz = u->xyzSizeWithGhostLayer();

         xyz.yMin() = xyz.yMax();
         for (auto cell = xyz.begin(); cell != xyz.end(); ++cell)
         {
            const Vector3< real_t > p = blocks->getBlockLocalCellCenter(*block, *cell);
            //  Set the dirichlet boundary to f(x) = 1 + sin(x) * x^2
            real_t v          = real_c(2000.0 + 1500 * std::sin(2 * math::pi * p[0]) * p[0] * p[0]);
            u->get(*cell)     = v;
            u_tmp->get(*cell) = v;
         }
      }
   }
}

int main(int argc, char** argv)
{
   mpi::Environment env(argc, argv);

   /////////////////////////////
   /// SIMULATION PARAMETERS ///
   /////////////////////////////

   //  Ensure matching aspect ratios of cells and domain.
   const uint_t xCells = uint_c(25);
   const uint_t yCells = uint_c(25);

   const real_t xSize = real_c(1.0);
   const real_t ySize = real_c(1.0);

   const uint_t xBlocks = uint_c(1);
   const uint_t yBlocks = uint_c(1);

   const uint_t processes = uint_c(MPIManager::instance()->numProcesses());

   if (processes != xBlocks * yBlocks)
   { WALBERLA_ABORT("The number of processes must be equal to the number of blocks!"); }

   const real_t dx = xSize / real_c(xBlocks * xCells + uint_t(1));
   const real_t dy = ySize / real_c(yBlocks * yCells + uint_t(1));

   WALBERLA_CHECK_FLOAT_EQUAL(dx, dy);

   const real_t dt    = real_c(1);
   const real_t thermal_diffusivity = real_c(1.0);

   ///////////////////////////
   /// BLOCK STORAGE SETUP ///
   ///////////////////////////

   auto aabb = math::AABB(real_c(0.5) * dx, real_c(0.5) * dy, real_c(0.0), xSize - real_c(0.5) * dx,
                          ySize - real_c(0.5) * dy, dx);

   shared_ptr< StructuredBlockForest > blocks = blockforest::createUniformBlockGrid(
      aabb, xBlocks, yBlocks, uint_c(1), xCells, yCells, 1, true, false, false, false);

   //////////////
   /// FIELDS ///
   //////////////

   BlockDataID uFieldId    = field::addToStorage< ScalarField >(blocks, "u", real_c(2000.0), field::fzyx, uint_c(1));
   BlockDataID uTmpFieldId = field::addToStorage< ScalarField >(blocks, "u_tmp", real_c(2000.0), field::fzyx, uint_c(1));
   BlockDataID thermalDiffusivityFieldId    = field::addToStorage< ScalarField >(blocks, "thermal_diffusivity", real_c(0.0), field::fzyx, uint_c(1));

   /////////////////////
   /// COMMUNICATION ///
   /////////////////////

   blockforest::communication::UniformBufferedScheme< stencil::D2Q9 > commScheme(blocks);
   commScheme.addPackInfo(make_shared< field::communication::PackInfo< ScalarField > >(uFieldId));

   //////////////////////////
   /// DIRICHLET BOUNDARY ///
   //////////////////////////

   initDirichletBoundaryNorth(blocks, uFieldId, uTmpFieldId);

   ////////////////////////
   /// NEUMANN BOUNDARY ///
   ////////////////////////

   pde::NeumannDomainBoundary< ScalarField > neumann(*blocks, uFieldId);

   neumann.excludeBoundary(stencil::N);
   neumann.excludeBoundary(stencil::B);
   neumann.excludeBoundary(stencil::T);

   ////////////////
   /// TIMELOOP ///
   ////////////////

   SweepTimeloop timeloop(blocks, uint_c(2e4));

   timeloop.add() << BeforeFunction(commScheme, "Communication") << BeforeFunction(neumann, "Neumann Boundaries")
                  << Sweep(HeatEquationKernelWithMaterial(thermalDiffusivityFieldId, uFieldId, uTmpFieldId, dt, dx), "HeatEquationKernelWithMaterial")
                  << AfterFunction([blocks, uFieldId, uTmpFieldId]() { swapFields(*blocks, uFieldId, uTmpFieldId); },
                                   "Swap");

   auto vtkOutput = vtk::createVTKOutput_BlockData(*blocks, "vtk", 200, 0, false, "vtk_out",
                                                   "simulation_step", false, true, true, false, 0);

   auto tempWriter =
      make_shared< field::VTKWriter< ScalarField > >(uFieldId, "temperature");
   vtkOutput->addCellDataWriter(tempWriter);

   auto kappaWriter =
      make_shared< field::VTKWriter< ScalarField > >(thermalDiffusivityFieldId, "thermal_diffusivity");
   vtkOutput->addCellDataWriter(kappaWriter);

   timeloop.addFuncAfterTimeStep(vtk::writeFiles(vtkOutput), "VTK Output");

   timeloop.run();

   return EXIT_SUCCESS;
}
}


void test_performance() {
    // Configuration parameters
    constexpr int warmupSteps = 2;
    constexpr int outerIterations = 5;
    constexpr int numCells = 64*64*64;

    // Setup test data
    SS316L test;
    std::vector<double> random_energies(numCells);

    // Generate random values
    std::random_device rd;
    std::mt19937 gen(rd());
    const double E_min = SS316L::E_neq.front() * 0.8;
    const double E_max = SS316L::E_neq.back() * 1.2;
    std::uniform_real_distribution<double> dist(E_min, E_max);

    // Fill random energies
    for(auto& E : random_energies) {
        E = dist(gen);
    }

    // Warmup runs
    std::cout << "Performing warmup steps..." << std::endl;
    for(int i = 0; i < warmupSteps; ++i) {
        for(const double& E : random_energies) {
            volatile double result = test.interpolateDL(E);
        }
    }
    for(int i = 0; i < warmupSteps; ++i) {
        for(const double& E : random_energies) {
            volatile double result = interpolate_binary_search_cpp(
                    SS316L::T_eq, E, SS316L::E_neq);
        }
    }

    // Performance measurement
    std::cout << "\nStarting performance measurement..." << std::endl;
    std::vector<double> timings_binary;
    std::vector<double> timings_double_lookup;

    for(int iter = 0; iter < outerIterations; ++iter) {
        std::cout << "\nIteration " << iter + 1 << "/" << outerIterations << std::endl;

        // Double Lookup timing
        {
            const auto start1 = std::chrono::high_resolution_clock::now();
            for(const double& E : random_energies) {
                volatile double result = test.interpolateDL(E);
            }
            const auto end1 = std::chrono::high_resolution_clock::now();
            const auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(
                end1 - start1).count();
            timings_double_lookup.push_back(static_cast<double>(duration1));

            std::cout << "Double Lookup - Iteration time: " << duration1 << " μs" << std::endl;
        }

        // Binary Search timing
        {
            const auto start2 = std::chrono::high_resolution_clock::now();
            for(const double& E : random_energies) {
                volatile double result = interpolate_binary_search_cpp(
                    SS316L::T_eq, E, SS316L::E_neq);
            }
            const auto end2 = std::chrono::high_resolution_clock::now();
            const auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(
                end2 - start2).count();
            timings_binary.push_back(static_cast<double>(duration2));

            std::cout << "Binary Search - Iteration time: " << duration2 << " μs" << std::endl;
        }
    }

    // Calculate and print statistics
    auto calc_stats = [](const std::vector<double>& timings) {
        double sum = std::accumulate(timings.begin(), timings.end(), 0.0);
        double mean = sum / static_cast<double>(timings.size());
        double sq_sum = std::inner_product(timings.begin(), timings.end(),
                                         timings.begin(), 0.0);
        double stdev = std::sqrt(sq_sum / static_cast<double>(timings.size()) - mean * mean);
        return std::make_pair(mean, stdev);
    };

    auto [binary_mean, binary_stdev] = calc_stats(timings_binary);
    auto [lookup_mean, lookup_stdev] = calc_stats(timings_double_lookup);

    std::cout << "\nPerformance Results (" << numCells << " cells, "
              << outerIterations << " iterations):" << std::endl;
    std::cout << "Binary Search:" << std::endl;
    std::cout << "  Mean time: " << binary_mean << " ± " << binary_stdev << " μs" << std::endl;
    std::cout << "  Per cell: " << binary_mean/numCells << " μs" << std::endl;

    std::cout << "Double Lookup:" << std::endl;
    std::cout << "  Mean time: " << lookup_mean << " ± " << lookup_stdev << " μs" << std::endl;
    std::cout << "  Per cell: " << lookup_mean/numCells << " μs" << std::endl;
}


int main(int argc, char** argv)
{
   walberla::main(argc, argv);

   /*constexpr SS316L_1 test_1;
   const double result_1 = test_1.interpolateDL(7.88552550e+09);
   const double result_2 = test_1.interpolateDL(1.02864255e+10);
   std::cout << result_1 << std::endl;
   std::cout << result_2 << std::endl;*/

   test_performance();
}
