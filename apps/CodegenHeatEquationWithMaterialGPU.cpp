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

#include "stencil/D3Q19.h"

#include "timeloop/SweepTimeloop.h"

#include "core/timing/RemainingTimeLogger.h"
#include "core/timing/TimingPool.h"

// GPU-specific includes
#include "gpu/AddGPUFieldToStorage.h"
#include "gpu/DeviceSelectMPI.h"
#include "gpu/FieldCopy.h"
#include "gpu/GPUWrapper.h"
#include "gpu/HostFieldAllocator.h"
#include "gpu/ParallelStreams.h"
#include "gpu/communication/MemcpyPackInfo.h"
#include "gpu/communication/UniformGPUScheme.h"

#include "gen/HeatEquationKernelWithMaterialGPU.hpp"

namespace walberla
{
///////////////////////
/// Typedef Aliases ///
///////////////////////

typedef GhostLayerField< real_t, 1 > ScalarField;
typedef gpu::GPUField<real_t> GPUScalarField;

void swapFields(StructuredBlockForest& blocks, BlockDataID uID, BlockDataID uTmpID)
{
   for (auto block = blocks.begin(); block != blocks.end(); ++block)
   {
      GPUScalarField* u     = block->getData< GPUScalarField >(uID);
      GPUScalarField* u_tmp = block->getData< GPUScalarField >(uTmpID);

      u->swapDataPointers(u_tmp);
   }
}

void initDirichletBoundaryNorth(const shared_ptr<StructuredBlockForest>& blocks,
                               BlockDataID uId, BlockDataID uTmpId,
                               BlockDataID uCpuId, BlockDataID uTmpCpuId)
{
   for (auto block = blocks->begin(); block != blocks->end(); ++block)
   {
      // Initialize on CPU first
      if (blocks->atDomainYMaxBorder(*block))
      {
         ScalarField* u     = block->getData< ScalarField >(uCpuId);
         ScalarField* u_tmp = block->getData< ScalarField >(uTmpCpuId);

         CellInterval xyz = u->xyzSizeWithGhostLayer();
         xyz.yMin() = xyz.yMax();
         for (auto cell = xyz.begin(); cell != xyz.end(); ++cell)
         {
            const Vector3< real_t > p = blocks->getBlockLocalCellCenter(*block, *cell);
            //  Set the dirichlet boundary to f(x) = 1 + sin(x) * x^2
            // real_t v          = real_c(3400.0 + 1700 * std::sin(2 * math::pi * p[0]) * p[1] * p[2]);
            real_t v          = real_c(3800.0);
            u->get(*cell)     = v;
            u_tmp->get(*cell) = v;
         }
      }
   }
   // Copy to GPU
   gpu::fieldCpy<GPUScalarField, ScalarField>(blocks, uId, uCpuId);
   gpu::fieldCpy<GPUScalarField, ScalarField>(blocks, uTmpId, uTmpCpuId);
}

int main(int argc, char** argv)
{
   mpi::Environment env(argc, argv);

   // GPU device selection
   gpu::selectDeviceBasedOnMpiRank();

   /////////////////////////////
   /// SIMULATION PARAMETERS ///
   /////////////////////////////

   //  Ensure matching aspect ratios of cells and domain.
   constexpr uint_t x = 256;
   const uint_t xCells = uint_c(x);
   const uint_t yCells = uint_c(x);
   const uint_t zCells = uint_c(x);

   const real_t xSize = real_c(1.0);
   const real_t ySize = real_c(1.0);
   const real_t zSize = real_c(1.0);

   const uint_t xBlocks = uint_c(1);
   const uint_t yBlocks = uint_c(1);
   const uint_t zBlocks = uint_c(1);

   const uint_t processes = uint_c(MPIManager::instance()->numProcesses());

   if (processes != xBlocks * yBlocks * zBlocks)
   { WALBERLA_ABORT("The number of processes must be equal to the number of blocks!"); }

   const real_t dx = xSize / real_c(xBlocks * xCells + uint_c(1));
   const real_t dy = ySize / real_c(yBlocks * yCells + uint_c(1));
   const real_t dz = zSize / real_c(zBlocks * zCells + uint_c(1));

   WALBERLA_CHECK_FLOAT_EQUAL(dx, dy);

   const real_t dt    = real_c(1);
   uint_t timeSteps = uint_c(2e4);
   uint_t vtkWriteFrequency = uint_c(200);

   ///////////////////////////
   /// BLOCK STORAGE SETUP ///
   ///////////////////////////

   auto aabb = math::AABB(real_c(0.5) * dx, real_c(0.5) * dy, real_c(0.5) * dz, xSize - real_c(0.5) * dx,
                          ySize - real_c(0.5) * dy, zSize - real_c(0.5) * dz);

   shared_ptr< StructuredBlockForest > blocks = blockforest::createUniformBlockGrid(
      aabb, xBlocks, yBlocks, zBlocks, xCells, yCells, zCells, true, false, false, false);

   //////////////
   /// FIELDS ///
   //////////////

   // CPU fields for initialization and VTK output
   auto allocator = make_shared<gpu::HostFieldAllocator<real_t>>(); // use pinned memory for faster transfers
   BlockDataID uFieldCpuId = field::addToStorage<ScalarField>(blocks, "u_cpu", real_c(300.0), field::fzyx, uint_c(1), allocator);
   BlockDataID uTmpFieldCpuId = field::addToStorage<ScalarField>(blocks, "u_tmp_cpu", real_c(300.0), field::fzyx, uint_c(1), allocator);
   BlockDataID alphaFieldCpuId = field::addToStorage<ScalarField>(blocks, "thermal_diffusivity_cpu", real_c(0.0), field::fzyx, uint_c(1), allocator);

   // GPU fields for computation
   BlockDataID uFieldId = gpu::addGPUFieldToStorage<ScalarField>(blocks, uFieldCpuId, "u", true);
   BlockDataID uTmpFieldId = gpu::addGPUFieldToStorage<ScalarField>(blocks, uTmpFieldCpuId, "u_tmp", true);
   BlockDataID alphaFieldId = gpu::addGPUFieldToStorage<ScalarField>(blocks, alphaFieldCpuId, "thermal_diffusivity", true);

   /////////////////////
   /// COMMUNICATION ///
   /////////////////////

   constexpr bool cudaEnabledMPI = false; // Set to true if CUDA-aware MPI is available
   gpu::communication::UniformGPUScheme<stencil::D3Q19> commScheme(blocks, cudaEnabledMPI);
   //auto packInfo = make_shared<gpu::communication::UniformGPUScheme<stencil::D3Q19>::PackInfo>(uFieldId);
   auto packInfo = make_shared<gpu::communication::MemcpyPackInfo<GPUScalarField>>(uFieldId);
   //auto packInfo = std::make_shared<lbm_generated::UniformGeneratedGPUPdfPackInfo< GPUScalarField >>(uFieldId);
   commScheme.addPackInfo(packInfo);

   //////////////////////////
   /// DIRICHLET BOUNDARY ///
   //////////////////////////

   initDirichletBoundaryNorth(blocks, uFieldId, uTmpFieldId, uFieldCpuId, uTmpFieldCpuId);

   ////////////////////////
   /// NEUMANN BOUNDARY ///
   ////////////////////////

   // Note: GPU version of Neumann boundary would need to be implemented
   pde::NeumannDomainBoundary< ScalarField > neumann(*blocks, uFieldCpuId);

   neumann.excludeBoundary(stencil::N);

   ////////////////
   /// TIMELOOP ///
   ////////////////

   // GPU block size configuration
   Vector3<int32_t> gpuBlockSize(256, 1, 1);

   SweepTimeloop timeloop(blocks, timeSteps);

   // Create GPU streams for better performance
   int streamHighPriority = 0;
   int streamLowPriority = 0;
   WALBERLA_GPU_CHECK(gpuDeviceGetStreamPriorityRange(&streamLowPriority, &streamHighPriority));

   timeloop.add() << BeforeFunction([&]() {
                     neumann(); // Apply Neumann boundaries on CPU
                     gpu::fieldCpy<GPUScalarField, ScalarField>(blocks, uFieldId, uFieldCpuId); // Copy updated boundary values to GPU
                     commScheme.communicate(); // GPU communication
                     }, "Communication and Boundaries")
                  << Sweep(HeatEquationKernelWithMaterialGPU(alphaFieldId, uFieldId, uTmpFieldId, dt, dx), "HeatEquationKernelWithMaterialGPU")
                  << AfterFunction([blocks, uFieldId, uTmpFieldId]() { swapFields(*blocks, uFieldId, uTmpFieldId); }, "Swap");

   if (vtkWriteFrequency > 0)
   {
      auto vtkOutput = vtk::createVTKOutput_BlockData(*blocks, "vtkGPU3d", vtkWriteFrequency, 0, false, "vtk_out_gpu_3d",
                                                      "simulation_step", false, true, true, false, 0);

      auto tempWriter = make_shared< field::VTKWriter< ScalarField > >(uFieldCpuId, "temperature");
      vtkOutput->addCellDataWriter(tempWriter);

      auto alphaWriter = make_shared< field::VTKWriter< ScalarField > >(alphaFieldCpuId, "thermal_diffusivity");
      vtkOutput->addCellDataWriter(alphaWriter);

      vtkOutput->addBeforeFunction([&]() {
          // Copy GPU data to CPU for VTK output
          gpu::fieldCpy<ScalarField, GPUScalarField>(blocks, uFieldCpuId, uFieldId);
          gpu::fieldCpy<ScalarField, GPUScalarField>(blocks, alphaFieldCpuId, alphaFieldId);
      });

      timeloop.addFuncAfterTimeStep(vtk::writeFiles(vtkOutput), "VTK Output GPU 3D");
   }

   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   ///                                               BENCHMARK                                                    ///
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

   const uint_t warmupSteps     = uint_c(5);
   const uint_t outerIterations = uint_c(1);

   for (uint_t i = 0; i < warmupSteps; ++i)
      timeloop.singleStep();

   auto remainingTimeLoggerFrequency = real_c(-1.0); // in seconds
   if (remainingTimeLoggerFrequency > 0)
   {
      auto logger = timing::RemainingTimeLogger(timeloop.getNrOfTimeSteps() * outerIterations,
                                                remainingTimeLoggerFrequency);
      timeloop.addFuncAfterTimeStep(logger, "remaining time logger");
   }

   for (uint_t outerIteration = 0; outerIteration < outerIterations; ++outerIteration)
   {

      WALBERLA_GPU_CHECK(gpuPeekAtLastError())
      timeloop.setCurrentTimeStepToZero();
      WcTimingPool timeloopTiming;
      WcTimer simTimer;

      WALBERLA_MPI_WORLD_BARRIER()
      WALBERLA_GPU_CHECK( gpuDeviceSynchronize() )
      WALBERLA_GPU_CHECK( gpuPeekAtLastError() )

      WALBERLA_LOG_INFO_ON_ROOT("Starting simulation with " << timeSteps << " time steps")
      simTimer.start();
      timeloop.run(timeloopTiming);
      WALBERLA_GPU_CHECK( gpuDeviceSynchronize() )
      simTimer.end();
      WALBERLA_LOG_INFO_ON_ROOT("Simulation finished")

      double simTime = simTimer.max();
      WALBERLA_MPI_SECTION() { walberla::mpi::reduceInplace(simTime, walberla::mpi::MAX); }

      const auto reducedTimeloopTiming = timeloopTiming.getReduced();
      WALBERLA_LOG_RESULT_ON_ROOT("Time loop timing:\n" << *reducedTimeloopTiming)

      uint_t mlups = timeSteps * xCells * yCells * zCells / (uint_c(simTime * 1000000.0));
      WALBERLA_LOG_RESULT_ON_ROOT("mlups:\t" << mlups)
   }

   return EXIT_SUCCESS;
}
}

int main(int argc, char** argv) { walberla::main(argc, argv); }
