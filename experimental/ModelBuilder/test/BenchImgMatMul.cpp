// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <string>

#include "experimental/ModelBuilder/ModelBuilder.h"
#include "experimental/ModelBuilder/ModelRunner.h"
#include "experimental/ModelBuilder/VulkanWrapperPass.h"
#include "iree/base/initializer.h"
#include "iree/compiler/Conversion/CodegenUtils/MatmulCodegenStrategy.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/MemorySpace.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Passes.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Utils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "mlir/Conversion/GPUToVulkan/ConvertGPUToVulkanPass.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRVPass.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/Linalg/EDSC/Intrinsics.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SPIRV/Passes.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/TargetAndABI.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/RunnerUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;                    // NOLINT
using namespace mlir::edsc;              // NOLINT
using namespace mlir::edsc::intrinsics;  // NOLINT

static llvm::cl::opt<std::string> vulkanWrapper(
    "vulkan-wrapper", llvm::cl::desc("Vulkan wrapper library"),
    llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static llvm::cl::opt<bool> correctness(
    "correctness",
    llvm::cl::desc(
        "Compare the result to value calculated on CPU. We will use a smaller "
        "matrix multiply in this case to avoid long runtime."),
    llvm::cl::init(false));

static llvm::cl::opt<bool> useWorkgroupMemory(
    "use-workgroup-memory", llvm::cl::desc("Enable use of workgroup memory"),
    llvm::cl::value_desc("boolean"), llvm::cl::init(false));

static llvm::cl::opt<bool> enableLICM(
    "enable-licm",
    llvm::cl::desc("Enable loop invariant hoisting optimizations"),
    llvm::cl::value_desc("boolean"), llvm::cl::init(false));

static void addLoweringPasses(mlir::PassManager &pm,
                              llvm::ArrayRef<int64_t> numWorkgroups,
                              llvm::ArrayRef<Type> args) {
  pm.addPass(mlir::iree_compiler::createVectorToGPUPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createLegalizeStdOpsForSPIRVLoweringPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::iree_compiler::createConvertToSPIRVPass());

  auto &spirvModulePM = pm.nest<mlir::spirv::ModuleOp>();
  spirvModulePM.addPass(mlir::createSetSpirvABIPass());
  spirvModulePM.addPass(mlir::spirv::createLowerABIAttributesPass());
  spirvModulePM.addPass(mlir::createCanonicalizerPass());
  spirvModulePM.addPass(mlir::createCSEPass());
  spirvModulePM.addPass(
      mlir::spirv::createUpdateVersionCapabilityExtensionPass());

  pm.addPass(mlir::createAddVulkanLaunchWrapperPass(numWorkgroups, args));
  mlir::LowerToLLVMOptions llvmOptions = {
      /*useBarePtrCallConv=*/false,
      /*emitCWrappers=*/true,
      /*indexBitwidth=*/mlir::kDeriveIndexBitwidthFromDataLayout};
  pm.addPass(createLowerToLLVMPass(llvmOptions));
  pm.addPass(mlir::createConvertVulkanLaunchFuncToVulkanCallsPass());
}

static void insertBarrier(OpBuilder &b, Location loc) {
  b.create<spirv::ControlBarrierOp>(loc, spirv::Scope::Workgroup,
                                    spirv::Scope::Workgroup,
                                    spirv::MemorySemantics::AcquireRelease);
}

template <typename IdOp, typename NProcsOp>
static SmallVector<linalg::ProcInfo, 2> getGpuProcIds(
    OpBuilder &b, Location loc, ArrayRef<SubViewOp::Range> parallelLoopRanges) {
  if (parallelLoopRanges.size() != 2)
    llvm_unreachable("expected two parallel loops for matmul operation");
  Type indexType = b.getIndexType();
  SmallVector<linalg::ProcInfo, 2> procInfo(2);
  procInfo[0] = {b.create<IdOp>(loc, indexType, b.getStringAttr("y")),
                 b.create<NProcsOp>(loc, indexType, b.getStringAttr("y"))};
  procInfo[1] = {b.create<IdOp>(loc, indexType, b.getStringAttr("x")),
                 b.create<NProcsOp>(loc, indexType, b.getStringAttr("x"))};
  return procInfo;
}

template <typename IdOp, int N>
static SmallVector<linalg::ProcInfo, 2> getGpuProcIds(
    OpBuilder &b, Location loc, ArrayRef<SubViewOp::Range> parallelLoopRanges) {
  if (parallelLoopRanges.size() != 2)
    llvm_unreachable("expected two parallel loops for matmul operation");
  Type indexType = b.getIndexType();
  SmallVector<linalg::ProcInfo, 2> procInfo(2);
  procInfo[0] = {b.create<IdOp>(loc, indexType, b.getStringAttr("x")),
                 b.create<ConstantIndexOp>(loc, N)};
  procInfo[1] = {b.create<ConstantIndexOp>(loc, 0),
                 b.create<ConstantIndexOp>(loc, N)};
  return procInfo;
}

struct MatMulF32 {
  using Type = float;
  static mlir::Type getMLIRType(MLIRContext &ctx) {
    return FloatType::getF32(&ctx);
  }
};

/// Functions to initialize matrix based on the type.
template <typename T>
static T getMatA(unsigned idx) {
  if (std::is_same<T, float>::value)
    return ((float)(idx % 5) - 1.0f) / 2.0f;
  else
    return (3 * idx + 1) % 117;
}

template <typename T>
static T getMatB(unsigned idx) {
  if (std::is_same<T, float>::value)
    return ((float)(idx % 7) - 1.0f) / 2.0f;
  else
    return idx % 127;
}

template <typename T>
static bool EqualOrClose(T a, T b) {
  if (std::is_same<T, float>::value)
    return fabs((float)a - (float)b) < 0.1f;
  return a == b;
}

constexpr int TileM = 32, TileN = 32, TileK = 4;
//constexpr int TileM = 4, TileN = 4, TileK = 2;

template <typename SrcT, typename DstT>
static void matMul(int m, int n, int k, int tileM, int tileN, int tileK,
                   const std::array<int64_t, 3> &nativeSize, bool correctness) {
  const int warpSize = TileM;
  const int resRows = m;
  const int resColumns = n;
  const int reductionSize = k;
  StringLiteral funcName = "kernel_matmul";
  ModelBuilder modelBuilder;
  MLIRContext &ctx = *modelBuilder.getContext();
  auto typeA = modelBuilder.getMemRefType({resRows, reductionSize},
                                          SrcT::getMLIRType(ctx));
  auto typeB = modelBuilder.getMemRefType({reductionSize, resColumns},
                                          SrcT::getMLIRType(ctx));
  auto typeC =
      modelBuilder.getMemRefType({resRows, resColumns}, DstT::getMLIRType(ctx));
  // 1. Build the kernel.
  {
    modelBuilder.addGPUAttr();
    FuncOp kernelFunc = modelBuilder.makeFunction(
        funcName, {}, {typeA, typeB, typeC}, MLIRFuncOpConfig());
    int workgroupSize;
    if (useWorkgroupMemory)
      workgroupSize = warpSize;
    else
      workgroupSize = warpSize;
    // Right now we map one workgroup to one warp.
    kernelFunc.setAttr(
        spirv::getEntryPointABIAttrName(),
        spirv::getEntryPointABIAttr({workgroupSize, 1, 1}, &ctx));
    OpBuilder b(&kernelFunc.getBody());
    ScopedContext scope(b, kernelFunc.getLoc());

    auto A = kernelFunc.getArgument(0);
    auto B = kernelFunc.getArgument(1);
    auto C = kernelFunc.getArgument(2);

    linalg_matmul(TypeRange{}, ValueRange{A, B, C});
    std_ret();
  }

  // 2. Compile the function, pass in runtime support library to the execution
  // engine for vector.print.
  ModelRunner runner(modelBuilder.getModuleRef(),
                     ModelRunner::Target::GPUTarget);
  CompilationOptions options;
  options.loweringPasses = [&](mlir::PassManager &pm) {
    MatmulCodegenStrategy strategy;

    linalg::LinalgLoopDistributionOptions WGDistribute;
    WGDistribute.distributionMethod = {
        linalg::DistributionMethod::CyclicNumProcsEqNumIters,
        linalg::DistributionMethod::CyclicNumProcsEqNumIters};
    WGDistribute.procInfo = getGpuProcIds<gpu::BlockIdOp, gpu::GridDimOp>;

    linalg::LinalgLoopDistributionOptions WIDistribute;
    WIDistribute.distributionMethod = {
      linalg::DistributionMethod::CyclicNumProcsEqNumIters,
      linalg::DistributionMethod::CyclicNumProcsEqNumIters};
    WIDistribute.procInfo = getGpuProcIds<gpu::ThreadIdOp, TileN>;

    strategy
        .tile<linalg::MatmulOp>(
            linalg::LinalgTilingOptions()
                .setLoopType(linalg::LinalgTilingLoopType::ParallelLoops)
                .setTileSizes({tileM, tileN, tileK})
                .setInterchange({1, 0, 2})
                .setDistributionOptions(WGDistribute))
        .setHoistInvariantCode(enableLICM);
    //if (useWorkgroupMemory) {
      strategy
          .promote<linalg::MatmulOp>(
              linalg::LinalgPromotionOptions()
                  .setAllocationDeallocationFns(
                      mlir::iree_compiler::allocateWorkgroupMemory,
                      mlir::iree_compiler::deallocateWorkgroupMemory)
                  .setCopyInOutFns(mlir::iree_compiler::copyToWorkgroupMemory,
                                   mlir::iree_compiler::copyToWorkgroupMemory)
                  // 0: A, 1: B, 2: C
                  // 2 = 0 x 1
                  .setOperandsToPromote({1})
                  //.setOperandsToPromote({0, 1}) // nv. 540 ms
                  //.setOperandsToPromote({0, 1, 2}) // nv. 154 ms
                  .setUseFullTileBuffers({false, false}))
          ;
    //}
    strategy
        .tile<linalg::MatmulOp>(
            linalg::LinalgTilingOptions()
                .setLoopType(linalg::LinalgTilingLoopType::ParallelLoops)
                .setTileSizes({1, tileN, 1})
                //.setTileSizes({1, tileN, tileK})
                //.setTileSizes({tileM, tileN, tileK})
                .setDistributionOptions(WIDistribute));
    strategy
        .tile<linalg::MatmulOp>(
            linalg::LinalgTilingOptions()
                .setLoopType(linalg::LinalgTilingLoopType::Loops)
                .setTileSizes({1, 8, 1})
                );

    strategy.vectorize<linalg::MatmulOp>().unrollVector<vector::ContractionOp>(
        nativeSize);
    modelBuilder.getModuleRef()->walk(
        [&](FuncOp fn) { strategy.transform(fn); });
    addLoweringPasses(pm, {resRows / tileM, resColumns / tileN, 1},
                      {typeA, typeB, typeC});
  };
  runner.compile(options, {vulkanWrapper});

  // 3. Allocate data within data structures that interoperate with the MLIR ABI
  // conventions used by codegen.
  auto initA = [](unsigned idx, typename SrcT::Type *ptr) {
    ptr[idx] = getMatA<typename SrcT::Type>(idx);
  };
  auto initB = [](unsigned idx, typename SrcT::Type *ptr) {
    ptr[idx] = getMatB<typename SrcT::Type>(idx);
  };
  auto zeroInit = [](unsigned idx, typename DstT::Type *ptr) { ptr[idx] = 0; };
  auto A = makeInitializedStridedMemRefDescriptor<typename SrcT::Type, 2>(
      {resRows, reductionSize}, initA);
  auto B = makeInitializedStridedMemRefDescriptor<typename SrcT::Type, 2>(
      {reductionSize, resColumns}, initB);
  auto C = makeInitializedStridedMemRefDescriptor<typename DstT::Type, 2>(
      {resRows, resColumns}, zeroInit);
  auto CPURes = makeInitializedStridedMemRefDescriptor<typename DstT::Type, 2>(
      {resRows, resColumns}, zeroInit);

  // Is checking corretness compare to the value computed on CPU.
  if (correctness) {
    for (int i = 0; i < resRows; i++) {
      for (int j = 0; j < resColumns; j++) {
        typename DstT::Type acc = (*C)[i][j];
        for (int k = 0; k < reductionSize; k++) {
          typename DstT::Type a = (*A)[i][k];
          typename DstT::Type b = (*B)[k][j];
          acc += a * b;
        }
        (*CPURes)[i][j] = acc;
      }
    }
  }

  // 4. Call the funcOp named `funcName`.
  auto err = runner.invoke(std::string(funcName) + "_wrapper", A, B, C);
  if (err) llvm_unreachable("Error running function.");

  int errcnt = 0;
  if (correctness) {
    bool correct = true;
    for (int i = 0; i < resRows; i++) {
      for (int j = 0; j < resColumns; j++) {
        if (!EqualOrClose((*CPURes)[i][j], (*C)[i][j]) && ++errcnt < 15) {
          correct = false;
          llvm::errs() << "mismatch at index(" << i << ", " << j
                       << ") was expecting " << (*CPURes)[i][j] << " but got "
                       << (*C)[i][j] << "\n";
        }
      }
    }
    if (correct) printf("pass\n");
    else {
      llvm::errs() << "mismatch count = " << errcnt << " ("
                   << (float)errcnt / (resRows*resColumns) << "%)\n";
    }
  }
}

static void matMul(int m, int n, int k, int tileM, int tileN, int tileK,
                   bool correctness) {
  std::array<int64_t, 3> nativeMatSize;
  nativeMatSize = {1, 1, 1};
  return matMul<MatMulF32, MatMulF32>(m, n, k, tileM, tileN, tileK,
                                      nativeMatSize, correctness);
}

int main(int argc, char **argv) {
  ModelBuilder::registerAllDialects();
  iree::Initializer::RunInitializers();
  // Allow LLVM setup through command line and parse the
  // test specific option for a runtime support library.
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "BenchMatMulVectorGPU\n");
  int m = 4096/4;
  int n = 4096/4;
  int k = 4096/4;
  if (correctness) {
    // m = 256;
    // n = 256;
    // k = 256;
  }
  printf("Matrix size: %ix%ix%i\n", m, n, k);
  matMul(m, n, k, TileM, TileN, TileK, correctness);
  return 0;
}