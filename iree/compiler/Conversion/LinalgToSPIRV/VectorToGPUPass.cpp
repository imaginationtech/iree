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

//===---- VectorToGPUPass.cpp - Pass for the final SPIR-V conversion
//-------===//
//
// This file implement a pass to convert vector dialect operations to GPU
// operations distributed across a subgroup.
//
//===----------------------------------------------------------------------===//
#include <memory>

#include "iree/compiler/Conversion/CodegenUtils/FunctionUtils.h"
#include "iree/compiler/Conversion/CodegenUtils/MatmulCodegenStrategy.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/CooperativeMatrixAnalysis.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/MarkerUtils.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOpUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SPIRV/TargetAndABI.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace {
// TODO(thomasraoux): Fetch this value from device properties.
static const int subgroupSize = 32;

struct ConvertVectorToGPUPass
    : public PassWrapper<ConvertVectorToGPUPass, OperationPass<FuncOp>> {
  void runOnOperation() override;

 private:
  void tileAndVectorizeLinalgCopy(FuncOp funcOp, MLIRContext *context);
  void imgHackConvertForOpResultToF32(FuncOp funcOp, MLIRContext *context);
};

// Common class for all vector to GPU patterns.
template <typename OpTy>
class VectorToGPUPattern : public OpConversionPattern<OpTy> {
 public:
  VectorToGPUPattern<OpTy>(
      MLIRContext *context,
      const CooperativeMatrixAnalysis &cooperativeMatrixAnalysis)
      : OpConversionPattern<OpTy>::OpConversionPattern(context),
        cooperativeMatrixAnalysis(cooperativeMatrixAnalysis) {}

 protected:
  const CooperativeMatrixAnalysis &cooperativeMatrixAnalysis;
};

/// Converts unary and binary standard operations using new type.
template <typename StdOp>
class UnaryAndBinaryOpPattern final : public VectorToGPUPattern<StdOp> {
 public:
  using VectorToGPUPattern<StdOp>::VectorToGPUPattern;

  LogicalResult matchAndRewrite(
      StdOp operation, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (VectorToGPUPattern<StdOp>::cooperativeMatrixAnalysis
            .usesCooperativeMatrixType(operation))
      return failure();
    Value newOp =
        rewriter.create<StdOp>(operation.getLoc(), ValueRange(operands));
    rewriter.replaceOp(operation, ValueRange(newOp));
    return success();
  }
};

class VectorTransferReadConversion
    : public VectorToGPUPattern<vector::TransferReadOp> {
 public:
  using VectorToGPUPattern<vector::TransferReadOp>::VectorToGPUPattern;

  LogicalResult matchAndRewrite(
      vector::TransferReadOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (cooperativeMatrixAnalysis.usesCooperativeMatrixType(op))
      return failure();
    // Only support identity map for now.
    if (!op.permutation_map().isIdentity()) return failure();
    if (op.getVectorType().getNumElements() != subgroupSize) return failure();
    // Only works for the case where one workgroups has only one subgroup.
    auto wgSize = spirv::lookupLocalWorkGroupSize(op);
    if (wgSize.getValue<int32_t>(0) != subgroupSize ||
        wgSize.getValue<int32_t>(1) != 1 || wgSize.getValue<int32_t>(2) != 1)
      return failure();
    auto loc = op.getLoc();
    SmallVector<Value, 4> indices(op.indices());
    // Use threadId.x as the subgroupInvocationId.
    // TODO(thomasraoux): Replace it once subgroup Ids are working.
    auto threadIndex = rewriter.create<gpu::ThreadIdOp>(
        loc, rewriter.getIndexType(), rewriter.getStringAttr("x"));
    Value index = rewriter.create<AddIOp>(loc, threadIndex, indices.back());
    indices.back() = index;
    Value newOp = rewriter.create<LoadOp>(loc, op.memref(), indices);
    rewriter.replaceOp(op, ValueRange(newOp));
    return success();
  }
};

class VectorTransferWriteConversion
    : public VectorToGPUPattern<vector::TransferWriteOp> {
 public:
  using VectorToGPUPattern<vector::TransferWriteOp>::VectorToGPUPattern;

  LogicalResult matchAndRewrite(
      vector::TransferWriteOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (cooperativeMatrixAnalysis.usesCooperativeMatrixType(op))
      return failure();
    if (!op.permutation_map().isIdentity()) return failure();
    if (op.getVectorType().getNumElements() != subgroupSize) return failure();
    auto loc = op.getLoc();
    SmallVector<Value, 4> indices(op.indices());
    auto ThreadIndex = rewriter.create<gpu::ThreadIdOp>(
        loc, rewriter.getIndexType(), rewriter.getStringAttr("x"));
    Value index = rewriter.create<AddIOp>(loc, ThreadIndex, indices.back());
    indices.back() = index;
    rewriter.create<StoreOp>(op.getLoc(), operands[0], operands[1], indices);
    rewriter.eraseOp(op);
    return success();
  }
};


// Convert degenerated vectors to scalar.
class VectorTransferReadConversionScalar
    : public VectorToGPUPattern<vector::TransferReadOp> {
 public:
  using VectorToGPUPattern<vector::TransferReadOp>::VectorToGPUPattern;

  LogicalResult matchAndRewrite(
      vector::TransferReadOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (op.getVectorType().getNumElements() != 1)
      return failure();
    auto loc = op.getLoc();
    Value newOp = rewriter.create<LoadOp>(loc, op.memref(), op.indices());
    rewriter.replaceOp(op, newOp);

    scf::ForOp loop;
    unsigned argNo;
    for (auto &use : op.vector().getUses()) {
      loop = dyn_cast<scf::ForOp>(use.getOwner());
      if (loop) {
        argNo = use.getOperandNumber() - 3;
        break;
      }
    }

    if (loop) {
      //llvm::errs() << "==== dump region ===\n";
      //loop.region();
      llvm::errs() << "==== modify block argument [" << argNo + 1 << "] type from ";
      loop.region().front().getArgument(argNo + 1).getType().dump();
      llvm::errs() << " to ";
      newOp.getType().dump();
      llvm::errs() << "\n";
      //LoadOp newLoadOp = dyn_cast<LoadOp>(newOp.getDefiningOp());
      // int i = 0;
      // for (Value arg : loop.initArgs()) {
      //   if (i == argNo) {
      //     arg.replaceAllUsesWith(newLoadOp.result());
      //     break;
      //   }
      //   ++i;
      // }
      //BlockArgument newArg = loop.region().front().insertArgument(argNo + 1, newOp.getType());

      // llvm::errs() << "its users:\n";
      // for (auto &use : loop.region().front().getArgument(argNo + 1).getUses()) {
      //   llvm::errs() << "  * ";
      //   use.getOwner()->dump();
      // }

      // directly set type:
      // loop.region().front().getArgument(argNo + 1).setType(newOp.getType());

      // replace with new type:
      //loop.region().front().addArgument(newOp.getType());
      BlockArgument newArg = loop.getBody()->insertArgument(argNo + 1, newOp.getType());
        //loop.getBody()->addArgument(newOp.getType());

      //loop.region().front().getArgument(argNo + 1).replaceAllUsesWith(newArg);
      loop.getBody()->getArgument(argNo + 2).replaceAllUsesWith(newArg);
      loop.getBody()->eraseArgument(argNo + 2);
      llvm::errs() << "number args = " << loop.getBody()->getNumArguments() << "\n";

      llvm::errs() << "newArg users:\n";
      for (auto &use : newArg.getUses()) {
        llvm::errs() << "  * ";
        use.getOwner()->dump();
      }

      //loop.region().front().getArgument(argNo + 2).replaceAllUsesWith(newArg);
      //loop.region().front().dump();
      //loop.initArgsMutable();
      //loop.region().front().eraseArgument(argNo + 2);
      //auto newArgs = loop.initArgs();
      //newArgs[argNo].
      //newArgs[argNo] = newLoadOp.result();
      //loop.initArgsMutable().assign(newArgs);
      //loop.region().front().dump();
      //loop.initArgsMutable().clear();
      //loop.region().front().getArgument(argNo).setType(newOp.getType());

      // debug dump
      // llvm::errs() << "\nblock dumps:\n\n";
      // loop.getBody()->dump();

      // llvm::errs() << "\nfunc dumps:\n\n";
      // loop.getParentRegion()->front().dump();
    }

    return success();
  }
};

// user -> new value
std::map<Operation *, Value> loopResultUserMap;

class VectorTransferWriteConversionScalar
    : public VectorToGPUPattern<vector::TransferWriteOp> {
 public:
  using VectorToGPUPattern<vector::TransferWriteOp>::VectorToGPUPattern;

  LogicalResult matchAndRewrite(
      vector::TransferWriteOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {

    if (op.getVectorType().getNumElements() != 1)
      return failure();
    StoreOp::Adaptor adaptor(operands);
    auto loc = op.getLoc();

    llvm::errs() << "repalce write op: ";
    op.dump();
    llvm::errs() << "\n";
    Value operand0 = operands[0];
    if (loopResultUserMap.find(op.getOperation()) != loopResultUserMap.end()) {
      loopResultUserMap[op.getOperation()].dump();
      operand0 = loopResultUserMap[op.getOperation()];
    }

    rewriter.create<StoreOp>(loc, operand0, //operands[0],
                             operands[1],
                             op.indices());
    rewriter.eraseOp(op);
    return success();
  }
};

class VectorContractScalar
    : public VectorToGPUPattern<vector::ContractionOp> {
 public:
  using VectorToGPUPattern<vector::ContractionOp>::VectorToGPUPattern;

  LogicalResult matchAndRewrite(
      vector::ContractionOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {

    llvm::errs() << " ====== op.getAccType().dump() ==== \n";
    //op.getAccType().dump();
    //if (op.getAccType().cast<VectorType>().getNumElements() != 1)
    //  return failure();
    vector::ContractionOp::Adaptor adaptor(operands);
    auto loc = op.getLoc();

    Value newOp = rewriter.create<MulFOp>(loc, adaptor.lhs(), adaptor.rhs());
    newOp = rewriter.create<AddFOp>(loc, newOp, adaptor.acc());
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

class VectorToGPUConversionTarget : public ConversionTarget {
 public:
  using ConversionTarget::ConversionTarget;

 protected:
  // Standard operation are legal if they operate on scalars. We need to
  // legalize operations on vectors.
  bool isDynamicallyLegal(Operation *op) const override {
    auto isVectorType = [](Type t) { return t.isa<VectorType>(); };
    if (llvm::any_of(op->getResultTypes(), isVectorType) ||
        llvm::any_of(op->getOperandTypes(), isVectorType))
      return false;
    return true;
  }
};

void ConvertVectorToGPUPass::tileAndVectorizeLinalgCopy(FuncOp funcOp,
                                                        MLIRContext *context) {
  // 1. Tile linalg and distribute it on invocations.
  std::unique_ptr<ConversionTarget> target =
      std::make_unique<ConversionTarget>(*context);
  target->addDynamicallyLegalOp<linalg::CopyOp>([&](linalg::CopyOp copy) {
    return !(hasMarker(copy, getCopyToWorkgroupMemoryMarker()));
  });
  target->markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  OwningRewritePatternList tileAndDistributePattern;
  populateLinalgTileAndDistributePatterns(context, tileAndDistributePattern);
  if (failed(
          applyPartialConversion(funcOp, *target, tileAndDistributePattern))) {
    return signalPassFailure();
  }

  // 2. Canonicalize the IR generated by tiling.
  OwningRewritePatternList canonicalizePatterns =
      linalg::getLinalgTilingCanonicalizationPatterns(context);
  canonicalizePatterns.insert<AffineMinCanonicalizationPattern,
                              linalg::AffineMinSCFCanonicalizationPattern>(
      context);
  applyPatternsAndFoldGreedily(funcOp, canonicalizePatterns);

  // 3. Vectorize the tiled linalg to be able to map it to load/store vector.
  OwningRewritePatternList vectorizationPatterns;
  vectorizationPatterns
      .insert<linalg::LinalgVectorizationPattern<linalg::CopyOp>>(
          context, linalg::LinalgMarker(
                       Identifier::get(getVectorizeMarker(), context), {}));
  applyPatternsAndFoldGreedily(funcOp, vectorizationPatterns);
}

static Operation *cloneWithNewResultTypes(Operation *op, TypeRange newResultTypes) {
  OperationState state(op->getLoc(), op->getName());
  state.addOperands(op->getOperands());
  state.addTypes(newResultTypes);
  state.addSuccessors(op->getSuccessors());
  state.addAttributes(op->getAttrs());
  for (unsigned i = 0, e = op->getNumRegions(); i < e; ++i) {
    state.addRegion();
  }
  Operation *newOp = Operation::create(state);
  for (unsigned i = 0, e = op->getNumRegions(); i < e; ++i) {
    newOp->getRegion(i).takeBody(op->getRegion(i));
  }
  return newOp;
}

void ConvertVectorToGPUPass::imgHackConvertForOpResultToF32(FuncOp funcOp, MLIRContext *ctx) {
  bool changed = true;
  while (changed) {
    changed = false;
    funcOp.walk([&](Operation *op) {
      auto loop = dyn_cast<scf::ForOp>(op);

      if (!loop || loop.getNumResults() == 0 || loop.getResult(0).getType() == FloatType::getF32(ctx))
        return WalkResult::advance();

      llvm::SmallVector<Type, 4> newResultTypes;
      for (unsigned i = 0; i < loop.getNumResults(); ++i)
        newResultTypes.push_back(FloatType::getF32(ctx));

      OpBuilder builder(loop);
      auto newloop = cloneWithNewResultTypes(loop, newResultTypes);
      builder.insert(newloop);

  auto &loopBody = *loop.getBody();
  auto &newLoopBody = *dyn_cast<scf::ForOp>(newloop).getBody();
  // auto yield = cast<scf::YieldOp>(loopBody.getTerminator());
  // auto yieldOperands = llvm::to_vector<4>(yield.getOperands());
  // auto newYield = cast<scf::YieldOp>(newLoopBody.getTerminator());

      // for (unsigned i = 0; i < loop.getNumResults(); ++i)
      //   loop.getResult(i).replaceAllUsesWith(newloop->getResult(i));
      unsigned i = 0;
      for (OpResult result : loop.getResults()) {
        for (auto &use : result.getUses()) {
          llvm::errs() << "=== new result type ==\n";
          //newloop->getResult(i).dump();
          //result.dump();
          loopResultUserMap[use.getOwner()] = newloop->getResult(i);
          //use.set(newloop->getResult(i));
        }
        ++i;
      }
      // llvm::errs() << "=== loopBody.getTerminator()->dump(); ==\n";
      // loopBody.getTerminator()->dump();
      llvm::errs() << "=== newLoopBody.getTerminator()->dump(); ==\n";
      newLoopBody.getTerminator()->dump();
      loop.erase();
      llvm::errs() << "=== newLoopBody.getTerminator()->dump(); ==\n";
      newLoopBody.getTerminator()->dump();

      changed = true;
      return WalkResult::interrupt();
    });
  }
  llvm::errs() << "=== rewrite output type ==\n";
  funcOp.dump();
}

void ConvertVectorToGPUPass::runOnOperation() {
  MLIRContext *context = &getContext();
  FuncOp funcOp = getOperation();
  tileAndVectorizeLinalgCopy(funcOp, context);
  imgHackConvertForOpResultToF32(funcOp, context);

  auto &cooperativeMatrixAnalysis = getAnalysis<CooperativeMatrixAnalysis>();
  OwningRewritePatternList patterns;
  // patterns.insert<UnaryAndBinaryOpPattern<AddFOp>, VectorTransferReadConversion,
  //                 VectorTransferWriteConversion>(context,
  //                                                cooperativeMatrixAnalysis);
  patterns
      .insert<UnaryAndBinaryOpPattern<AddFOp>, VectorTransferReadConversion,
              VectorTransferWriteConversion, VectorTransferReadConversionScalar,
              VectorTransferWriteConversionScalar, VectorContractScalar>(
          context, cooperativeMatrixAnalysis);

  std::unique_ptr<VectorToGPUConversionTarget> target =
      std::make_unique<VectorToGPUConversionTarget>(*context);
  target->addDynamicallyLegalDialect<StandardOpsDialect>();
  target->addIllegalOp<scf::ParallelOp>();
  target->addLegalOp<scf::YieldOp>();
  target->addLegalOp<scf::ForOp>();
  target->addLegalDialect<gpu::GPUDialect>();
  if (failed(applyPartialConversion(funcOp, *target, patterns)))
    return signalPassFailure();
}
}  // namespace

//===----------------------------------------------------------------------===//
// Pass entry point and registration
//===----------------------------------------------------------------------===//
std::unique_ptr<OperationPass<FuncOp>> createVectorToGPUPass() {
  return std::make_unique<ConvertVectorToGPUPass>();
}

static PassRegistration<ConvertVectorToGPUPass> pass(
    "iree-codegen-vector-to-gpu",
    "Convert vector dialect to gpu subgroup level GPU instructions");
}  // namespace iree_compiler
}  // namespace mlir
