#include "ReduceScanCommon.h"
#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::delinearize;
using ::mlir::LLVM::linearize;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getThreadOrder;
using ::mlir::triton::gpu::getTotalElemsPerThread;

static Value perturb_value(Value val, ConversionPatternRewriter &rewriter,
                           Location loc, Type floatTy) {
  if (!mlir::isa<FloatType>(floatTy))
    return val;

  auto getIntTy = [&](int bits) { return rewriter.getIntegerType(bits); };

  auto makeIntConst = [&](int bits, int64_t v) -> Value {
    return rewriter.create<LLVM::ConstantOp>(
        loc, getIntTy(bits), rewriter.getIntegerAttr(getIntTy(bits), v));
  };
  
  auto makeFloatConst = [&](double v) -> Value {
    int bits = floatTy.getIntOrFloatBitWidth();
    if (bits == 8) {  
      int64_t bitPattern;
      if (v == 0.0) {
        bitPattern = 0;
      } else {
        bitPattern = static_cast<int64_t>(v); 
      }
      Value intConst = makeIntConst(8, bitPattern);
      return rewriter.create<LLVM::BitcastOp>(loc, floatTy, intConst);
    } else {
      return rewriter.create<LLVM::ConstantOp>(
          loc, floatTy, rewriter.getFloatAttr(floatTy, v));
    }
  };

  int bits = floatTy.getIntOrFloatBitWidth();
  if (bits != 8 && bits != 16 && bits != 32 && bits != 64)
    return val;

  auto intTy = getIntTy(bits);
  Value one = makeIntConst(bits, 1); 
  Value zeroF = makeFloatConst(0.0);

  Value isFinite = rewriter.create<LLVM::FCmpOp>(
      loc, LLVM::FCmpPredicate::ord, val, val);
  
  Value isZero = rewriter.create<LLVM::FCmpOp>(
      loc, LLVM::FCmpPredicate::oeq, val, zeroF);
  
  Value smallestNegBits;
  if (bits == 8) {
    smallestNegBits = makeIntConst(8, (1 << 7) | 1);  // 0x81 (E5M2/E4M3 min neg)
  } else {
    smallestNegBits = makeIntConst(bits, (1LL << (bits - 1)) | 1);
  }
  Value smallestNegF = rewriter.create<LLVM::BitcastOp>(loc, floatTy, smallestNegBits);

  Value isNeg = rewriter.create<LLVM::FCmpOp>(loc, LLVM::FCmpPredicate::olt, val, zeroF);
  
  Value bitsVal = rewriter.create<LLVM::BitcastOp>(loc, intTy, val);
  
  Value bitsPlus = rewriter.create<LLVM::AddOp>(loc, intTy, bitsVal, one);
  Value bitsMinus = rewriter.create<LLVM::SubOp>(loc, intTy, bitsVal, one);
  Value perturbedBits = rewriter.create<LLVM::SelectOp>(loc, isNeg, bitsPlus, bitsMinus);
  
  Value perturbedF = rewriter.create<LLVM::BitcastOp>(loc, floatTy, perturbedBits);
  
  Value result = rewriter.create<LLVM::SelectOp>(loc, isZero, smallestNegF, perturbedF);
  result = rewriter.create<LLVM::SelectOp>(loc, isFinite, result, val);

  return result;
}

namespace {
struct ReduceOpConversion
    : public ConvertTritonGPUReduceScanToLLVMPattern<triton::ReduceOp> {
public:
  ReduceOpConversion(LLVMTypeConverter &typeConverter,
                     const TargetInfoBase &targetInfo, PatternBenefit benefit)
      : ConvertTritonGPUReduceScanToLLVMPattern<triton::ReduceOp>(typeConverter,
                                                                  benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ReduceOpHelper helper(op);
    assert(helper.isSupportedLayout() &&
           "Unexpected srcLayout in ReduceOpConversion");
    Location loc = op->getLoc();

    auto srcValues = unpackInputs(loc, op, adaptor, rewriter);
    std::map<SmallVector<unsigned>, SmallVector<Value>> accs;
    std::map<SmallVector<unsigned>, SmallVector<Value>> indices;
    
    reduceWithinThreads(helper, srcValues, accs, indices, rewriter);

    reduceWithinWarps(helper, accs, rewriter);

    if (helper.isWarpSynchronous()) {
      packResults(helper, accs, rewriter);
      return success();
    }

    auto smemShape = helper.getScratchRepShape();
    SmallVector<Value> smemBases =
        getSmemBases(op, product<unsigned>(smemShape), rewriter, targetInfo);

    storeWarpReduceToSharedMemory(helper, accs, indices, smemBases, rewriter);

    sync(rewriter, loc, op);

    accumulatePartialReductions(helper, smemBases, rewriter);

    sync(rewriter, loc, op);

    loadReductionAndPackResult(helper, smemShape, smemBases, rewriter);

    return success();
  }

private:
  const TargetInfoBase &targetInfo;

  void accumulate(Location loc, ConversionPatternRewriter &rewriter,
                  Region &combineOp, SmallVector<Value> &acc, ValueRange cur,
                  Value pred = {}) const {

    bool isFloatAdd = false;
    Block &block = combineOp.front();
    if (block.getOperations().size() == 2 && 
        isa<arith::AddFOp>(block.front())) {
        isFloatAdd = true;
    }

    if (!isFloatAdd) {
        auto results = applyCombineOp(loc, rewriter, combineOp, acc, cur, pred);
        if (acc.size() < results.size()) {
            acc.resize(results.size());
        }
        for (unsigned i = 0; i < acc.size(); ++i) {
            acc[i] = results[i];
        }
        return;
    }

    if (acc.empty()) {
      acc.push_back(cur[0]);
      return;
    }
    
    Value oldAcc = acc[0];
    Value currentVal = cur[0];
    Type elemTy = oldAcc.getType();

    Value rawSum = rewriter.create<LLVM::FAddOp>(loc, oldAcc, currentVal);

    Value perturbedSum = perturb_value(rawSum, rewriter, loc, elemTy);

    Value newAcc = perturbedSum;
    if (pred) {
      newAcc = rewriter.create<LLVM::SelectOp>(loc, pred, newAcc, oldAcc);
    }
    
    acc[0] = newAcc;
  }

  SmallVector<SmallVector<Value>>
  unpackInputs(Location loc, triton::ReduceOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const {
    auto types = op.getInputTypes();
    auto operands = adaptor.getOperands();
    unsigned srcElems = getTotalElemsPerThread(types[0]);
    SmallVector<SmallVector<Value>> srcValues(srcElems);
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto values = unpackLLElements(loc, operands[i], rewriter);

      assert(values.size() == srcValues.size());
      for (unsigned j = 0; j < srcValues.size(); ++j) {
        srcValues[j].push_back(values[j]);
      }
    }
    return srcValues;
  }

  void sync(ConversionPatternRewriter &rewriter, Location loc,
            triton::ReduceOp op) const {
    barrier();
  }

  void reduceWithinThreads(
      ReduceOpHelper &helper, SmallVector<SmallVector<Value>> &srcValues,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &indices,
      ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    RankedTensorType operandType = op.getInputTypes()[0];
    SmallVector<SmallVector<unsigned>> offsets =
        emitOffsetForLayout(helper.getSrcLayout(), operandType);

    llvm::MapVector<ArrayRef<unsigned>, int> uniqueOffsets;
    for (int i = 0; i < offsets.size(); ++i) {
      uniqueOffsets.insert({offsets[i], i});
    }

    auto *combineOp = &op.getCombineOp();
    auto srcIndices = emitIndices(op.getLoc(), rewriter, targetInfo,
                                  helper.getSrcLayout(), operandType, true);
    for (const auto &[_, i] : uniqueOffsets) {
      SmallVector<unsigned> key = offsets[i];
      key[op.getAxis()] = 0;
      bool isFirst = accs.find(key) == accs.end();
      accumulate(op.getLoc(), rewriter, *combineOp, accs[key], srcValues[i]);
      if (isFirst)
        indices[key] = srcIndices[i];
    }
  }

  void warpReduce(ConversionPatternRewriter &rewriter, Location loc,
                  SmallVector<Value> &acc, triton::ReduceOp op,
                  unsigned numLaneToReduce, unsigned interleave,
                  Value pred = {}) const {
    auto success = targetInfo.warpReduce(rewriter, loc, acc, op,
                                         numLaneToReduce, interleave);
    if (success)
      return;

    auto mod = op->getParentOfType<ModuleOp>();
    // unsigned iWarpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod); 
    // 注：上面的 iWarpSize 似乎未被使用，保留原逻辑即可

    for (unsigned N = numLaneToReduce / 2; N > 0; N >>= 1) {
      SmallVector<Value> shfl(acc.size());
      for (unsigned i = 0; i < acc.size(); ++i) {
        shfl[i] = targetInfo.shuffleXor(rewriter, loc, acc[i], N * interleave);
      }
      accumulate(op.getLoc(), rewriter, op.getCombineOp(), acc, shfl, pred);
    }
  }

  void
  reduceWithinWarps(ReduceOpHelper &helper,
                    std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
                    ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    unsigned sizeIntraWarps = helper.getIntraWarpSizeWithUniqueData();
    unsigned threadOffsetOnReductionAxis =
        helper.getThreadOffsetOnReductionAxis();
    for (auto it : accs) {
      const SmallVector<unsigned> &key = it.first;
      SmallVector<Value> &acc = accs[key];
      warpReduce(rewriter, op.getLoc(), acc, op, sizeIntraWarps,
                 threadOffsetOnReductionAxis);
    }
  }

  void packResults(ReduceOpHelper &helper,
                   std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
                   ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    Location loc = op.getLoc();
    unsigned axis = op.getAxis();
    SmallVector<Value> results(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      if (auto resultTy =
              dyn_cast<RankedTensorType>(op.getResult()[i].getType())) {
        auto resultLayout = cast<SliceEncodingAttr>(resultTy.getEncoding());
        unsigned resultElems = getTotalElemsPerThread(resultTy);
        SmallVector<SmallVector<unsigned>> resultOffset =
            emitOffsetForLayout(resultLayout, resultTy);
        SmallVector<Value> resultVals;
        for (int j = 0; j < resultElems; j++) {
          auto key = resultOffset[j];
          key.insert(key.begin() + axis, 0);
          resultVals.push_back(accs[key][i]);
        }
        results[i] = packLLElements(loc, getTypeConverter(), resultVals,
                                    rewriter, resultTy);
      } else
        results[i] = accs.begin()->second[i];
    }
    rewriter.replaceOp(op, results);
  }

  SmallVector<Value>
  getMultiDimWarpId(ReduceOpHelper &helper, Value &warpId, Location &loc,
                    ConversionPatternRewriter &rewriter) const {
    auto srcLayout = helper.getSrcLayout();
    auto srcShape = helper.getSrcShape();
    auto order = triton::gpu::getWarpOrder(srcLayout);
    SmallVector<Value> multiDimWarpId;

    if (auto sliceLayout = mlir::dyn_cast<SliceEncodingAttr>(srcLayout)) {
      auto parentLayout = sliceLayout.getParent();
      auto parentWarpsPerCTA = triton::gpu::getWarpsPerCTA(parentLayout);
      auto parentOrder = triton::gpu::getWarpOrder(parentLayout);
      multiDimWarpId =
          delinearize(rewriter, loc, warpId, parentWarpsPerCTA, parentOrder);
      multiDimWarpId.erase(multiDimWarpId.begin() + sliceLayout.getDim());
    } else {
      SmallVector<unsigned> warpsPerCTA =
          triton::gpu::getWarpsPerCTA(srcLayout);
      warpsPerCTA[helper.getAxis()] = triton::gpu::getWarpsPerCTAWithUniqueData(
          srcLayout, srcShape)[helper.getAxis()];
      multiDimWarpId = delinearize(rewriter, loc, warpId, warpsPerCTA, order);
    }
    return multiDimWarpId;
  }

  void storeWarpReduceToSharedMemory(
      ReduceOpHelper &helper,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &indices,
      SmallVector<Value> &smemBases,
      ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    Location loc = op.getLoc();
    Value threadId = getThreadId(rewriter, loc);
    auto srcLayout = helper.getSrcLayout();
    Value warpSize = i32_val(triton::gpu::getWarpSize(srcLayout));
    Value warpId = udiv(threadId, warpSize);
    Value laneId = urem(threadId, warpSize);
    auto srcShape = helper.getSrcShape();
    unsigned axis = op.getAxis();
    auto smemShape = helper.getScratchRepShape();

    auto threadsPerWarp =
        triton::gpu::getThreadsPerWarpWithUniqueData(srcLayout, srcShape);
    auto order = getThreadOrder(srcLayout);
    SmallVector<Value> multiDimLaneId =
        delinearize(rewriter, loc, laneId, threadsPerWarp, order);
    Value laneIdAxis = multiDimLaneId[axis];
    Value zero = i32_val(0);
    Value laneZero = icmp_eq(laneIdAxis, zero);

    SmallVector<Value> multiDimWarpId =
        getMultiDimWarpId(helper, warpId, loc, rewriter);
    Value warpIdAxis = multiDimWarpId[axis];

    auto smemOrder = helper.getOrderWithAxisAtBeginning();
    for (auto it : accs) {
      const SmallVector<unsigned> &key = it.first;
      SmallVector<Value> &acc = it.second;

      SmallVector<Value> writeIdx = indices[key];
      writeIdx[axis] = warpIdAxis;
      Value writeOffset =
          linearize(rewriter, loc, writeIdx, smemShape, smemOrder);
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        auto elemTy = getElementType(op, i);
        Value writePtr =
            gep(smemBases[i].getType(), elemTy, smemBases[i], writeOffset);
        targetInfo.storeShared(rewriter, loc, writePtr, acc[i], laneZero);
      }
    }
  }

  void accumulatePartialReductions(ReduceOpHelper &helper,
                                   SmallVector<Value> &smemBases,
                                   ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    auto srcLayout = helper.getSrcLayout();
    auto smemShape = helper.getScratchRepShape();
    unsigned elems = product<unsigned>(smemShape);
    unsigned sizeInterWarps = helper.getInterWarpSizeWithUniqueData();
    Location loc = op.getLoc();

    Value threadId = getThreadId(rewriter, loc);
    Value warpSize = i32_val(triton::gpu::getWarpSize(srcLayout));
    Value laneId = urem(threadId, warpSize);
    Value zero = i32_val(0);

    auto mod = op.getOperation()->getParentOfType<ModuleOp>();
    unsigned numThreads =
        product<unsigned>(triton::gpu::getWarpsPerCTA(srcLayout)) *
        triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    unsigned elemsPerThread = std::max<unsigned>(elems / numThreads, 1);
    Value threadIsNeeded = icmp_slt(threadId, i32_val(elems));
    Value readOffset = threadId;
    for (unsigned round = 0; round < elemsPerThread; ++round) {
      SmallVector<Value> acc(op.getNumOperands());
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        auto elemTy = getElementType(op, i);
        Value readPtr =
            gep(smemBases[i].getType(), elemTy, smemBases[i], readOffset);
        acc[i] = targetInfo.loadShared(rewriter, loc, readPtr, elemTy,
                                       threadIsNeeded);
      }
      warpReduce(rewriter, loc, acc, op, sizeInterWarps, 1 /* interleave */,
                 threadIsNeeded);
      
      Value writeOffset = readOffset;
      SmallVector<Value> writePtrs(op.getNumOperands());
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        auto elemTy = getElementType(op, i);
        writePtrs[i] =
            gep(smemBases[i].getType(), elemTy, smemBases[i], writeOffset);
      }

      Value laneIdModSizeInterWarps = urem(laneId, i32_val(sizeInterWarps));
      Value laneIdModSizeInterWarpsIsZero =
          icmp_eq(laneIdModSizeInterWarps, zero);
      Value pred = and_(threadIsNeeded, laneIdModSizeInterWarpsIsZero);

      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        targetInfo.storeShared(rewriter, loc, writePtrs[i], acc[i], pred);
      }

      if (round != elemsPerThread - 1) {
        readOffset = add(readOffset, i32_val(numThreads));
      }
    }
  }

  void loadReductionAndPackResult(ReduceOpHelper &helper,
                                  SmallVector<unsigned> smemShape,
                                  SmallVector<Value> &smemBases,
                                  ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    Location loc = op.getLoc();
    auto srcLayout = helper.getSrcLayout();
    // auto axis = op.getAxis(); // 未使用
    auto smemOrder = helper.getOrderWithAxisAtBeginning();
    SmallVector<Value> results(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto elemTy = getElementType(op, i);
      if (auto resultTy =
              dyn_cast<RankedTensorType>(op.getResult()[i].getType())) {
        auto resultLayout = cast<SliceEncodingAttr>(resultTy.getEncoding());
        unsigned resultElems = getTotalElemsPerThread(resultTy);
        auto resultIndices = emitIndices(loc, rewriter, targetInfo,
                                         resultLayout, resultTy, true);
        auto resultShape = resultTy.getShape();
        auto resultCTATile = getShapePerCTATile(resultLayout, resultShape);
        assert(resultIndices.size() == resultElems);

        SmallVector<Value> resultVals(resultElems);
        for (size_t j = 0; j < resultElems; ++j) {
          SmallVector<Value> readIdx = resultIndices[j];
          readIdx.insert(readIdx.begin() + op.getAxis(), i32_val(0));
          for (size_t resultIdx = 0, resultDim = resultShape.size();
               resultIdx < resultDim; ++resultIdx) {
            auto smemIdx = resultIdx < op.getAxis() ? resultIdx : resultIdx + 1;
            if (resultCTATile[resultIdx] > smemShape[smemIdx] ||
                resultShape[resultIdx] > smemShape[smemIdx]) {
              readIdx[smemIdx] =
                  urem(readIdx[smemIdx], i32_val(smemShape[smemIdx]));
            }
          }
          Value readOffset =
              linearize(rewriter, loc, readIdx, smemShape, smemOrder);
          Value readPtr =
              gep(smemBases[i].getType(), elemTy, smemBases[i], readOffset);
          resultVals[j] = load(elemTy, readPtr);
        }

        results[i] = packLLElements(loc, getTypeConverter(), resultVals,
                                    rewriter, resultTy);
      } else {
        results[i] = load(elemTy, smemBases[i]);
      }
    }
    rewriter.replaceOp(op, results);
  }
};
} // namespace

void mlir::triton::populateReduceOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<ReduceOpConversion>(typeConverter, targetInfo, benefit);
}