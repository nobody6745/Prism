#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::NvidiaMmaEncodingAttr;

using ValueTableFMA = std::map<std::pair<int, int>, Value>;

static ValueTableFMA
getValueTable_M_Major(Value val, int K, int n0, int shapePerCTATile,
                      int sizePerThread,
                      ConversionPatternRewriter &rewriter, Location loc,
                      const LLVMTypeConverter *typeConverter, Type type) {
  ValueTableFMA res;
  auto elems = unpackLLElements(loc, val, rewriter);
  int index = 0;
  for (unsigned k = 0; k < K; ++k) {
    for (unsigned m = 0; m < n0; m += shapePerCTATile) {
      int remaining = n0 - m;
      int num_mm = std::min(sizePerThread, remaining);
      for (unsigned mm = 0; mm < num_mm; ++mm) {
        if (index >= static_cast<int>(elems.size())) goto end_loop;
        res[{m + mm, k}] = elems[index++];
      }
    }
  }
end_loop:
  return res;
}

static ValueTableFMA
getValueTable_K_Major(Value val, int K, int n0, int shapePerCTATile,
                      int sizePerThread,
                      ConversionPatternRewriter &rewriter, Location loc,
                      const LLVMTypeConverter *typeConverter, Type type) {
  ValueTableFMA res;
  auto elems = unpackLLElements(loc, val, rewriter);
  int index = 0;
  for (unsigned k = 0; k < K; ++k) {
    for (unsigned m = 0; m < n0; m += shapePerCTATile) {
      int remaining = n0 - m;
      int num_mm = std::min(sizePerThread, remaining);
      for (unsigned mm = 0; mm < num_mm; ++mm) {
        if (index >= static_cast<int>(elems.size())) goto end_loop;
        res[{k, m + mm}] = elems[index++];
      }
    }
  }
end_loop:
  return res;
}

static Value simulateTF32Round(Value val,
                               ConversionPatternRewriter &rewriter,
                               Location loc) {
  if (!val.getType().isF32()) return val;

  auto intTy = rewriter.getIntegerType(32);
  Value bits = rewriter.create<LLVM::BitcastOp>(loc, intTy, val);
  int64_t maskVal = 0xFFFFE000;
  Value mask = rewriter.create<LLVM::ConstantOp>(
      loc, intTy, rewriter.getIntegerAttr(intTy, maskVal));
  Value truncatedBits = rewriter.create<LLVM::AndOp>(loc, intTy, bits, mask);
  return rewriter.create<LLVM::BitcastOp>(loc, val.getType(), truncatedBits);
}

static Value perturb_toward_neg_inf(Value val, ConversionPatternRewriter &rewriter,
                                    Location loc, int64_t shiftAmount) {
  Type floatTy = val.getType();
  if (!mlir::isa<FloatType>(floatTy)) return val;
  
  if (shiftAmount == 0) return val;

  int bits = floatTy.getIntOrFloatBitWidth();
  auto intTy = rewriter.getIntegerType(bits);

  Value shiftVal = rewriter.create<LLVM::ConstantOp>(
      loc, intTy, rewriter.getIntegerAttr(intTy, shiftAmount));
  
  Value zeroF = rewriter.create<LLVM::ConstantOp>(
      loc, floatTy, rewriter.getFloatAttr(floatTy, 0.0));
  
  int64_t smallestNegSubnormalBits = (1LL << (bits - 1)) | 1LL;
  Value smallestNegConst = rewriter.create<LLVM::ConstantOp>(
      loc, intTy, rewriter.getIntegerAttr(intTy, smallestNegSubnormalBits));
  Value smallestNegF = rewriter.create<LLVM::BitcastOp>(loc, floatTy, smallestNegConst);

  Value isFinite = rewriter.create<LLVM::FCmpOp>(loc, LLVM::FCmpPredicate::ord, val, val);
  Value isZero = rewriter.create<LLVM::FCmpOp>(loc, LLVM::FCmpPredicate::oeq, val, zeroF);
  Value isNeg = rewriter.create<LLVM::FCmpOp>(loc, LLVM::FCmpPredicate::olt, val, zeroF);
  
  Value valBits = rewriter.create<LLVM::BitcastOp>(loc, intTy, val);

  Value bitsSub = rewriter.create<LLVM::SubOp>(loc, intTy, valBits, shiftVal);
  Value bitsAdd = rewriter.create<LLVM::AddOp>(loc, intTy, valBits, shiftVal);
  
  Value perturbedBits = rewriter.create<LLVM::SelectOp>(loc, isNeg, bitsAdd, bitsSub);
  Value perturbedF = rewriter.create<LLVM::BitcastOp>(loc, floatTy, perturbedBits);

  Value resultOrZero = rewriter.create<LLVM::SelectOp>(loc, isZero, smallestNegF, perturbedF);
  return rewriter.create<LLVM::SelectOp>(loc, isFinite, resultOrZero, val);
}

LogicalResult convertFMADot(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                            const LLVMTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter) {
  auto *ctx = rewriter.getContext();
  auto loc = op.getLoc();

  auto A = op.getA();
  auto B = op.getB();
  auto D = op.getResult();

  auto aTensorTy = cast<RankedTensorType>(A.getType());
  auto bTensorTy = cast<RankedTensorType>(B.getType());
  auto dTensorTy = cast<RankedTensorType>(D.getType());

  Type origTy = aTensorTy.getElementType();
  Type mulTy;
  Type accTy;
  
  bool needInputRounding = false; 
  int64_t mulPerturbAmount = 0; 
  int64_t accPerturbAmount = 1; 

  if (origTy.isF32()) {
    mulTy = rewriter.getF32Type();
    accTy = rewriter.getF32Type();
    
    needInputRounding = true; 
    

    mulPerturbAmount = (2 * (1LL << 13)) + 8; 
    accPerturbAmount = 1;

  } else if (origTy.isF16()) {
    mulTy = rewriter.getF32Type();
    accTy = rewriter.getF32Type();
    
    needInputRounding = false; 

    mulPerturbAmount = 1; 
    accPerturbAmount = 1; 
    
  } else if (origTy.isBF16()) {
    mulTy = rewriter.getF32Type();
    accTy = rewriter.getF32Type();
    needInputRounding = false; 
    mulPerturbAmount = 0;
    accPerturbAmount = 1;
  } else {
    mulTy = origTy;
    accTy = origTy;
  }

  auto aShapePerCTA = getShapePerCTA(aTensorTy);
  auto bShapePerCTA = getShapePerCTA(bTensorTy);
  BlockedEncodingAttr dLayout = cast<BlockedEncodingAttr>(dTensorTy.getEncoding());
  auto order = dLayout.getOrder();
  
  auto cc = unpackLLElements(loc, adaptor.getC(), rewriter);
  SmallVector<Value> c_vals(cc.size()); 
  for (size_t i = 0; i < cc.size(); ++i) {
    if (accTy != origTy) {
      c_vals[i] = rewriter.create<LLVM::FPExtOp>(loc, accTy, cc[i]);
    } else {
      c_vals[i] = cc[i];
    }
  }

  Value zeroAcc = rewriter.create<LLVM::ConstantOp>(
      loc, accTy, rewriter.getFloatAttr(accTy, 0.0));
  
  Type targetStructTy = typeConverter->convertType(dTensorTy);
  unsigned numDElems = 0;
  if (auto structTy = dyn_cast<LLVM::LLVMStructType>(targetStructTy)) {
      numDElems = structTy.getBody().size();
  } else {
      numDElems = 1; 
  }
  SmallVector<Value> ret(numDElems, zeroAcc); 
  
  Value llA = adaptor.getA();
  Value llB = adaptor.getB();

  auto sizePerThread = getSizePerThread(dLayout);
  auto shapePerCTATile = getShapePerCTATile(dLayout);
  
  int K = aShapePerCTA[1];
  int M = aShapePerCTA[0];
  int N = bShapePerCTA[1];

  int mShapePerCTATile = order[0] == 1 ? shapePerCTATile[order[1]] : shapePerCTATile[order[0]];
  int mSizePerThread = order[0] == 1 ? sizePerThread[order[1]] : sizePerThread[order[0]];
  int nShapePerCTATile = order[0] == 0 ? shapePerCTATile[order[1]] : shapePerCTATile[order[0]];
  int nSizePerThread = order[0] == 0 ? sizePerThread[order[1]] : sizePerThread[order[0]];

  auto has = getValueTable_M_Major(llA, K, M, mShapePerCTATile, mSizePerThread, rewriter, loc, typeConverter, aTensorTy);
  auto hbs = getValueTable_K_Major(llB, K, N, nShapePerCTATile, nSizePerThread, rewriter, loc, typeConverter, bTensorTy);
  
  bool isCRow = order[0] == 1;
  int numMBlocks = (M + mShapePerCTATile - 1) / mShapePerCTATile;
  int totalMThreadSlots = numMBlocks * mSizePerThread;
  int numNBlocks = (N + nShapePerCTATile - 1) / nShapePerCTATile;
  int totalNThreadSlots = numNBlocks * nSizePerThread;

  auto convertFloat = [&](Value src, Type dstTy) -> Value {
    int srcBits = src.getType().getIntOrFloatBitWidth();
    int dstBits = dstTy.getIntOrFloatBitWidth();
    if (srcBits == dstBits) return src; 
    if (dstBits > srcBits) return rewriter.create<LLVM::FPExtOp>(loc, dstTy, src);
    return rewriter.create<LLVM::FPTruncOp>(loc, dstTy, src);
  };

  for (unsigned k = 0; k < K; k++) {
    for (unsigned m = 0; m < M; m += mShapePerCTATile) {
      for (unsigned n = 0; n < N; n += nShapePerCTATile) {
        for (unsigned mm = 0; mm < mSizePerThread; ++mm) {
          unsigned row = m + mm;
          if (row >= static_cast<unsigned>(M)) continue;
          for (unsigned nn = 0; nn < nSizePerThread; ++nn) {
            unsigned col = n + nn;
            if (col >= static_cast<unsigned>(N)) continue;
            unsigned m_block_idx = m / mShapePerCTATile;
            unsigned n_block_idx = n / nShapePerCTATile;
            int mIdx = static_cast<int>(m_block_idx * static_cast<unsigned>(mSizePerThread) + mm);
            int nIdx = static_cast<int>(n_block_idx * static_cast<unsigned>(nSizePerThread) + nn);
            int z = isCRow ? mIdx * totalNThreadSlots + nIdx : nIdx * totalMThreadSlots + mIdx;

            if (z < 0 || z >= static_cast<int>(ret.size())) continue; 

            Value aVal = has[{static_cast<int>(row), k}];
            Value bVal = hbs[{k, static_cast<int>(col)}];

            if (!aVal || !bVal) continue;

            Value a_mul = convertFloat(aVal, mulTy);
            Value b_mul = convertFloat(bVal, mulTy);

            if (needInputRounding) {
                a_mul = simulateTF32Round(a_mul, rewriter, loc);
                b_mul = simulateTF32Round(b_mul, rewriter, loc);
            }
            Value product = rewriter.create<LLVM::FMulOp>(loc, mulTy, a_mul, b_mul);

            Value perturbedProduct = perturb_toward_neg_inf(product, rewriter, loc, mulPerturbAmount);
            
            Value product_acc = convertFloat(perturbedProduct, accTy);
            
            Value currentAcc = ret[z];
            Value tempSum = rewriter.create<LLVM::FAddOp>(loc, accTy, currentAcc, product_acc);

            Value perturbedSum = perturb_toward_neg_inf(tempSum, rewriter, loc, accPerturbAmount);

            ret[z] = perturbedSum; 
          }
        }
      }
    }
  }

  for (size_t i = 0; i < ret.size(); ++i) {
    Value c_i = (c_vals.size() == 1) ? c_vals[0] : c_vals[i];
    Value result_i = rewriter.create<LLVM::FAddOp>(loc, accTy, ret[i], c_i);
    ret[i] = result_i;
  }

  Type dElemTy = dTensorTy.getElementType(); 
  Value finalStruct = rewriter.create<LLVM::UndefOp>(loc, targetStructTy);
  
  for (const auto &en : llvm::enumerate(ret)) {
      Value valToInsert = en.value(); 
      
      if (accTy != dElemTy) {
        valToInsert = rewriter.create<LLVM::FPTruncOp>(loc, dElemTy, valToInsert);
      }

      finalStruct = rewriter.create<LLVM::InsertValueOp>(loc, finalStruct, valToInsert, en.index());
    }

  rewriter.replaceOp(op, finalStruct);
  return success();
}