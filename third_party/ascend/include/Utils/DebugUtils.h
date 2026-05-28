/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef ASCEND_UTILS_DEBUGUTILS_H
#define ASCEND_UTILS_DEBUGUTILS_H

#include <cstdlib>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Types.h>
#include <mlir/Support/LLVM.h>
#include <string>
#include <triton/Tools/Sys/GetEnv.hpp>

inline mlir::Location unwrapFusedLocForDebug(mlir::Location loc) {
  // CallSiteLoc: syntax is callsite(callee at caller). For library calls like
  // tl.sum -> callsite(standard.py:306 at kernel.py:36), the kernel line is the
  // caller, so walk the caller side. Nested callsites walk back to the kernel.
  if (auto cs = mlir::dyn_cast<mlir::CallSiteLoc>(loc))
    return unwrapFusedLocForDebug(cs.getCaller());
  if (auto fused = mlir::dyn_cast<mlir::FusedLoc>(loc)) {
    for (auto inner : llvm::reverse(fused.getLocations())) {
      if (!mlir::isa<mlir::UnknownLoc>(inner))
        return unwrapFusedLocForDebug(inner);
    }
  }
  // NameLoc wraps a child (e.g. "extra"(#loc...)); peel to the child.
  if (auto named = mlir::dyn_cast<mlir::NameLoc>(loc))
    return unwrapFusedLocForDebug(named.getChildLoc());
  return loc;
}

/// Insert a side‑effecting nop when TRITON_DEBUG=1 to preserve a source
/// location. Must be called before the operation that carries the location is
/// erased.
inline void insertDebugNop(mlir::Location loc,
                           mlir::PatternRewriter &rewriter) {
  if (!mlir::triton::tools::getBoolEnv("TRITON_DEBUG"))
    return;
  auto unwrapped = unwrapFusedLocForDebug(loc);

  auto ctx = rewriter.getContext();
  rewriter.create<mlir::LLVM::InlineAsmOp>(
      unwrapped,
      /*resultTypes=*/mlir::TypeRange(),
      /*operands=*/mlir::ValueRange(),
      /*asm_string=*/"nop",
      /*constraints=*/"",
      /*has_side_effects=*/true,
      /*is_align_stack=*/false, mlir::LLVM::tailcallkind::TailCallKind::None,
      mlir::LLVM::AsmDialectAttr::get(ctx, mlir::LLVM::AsmDialect::AD_ATT),
      mlir::ArrayAttr());
}

// Emit one NOP per distinct source line in a (possibly fused) location.
// Covers the case where two source-level ops (e.g. tl.full + tl.broadcast_to)
// collapse into a single op carrying a FusedLoc — each original line should
// still get its own breakable PC.
inline void
insertDebugNopForAllLines(mlir::Location loc,
                          mlir::ConversionPatternRewriter &rewriter) {
  if (!mlir::triton::tools::getBoolEnv("TRITON_DEBUG"))
    return;
  if (auto fused = mlir::dyn_cast<mlir::FusedLoc>(loc)) {
    llvm::SmallDenseSet<std::pair<unsigned, unsigned>> seen;
    for (auto inner : fused.getLocations()) {
      auto u = unwrapFusedLocForDebug(inner);
      if (auto flc = mlir::dyn_cast<mlir::FileLineColLoc>(u)) {
        if (seen.insert({flc.getLine(), flc.getColumn()}).second)
          insertDebugNop(u, rewriter); // emits one NOP at this concrete loc
      }
    }
    return;
  }
  insertDebugNop(loc, rewriter);
}

#endif // ASCEND_UTILS_DEBUGUTILS_H
