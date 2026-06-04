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
#include <functional>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Types.h>
#include <mlir/Support/LLVM.h>
#include <string>
#include <triton/Tools/Sys/GetEnv.hpp>

inline mlir::Location unwrapFusedLocForDebug(mlir::Location loc) {
  if (auto cs = mlir::dyn_cast<mlir::CallSiteLoc>(loc))
    return unwrapFusedLocForDebug(cs.getCaller());
  if (auto fused = mlir::dyn_cast<mlir::FusedLoc>(loc)) {
    for (auto inner : llvm::reverse(fused.getLocations())) {
      if (!mlir::isa<mlir::UnknownLoc>(inner))
        return unwrapFusedLocForDebug(inner);
    }
  }
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

inline void
insertDebugNopForAllLines(mlir::Location loc,
                          mlir::ConversionPatternRewriter &rewriter) {
  if (!mlir::triton::tools::getBoolEnv("TRITON_DEBUG"))
    return;

  std::function<mlir::Location(mlir::Location)> deepUnwrap =
      [&](mlir::Location x) -> mlir::Location {
    if (auto cs = mlir::dyn_cast<mlir::CallSiteLoc>(x))
      return deepUnwrap(cs.getCaller());
    if (auto n = mlir::dyn_cast<mlir::NameLoc>(x))
      return deepUnwrap(n.getChildLoc());
    if (auto f = mlir::dyn_cast<mlir::FusedLoc>(x)) {
      for (auto inner : llvm::reverse(f.getLocations()))
        if (!mlir::isa<mlir::UnknownLoc>(inner))
          return deepUnwrap(inner);
    }
    return x;
  };

  if (auto fused = mlir::dyn_cast<mlir::FusedLoc>(loc)) {
    llvm::SmallDenseSet<std::pair<unsigned, unsigned>> seen;
    for (auto inner : fused.getLocations()) {
      auto u = deepUnwrap(inner);
      if (auto flc = mlir::dyn_cast<mlir::FileLineColLoc>(u)) {
        if (seen.insert({flc.getLine(), flc.getColumn()}).second)
          insertDebugNop(u, rewriter);
      }
    }
    return;
  }
  insertDebugNop(loc, rewriter);
}

namespace mlir {
namespace triton {
namespace debug {

/// A library/stdlib file inlined into the kernel (e.g. triton/language/
/// standard.py). Heuristic: lives under a site-packages tree.
inline bool isForeignFile(llvm::StringRef filename) {
  return filename.contains("/site-packages/");
}

/// Is `op` one of our debug NOPs?
inline bool isDebugNop(Operation *op) {
  auto asmOp = dyn_cast<LLVM::InlineAsmOp>(op);
  if (!asmOp)
    return false;
  if (!asmOp.getHasSideEffects())
    return false;
  if (asmOp.getAsmString() != "nop")
    return false;
  if (asmOp->getNumResults() != 0 || asmOp->getNumOperands() != 0)
    return false;
  return true;
}

/// First FileLineColLoc reachable from `loc` (callee-first for call sites).
/// Answers "what file/line does this point at".
inline FileLineColLoc firstFileLineCol(Location loc, unsigned depth = 0) {
  if (depth > 16)
    return {};
  if (auto flc = dyn_cast<FileLineColLoc>(loc))
    return flc;
  if (auto named = dyn_cast<NameLoc>(loc))
    return firstFileLineCol(named.getChildLoc(), depth + 1);
  if (auto cs = dyn_cast<CallSiteLoc>(loc)) {
    if (auto c = firstFileLineCol(cs.getCallee(), depth + 1))
      return c;
    return firstFileLineCol(cs.getCaller(), depth + 1);
  }
  if (auto fused = dyn_cast<FusedLoc>(loc))
    for (Location sub : fused.getLocations())
      if (auto c = firstFileLineCol(sub, depth + 1))
        return c;
  return {};
}

/// Unwrap to the FileLineColLoc the *user* should see, preferring the CALLER
/// frame for call sites. callsite(stdlib at user) -> the user's line.
inline FileLineColLoc unwrapToUserFileLineCol(Location loc,
                                              unsigned depth = 0) {
  if (depth > 16)
    return {};
  if (auto flc = dyn_cast<FileLineColLoc>(loc))
    return flc;
  if (auto named = dyn_cast<NameLoc>(loc))
    return unwrapToUserFileLineCol(named.getChildLoc(), depth + 1);
  if (auto cs = dyn_cast<CallSiteLoc>(loc)) {
    if (auto outer = unwrapToUserFileLineCol(cs.getCaller(), depth + 1))
      return outer;
    return unwrapToUserFileLineCol(cs.getCallee(), depth + 1);
  }
  if (auto fused = dyn_cast<FusedLoc>(loc))
    for (Location sub : fused.getLocations())
      if (auto inner = unwrapToUserFileLineCol(sub, depth + 1))
        return inner;
  return {};
}

/// Rewrite call-site locations whose callee resolves to a foreign (stdlib)
/// file so they collapse to their caller (user) frame. Recurses through
/// NameLoc / FusedLoc / nested call sites.
inline Location collapseForeignCallsites(Location loc, unsigned depth = 0) {
  if (depth > 16)
    return loc;

  if (auto named = dyn_cast<NameLoc>(loc)) {
    Location child = collapseForeignCallsites(named.getChildLoc(), depth + 1);
    return child == named.getChildLoc()
               ? loc
               : Location(NameLoc::get(named.getName(), child));
  }

  if (auto cs = dyn_cast<CallSiteLoc>(loc)) {
    Location caller = collapseForeignCallsites(cs.getCaller(), depth + 1);
    if (FileLineColLoc calleeFlc = firstFileLineCol(cs.getCallee()))
      if (isForeignFile(calleeFlc.getFilename().getValue()))
        return caller; // drop the inlined library frame
    Location callee = collapseForeignCallsites(cs.getCallee(), depth + 1);
    if (callee == cs.getCallee() && caller == cs.getCaller())
      return loc;
    return Location(CallSiteLoc::get(callee, caller));
  }

  if (auto fused = dyn_cast<FusedLoc>(loc)) {
    llvm::SmallVector<Location> newLocs;
    bool changed = false;
    for (Location sub : fused.getLocations()) {
      Location c = collapseForeignCallsites(sub, depth + 1);
      changed |= (c != sub);
      newLocs.push_back(c);
    }
    return changed ? Location(FusedLoc::get(loc.getContext(), newLocs,
                                            fused.getMetadata()))
                   : loc;
  }

  return loc;
}

} // namespace debug
} // namespace triton
} // namespace mlir

#endif // ASCEND_UTILS_DEBUGUTILS_H
