/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#ifndef TRITON_ASCEND_UTILS_DEBUGUTILS_H
#define TRITON_ASCEND_UTILS_DEBUGUTILS_H

#include <llvm/ADT/StringRef.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Value.h>

namespace mlir {
class Operation;
class PatternRewriter;
class ConversionPatternRewriter;
} // namespace mlir

//===----------------------------------------------------------------------===//
// NOP insertion helpers (gated by LLVM_EXTRACT_DI_LOCAL_VARIABLES). Definitions
// in DebugUtils.cpp.
//===----------------------------------------------------------------------===//

mlir::Location unwrapFusedLocForDebug(mlir::Location loc, unsigned depth = 0);

/// Insert a side-effecting nop when LLVM_EXTRACT_DI_LOCAL_VARIABLES=1 to
/// preserve a source location. Must be called before the op carrying the
/// location is erased.
void insertDebugNop(mlir::Location loc, mlir::PatternRewriter &rewriter);

void insertDebugNopForMask(mlir::Value mask, mlir::PatternRewriter &rewriter);

void insertDebugNopForAllLines(mlir::Location loc,
                               mlir::ConversionPatternRewriter &rewriter);

//===----------------------------------------------------------------------===//
// Shared location analysis / rewrite helpers used by the debug passes
// (CanonicalizeDebugLocationsPass, DeduplicateDebugNopsPass).
// Definitions in DebugUtils.cpp. These do NOT self-gate; the passes gate.
//===----------------------------------------------------------------------===//

namespace mlir {
namespace triton {
namespace debug {

bool isForeignFile(llvm::StringRef filename);

bool isDebugNop(Operation *op);

FileLineColLoc unwrapToUserFileLineCol(Location loc);

Location collapseForeignCallsites(Location loc, unsigned depth = 0);

bool isDebugNopEnabled();

} // namespace debug
} // namespace triton
} // namespace mlir

#endif // TRITON_ASCEND_UTILS_DEBUGUTILS_H
