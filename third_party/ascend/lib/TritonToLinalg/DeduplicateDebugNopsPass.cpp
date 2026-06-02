//===----------------------------------------------------------------------===//
// DeduplicateDebugNopsPass
//
// Walks each function and removes redundant `llvm.inline_asm "nop"` ops that
// were inserted as DWARF anchors by the triton-to-linalg converters. Keeps
// the first NOP per unique source line; erases the rest.
//
// Gated by env var `LLVM_EXTRACT_DI_LOCAL_VARIABLES=1`.
//===----------------------------------------------------------------------===//

#include "TritonToLinalg/DeduplicateDebugNopsPass.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"

#include "triton/Tools/Sys/GetEnv.hpp"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "dedup-debug-nops"

namespace mlir {
namespace triton {

namespace {

/// Unwrap nested locations (NameLoc, CallSiteLoc, FusedLoc) to find the
/// underlying `FileLineColLoc`. Returns a null FileLineColLoc if nothing usable.
FileLineColLoc unwrapToFileLineCol(Location loc, unsigned depth = 0) {
  // Bound recursion in case of unexpectedly nested locations.
  if (depth > 16)
    return {};

  if (auto flc = dyn_cast<FileLineColLoc>(loc))
    return flc;

  if (auto named = dyn_cast<NameLoc>(loc))
    return unwrapToFileLineCol(named.getChildLoc(), depth + 1);

  if (auto cs = dyn_cast<CallSiteLoc>(loc)) {
    // Prefer the callee location (the place the code was "logically" at).
    if (auto inner = unwrapToFileLineCol(cs.getCallee(), depth + 1))
      return inner;
    return unwrapToFileLineCol(cs.getCaller(), depth + 1);
  }

  if (auto fused = dyn_cast<FusedLoc>(loc)) {
    for (Location sub : fused.getLocations()) {
      if (auto inner = unwrapToFileLineCol(sub, depth + 1))
        return inner;
    }
  }

  return {};
}

/// Returns true if `op` is one of our debug NOPs:
///   `llvm.inline_asm has_side_effects asm_dialect = att "nop", "" : () -> ()`
bool isDebugNop(Operation *op) {
  auto asmOp = dyn_cast<LLVM::InlineAsmOp>(op);
  if (!asmOp)
    return false;
  if (!asmOp.getHasSideEffects())
    return false;
  if (asmOp.getAsmString() != "nop")
    return false;
  // Sanity: our NOPs have no results and no operands.
  if (asmOp->getNumResults() != 0 || asmOp->getNumOperands() != 0)
    return false;
  return true;
}

struct DeduplicateDebugNopsPass
    : public PassWrapper<DeduplicateDebugNopsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DeduplicateDebugNopsPass)

  StringRef getArgument() const final { return "deduplicate-debug-nops"; }

  StringRef getDescription() const final {
    return "Deduplicate llvm.inline_asm 'nop' debug-anchor ops by source line.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    // Opt-in via env var; do nothing in production builds.
    if (!::triton::tools::getBoolEnv("LLVM_EXTRACT_DI_LOCAL_VARIABLES"))
      return;

    ModuleOp moduleOp = getOperation();
    unsigned totalDropped = 0;
    unsigned totalKept = 0;

    moduleOp.walk([&](func::FuncOp func) {
      // Per-function dedup: key = (filename, line). Column intentionally
      // ignored — for debugger stepping behaviour, two NOPs on the same line
      // at different columns are equivalent.
      llvm::DenseSet<std::pair<StringRef, unsigned>> seen;
      llvm::SmallVector<Operation *, 16> toErase;

      func.walk([&](Operation *op) {
        if (!isDebugNop(op))
          return;

        FileLineColLoc flc = unwrapToFileLineCol(op->getLoc());
        if (!flc) {
          // No resolvable source location; leave the NOP alone.
          LLVM_DEBUG(llvm::dbgs()
                     << "[dedup-nops] NOP with no FileLineColLoc, keeping: "
                     << *op << "\n");
          return;
        }

        auto key = std::make_pair(flc.getFilename().getValue(), flc.getLine());
        if (!seen.insert(key).second) {
          toErase.push_back(op);
        }
      });

      totalKept += seen.size();
      totalDropped += toErase.size();

      for (Operation *op : toErase)
        op->erase();
    });

    LLVM_DEBUG(llvm::dbgs() << "[dedup-nops] dropped " << totalDropped
                            << " duplicate NOPs, kept " << totalKept
                            << " unique anchors\n");
  }
};

} // namespace

std::unique_ptr<Pass> createDeduplicateDebugNopsPass() {
  return std::make_unique<DeduplicateDebugNopsPass>();
}

} // namespace triton
} // namespace mlir