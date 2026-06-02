#ifndef TRITON_ASCEND_TRITONTOLINALG_DEDUPLICATE_DEBUG_NOPS_H
#define TRITON_ASCEND_TRITONTOLINALG_DEDUPLICATE_DEBUG_NOPS_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

/// Creates a pass that deduplicates `llvm.inline_asm "nop"` debug-anchor ops
/// inserted by the converters during triton-to-linalg lowering.
///
/// Each source line typically has many NOPs scattered through the lowered IR
/// because converters insert one NOP per occurrence (e.g., every consumer of
/// an `offsets` value gets a NOP at its source line). This produces noisy
/// DWARF line tables with the same source line repeated at many PCs, which
/// makes debugger `next`-stepping bounce around.
///
/// This pass keeps only the FIRST NOP per unique source line (by `(filename,
/// line)`) within each function, in IR walk order. The first NOP gets a real
/// PC; subsequent duplicates are erased.
///
/// The pass is opt-in: it does nothing unless the env var
/// `LLVM_EXTRACT_DI_LOCAL_VARIABLES=1` is set. Production compiles are
/// unaffected.
std::unique_ptr<Pass> createDeduplicateDebugNopsPass();

#define GEN_PASS_DECL_DEDUPLICATEDEBUGNOPS
#include "TritonToLinalg/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TRITON_ASCEND_TRITONTOLINALG_DEDUPLICATE_DEBUG_NOPS_H