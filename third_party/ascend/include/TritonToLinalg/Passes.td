#ifndef TRITON_TO_LINALG_CONVERSION_PASSES
#define TRITON_TO_LINALG_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonToLinalg : Pass<"triton-to-linalg", "mlir::ModuleOp"> {
    let summary = "Convert Triton to Linalg dialect";
    let constructor = "triton::createTritonToLinalgPass()";
    let options = [
        Option<"globalKernel", "global-kernel",
            "bool", /*default*/"true",
            "generate a global kernel">,
        Option<"namedOps", "named-ops",
            "bool", /*default*/"false",
            "use linalg named ops instead of linalg.generic">,
        Option<"enableNd2nzOnVector", "enable-nd2nz-on-vector",
            "bool", /*default*/"false",
            "enable nd2nz on vector">,
        Option<"enableSelectAnalysis", "enable-select-analysis",
            "bool", /*default*/"true",
            "enable select analysis">,
        Option<"compileOn91095", "compile-on-910-95",
            "bool", /*default*/"false",
            "compile on 910_95">
    ];
}

def MarkTensorKind : Pass<"mark-tensor-kind", "mlir::ModuleOp"> {
    let summary = "Mark tensor kind (INPUT/OUTPUT/INPUT_OUTPUT) on Triton function arguments";
    let constructor = "triton::createMarkTensorKindPass()";
}

def DeduplicateDebugNops : Pass<"deduplicate-debug-nops", "mlir::ModuleOp"> {
    let summary = "Deduplicate llvm.inline_asm 'nop' debug-anchor ops by source line.";
    let description = [{
        Removes redundant `llvm.inline_asm "nop"` ops that share a
        `(filename, line)` source location within a function. Keeps the
        first occurrence in IR walk order; erases the rest.

        Opt-in via env var `LLVM_EXTRACT_DI_LOCAL_VARIABLES=1`. With the
        env var unset or zero the pass is a no-op, so production builds
        are unaffected.

        These NOPs are inserted by the triton-to-linalg converters as
        DWARF anchors for ops that would otherwise leave no machine code
        behind (e.g. `tt.get_program_id`, `tt.reshape`). Each consumer of
        a "vanishing" SSA value tends to receive its own NOP, producing
        many anchors at the same source line. This pass collapses those
        duplicates so the DWARF line table has at most one is_stmt entry
        per source line within a function.
    }];
    let constructor = "triton::createDeduplicateDebugNopsPass()";
    let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}

#endif // TRITON_TO_LINALG_CONVERSION_PASSES
