#ifndef ASCEND_UTILS_DEBUGUTILS_H
#define ASCEND_UTILS_DEBUGUTILS_H

#include <cstdlib>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Types.h>
#include <mlir/Support/LLVM.h>
#include <string>

/// Insert a side‑effecting nop when TRITON_DEBUG=1 to preserve a source
/// location. Must be called before the operation that carries the location is
/// erased.
inline void insertDebugNop(mlir::Location loc,
                           mlir::ConversionPatternRewriter &rewriter) {
  bool debugMode = false;
  if (const char *env = std::getenv("TRITON_DEBUG"))
    debugMode = (std::string(env) == "1");
  if (!debugMode)
    return;

  auto ctx = rewriter.getContext();
  rewriter.create<mlir::LLVM::InlineAsmOp>(
      loc,
      /*resultTypes=*/mlir::TypeRange(),
      /*operands=*/mlir::ValueRange(),
      /*asm_string=*/"nop",
      /*constraints=*/"",
      /*has_side_effects=*/true,
      /*is_align_stack=*/false, mlir::LLVM::tailcallkind::TailCallKind::None,
      mlir::LLVM::AsmDialectAttr::get(ctx, mlir::LLVM::AsmDialect::AD_ATT),
      mlir::ArrayAttr());
}

#endif // ASCEND_UTILS_DEBUGUTILS_H
