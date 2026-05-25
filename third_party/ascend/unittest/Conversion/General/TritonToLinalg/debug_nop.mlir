// Test: insertDebugNop emits llvm.inline_asm "nop" ops only when
// TRITON_DEBUG=1 is set, with the original source location preserved
// on each NOP. Without TRITON_DEBUG, no inline asm is emitted.
//
// RUN:                  %triton-opt --load-dialect=llvm --triton-to-linalg --split-input-file %s | %FileCheck %s --check-prefix=NODEBUG
// RUN: TRITON_DEBUG=1    %triton-opt --load-dialect=llvm --triton-to-linalg --split-input-file %s | %FileCheck %s --check-prefix=DEBUG

#loc = loc("test.py":6:0)
#loc1 = loc("test.py":11:24)
#loc2 = loc("test.py":12:32)
#loc3 = loc("test.py":16:28)
#loc_pid = loc("pid"(#loc1))
#loc_numprogs = loc("num_progs"(#loc2))
#loc_offsets = loc("offsets"(#loc3))

tt.func public @test_kernel(
    %ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
    %n: i32 {tt.divisibility = 16 : i32},
    %arg2: i32, %arg3: i32, %arg4: i32,
    %arg5: i32, %arg6: i32, %arg7: i32
) attributes {noinline = false} {
  %pid = tt.get_program_id x : i32 loc(#loc_pid)
  %np  = tt.get_num_programs x : i32 loc(#loc_numprogs)
  %c128 = arith.constant 128 : i32
  %block_start = arith.muli %pid, %c128 : i32
  %offs = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %splat_bs = tt.splat %block_start : i32 -> tensor<128xi32>
  %abs_offs = arith.addi %splat_bs, %offs : tensor<128xi32> loc(#loc_offsets)
  %splat_ptr = tt.splat %ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %addptr = tt.addptr %splat_ptr, %abs_offs : tensor<128x!tt.ptr<f32>>, tensor<128xi32> loc(#loc_offsets)
  %sum = arith.addi %pid, %np : i32
  %splat_sum = tt.splat %sum : i32 -> tensor<128xi32>
  %sum_f32 = arith.sitofp %splat_sum : tensor<128xi32> to tensor<128xf32>
  tt.store %addptr, %sum_f32 : tensor<128x!tt.ptr<f32>>
  tt.return
} loc(#loc)

// DEBUG-LABEL: func.func @test_kernel
// DEBUG: llvm.inline_asm has_side_effects asm_dialect = att "nop" {{.*}}loc("pid"
// DEBUG: llvm.inline_asm has_side_effects asm_dialect = att "nop" {{.*}}loc("num_progs"
// DEBUG: llvm.inline_asm has_side_effects asm_dialect = att "nop" {{.*}}loc("offsets"

// NODEBUG-LABEL: func.func @test_kernel
// NODEBUG-NOT: llvm.inline_asm
// NODEBUG: return