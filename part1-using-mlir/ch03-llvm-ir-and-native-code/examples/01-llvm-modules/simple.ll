; The smallest useful LLVM IR module: one function that returns 42.
;
; Equivalent C:
;
;   int main() {
;     return 42;
;   }
define i32 @main() {
  ret i32 42
}
