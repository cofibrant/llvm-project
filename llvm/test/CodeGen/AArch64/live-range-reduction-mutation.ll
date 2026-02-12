; REQUIRES: asserts
; RUN: llc < %s -mtriple=arm64-apple-ios -debug-only=machine-scheduler -aarch64-enable-live-range-reduction=true -o /dev/null 2>&1 | FileCheck %s --check-prefix=ENABLED
; RUN: llc < %s -mtriple=arm64-apple-ios -debug-only=machine-scheduler -aarch64-enable-live-range-reduction=false -o /dev/null 2>&1 | FileCheck %s --check-prefix=DISABLED

; ENABLED: *** Begin live range reduction mutation ***
; ENABLED: *** End live range reduction mutation ***

; DISABLED-NOT: *** Begin live range reduction mutation ***

define void @test(ptr %p, i64 %v) {
  store i64 %v, ptr %p
  ret void
}
