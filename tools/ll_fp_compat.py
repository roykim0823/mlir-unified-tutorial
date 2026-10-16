#!/usr/bin/env python3
"""Rewrite LLVM 24 typed float hex literals to the legacy syntax.

The MLIR wheel's `mlir-translate` (LLVM 24) prints float constants as typed
hex literals like `float f0x3EA3B35C`. Older clang/llc (Apple clang, LLVM
<= 20) only accept the legacy form, where a float literal is written as the
16-digit hex of the *double* with the same value: `float 0x3FD4766B80000000`.

LLVM 24 also prints f32 constants as shortest-round-trip decimals (e.g.
`float -4.291610e-01`); older parsers reject those as "invalid for type"
because narrowing the decimal-as-double to float is not exact. So this
filter converts:        stdin (.ll text)  ->  stdout (.ll text)
  - f0x + 16 hex digits (double)   -> 0x + same digits
  - f0x +  8 hex digits (float)    -> 0x + double-widened digits
  - float <decimal>                -> float 0x<double-widened digits>
  - the same for operands written without their type: the second operand
    of fadd/fsub/fmul/fdiv/frem/fcmp on float, and `phi float` incomings

  - float/double inf, -inf, nan       -> the equivalent hex literal
  - the `nocreateundeforpoison` function attribute (LLVM 24 puts it on
    intrinsic declarations such as @llvm.exp.f32) -> dropped; clang 20
    rejects it as "unterminated attribute group" and it is only an
    optimization hint

Other typed literals (half/bfloat/fp128 forms) are left untouched and will
still be rejected downstream — loudly, which is what you want.
"""

import re
import struct
import sys


def widen_bits32(bits32: int) -> str:
    (value,) = struct.unpack(">f", bits32.to_bytes(4, "big"))
    (bits64,) = struct.unpack(">Q", struct.pack(">d", value))
    return f"0x{bits64:016X}"


def widen_f32_hex(match: re.Match) -> str:
    return widen_bits32(int(match.group(1), 16))


def widen_f32_decimal(match: re.Match) -> str:
    # Round the decimal to the nearest float (what the printer meant),
    # then print that float exactly as a double-widened hex literal.
    (bits32,) = struct.unpack(">I", struct.pack(">f", float(match.group(1))))
    return f"float {widen_bits32(bits32)}"


_SPECIAL = {"inf": "0x7FF0000000000000", "-inf": "0xFFF0000000000000",
            "nan": "0x7FF8000000000000", "-nan": "0xFFF8000000000000"}


def special_literal(match: re.Match) -> str:
    value = match.group(2).lstrip("+")
    return f"{match.group(1)} {_SPECIAL[value]}"


_DEC = r"-?\d+\.\d+(?:[eE][+-]?\d+)?"
# Instructions whose type appears once, before the first operand.
_FLOAT_BINOP = re.compile(r"\b(?:fadd|fsub|fmul|fdiv|frem|fcmp\s+\w+)\b[^,]*?\bfloat\b")
_FLOAT_PHI = re.compile(r"\bphi\b[^\[]*?\bfloat\b")


def untyped_operand(match: re.Match) -> str:
    value = match.group(2)
    if value.lstrip("+-") in ("inf", "nan"):
        return match.group(1) + _SPECIAL[value.lstrip("+")]
    (bits32,) = struct.unpack(">I", struct.pack(">f", float(value)))
    return match.group(1) + widen_bits32(bits32)


def fix_untyped_operands(line: str) -> str:
    operand = rf"({_DEC}|[+-]?(?:inf|nan))\b"
    if _FLOAT_BINOP.search(line):
        line = re.sub(rf"(,\s*){operand}", untyped_operand, line)
    if _FLOAT_PHI.search(line):
        line = re.sub(rf"(\[\s*){operand}", untyped_operand, line)
    return line


_F64_HEX = re.compile(r"\bf0x([0-9A-Fa-f]{16})\b")
_F32_HEX = re.compile(r"\bf0x([0-9A-Fa-f]{8})\b")
_F32_DEC = re.compile(r"\bfloat (-?\d+\.\d+(?:[eE][+-]?\d+)?)\b")
_INF_NAN = re.compile(r"\b(float|double) ([+-]?(?:inf|nan))\b")


def fix_line(line: str) -> str:
    line = _F64_HEX.sub(r"0x\1", line)
    line = _F32_HEX.sub(widen_f32_hex, line)
    line = _F32_DEC.sub(widen_f32_decimal, line)
    line = _INF_NAN.sub(special_literal, line)
    line = fix_untyped_operands(line)
    return line.replace(" nocreateundeforpoison", "")


def main() -> None:
    # Every rewrite is local to one line, so stream: a full GPT-2 module is
    # a 2+ GB .ll file that should not be held in memory several times over.
    for line in sys.stdin:
        sys.stdout.write(fix_line(line))


if __name__ == "__main__":
    main()
