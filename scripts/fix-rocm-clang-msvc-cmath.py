#!/usr/bin/env python3
"""Make ROCm clang 23's HIP headers usable with VS 18 / MSVC 14.51 (idempotent).

MSVC 14.51 declares the two-argument comparison predicates in <cmath> as clang builtin
wrappers, and <limits> -- which ROCm's HIP math headers include -- now pulls <cmath> in. The
__device__ overloads the HIP headers then declare cannot be overloaded onto those, so every HIP
translation unit fails with:

    error: __device__ function 'isgreater' cannot overload __host__ __device__ function 'isgreater'

MSVC's own declarations call the same __builtin_* and are visible from device code, so skipping
the duplicates keeps the semantics. Run this against a ROCm SDK prefix (default: $HIP_PATH):

    python scripts/fix-rocm-clang-msvc-cmath.py [--prefix D:\\path\\to\\rocm] [--revert]
"""
import argparse, glob, os, re, sys

MARK = "LOCAL PATCH (not upstream)"

# (file, block of declarations to guard) -- the guard is inserted before the first line of the
# block and closed after the last, skipped entirely when the file already carries the marker.
BLOCKS = {
    "__clang_hip_cmath.h": [
        ("__DEVICE__ __CONSTEXPR__ bool isgreater(float __x, float __y) {",
         "__DEVICE__ __CONSTEXPR__ bool islessgreater(double __x, double __y) {"),
        ("__DEVICE__ __CONSTEXPR__ bool isunordered(float __x, float __y) {",
         "__DEVICE__ __CONSTEXPR__ bool isunordered(double __x, double __y) {"),
    ],
    "__clang_cuda_math_forward_declares.h": [
        ("__DEVICE__ bool isgreater(double, double);",
         "__DEVICE__ bool isunordered(float, float);"),
    ],
}

def guard(text, first, last, revert):
    """Wrap text from `first` to the end of the declaration starting with `last` in
    #if !defined(_MSC_VER) ... #endif (or unwrap when reverting)."""
    if revert:
        before = text
        text = text.replace("#if !defined(_MSC_VER)\r\n", "").replace("#if !defined(_MSC_VER)\n", "")
        text = text.replace("#endif // !defined(_MSC_VER)\r\n", "").replace("#endif // !defined(_MSC_VER)\n", "")
        return text, text != before
    if first not in text or last not in text:
        return text, False
    i = text.index(first)
    j = text.index(last, i)
    # close after the whole declaration that starts at j
    k = text.index("}\n", j) if "}\n" in text[j:] else text.index(";\n", j)
    if text[k:k + 2] == "}\n":
        k += 2
    else:
        k += 2
    return text[:i] + "#if !defined(_MSC_VER)\r\n" + text[i:k] + "#endif // !defined(_MSC_VER)\r\n" + text[k:], True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default=os.environ.get("HIP_PATH") or os.environ.get("ROCM_PATH"))
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    if not a.prefix:
        sys.exit("give --prefix or set HIP_PATH")

    incs = sorted(glob.glob(os.path.join(a.prefix, "lib", "llvm", "lib", "clang", "*", "include")))
    if not incs:
        sys.exit(f"no clang resource include dir under {a.prefix}")
    total = 0
    for inc in incs:
        for name, blocks in BLOCKS.items():
            path = os.path.join(inc, name)
            if not os.path.exists(path):
                continue
            text = open(path, encoding="utf-8", newline="").read()
            if not a.revert and MARK in text:
                print(f"{path}: already patched")
                continue
            changed = False
            for first, last in blocks:
                text, c = guard(text, first, last, a.revert)
                changed = changed or c
            if changed:
                if not a.revert and MARK not in text:
                    text = ("// " + MARK + "): see scripts/fix-rocm-clang-msvc-cmath.py -- VS 18's\n"
                            "// <cmath> already declares these as clang builtins, and <limits> pulls it in.\n"
                            + text)
                open(path, "w", encoding="utf-8", newline="").write(text)
                print(f"{path}: {'reverted' if a.revert else 'patched'}")
                total += 1
            else:
                print(f"{path}: nothing to do")
    print(f"{total} file(s) changed")
    return 0

if __name__ == "__main__":
    sys.exit(main())
