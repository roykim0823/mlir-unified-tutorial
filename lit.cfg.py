import os
import lit.formats
from lit.llvm import llvm_config

config.name = "MLIRUnifiedTutorial"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".mlir"]
config.test_source_root = os.path.join(os.path.dirname(__file__), "test")
config.test_exec_root = os.path.join(config.tutorial_obj_root, "test")

# Chapters whose example/exercise .mlir files carry their own RUN lines are
# pulled into the suite through symlinks under test/ (e.g.
# test/part5/ch22-linalg-transform-dialect -> the chapter folder).
# Exercise skeletons (TODO versions of the solutions, kept in
# exercises/skeletons/) are reading material, never tests.
config.excludes = ["skeletons"]

config.substitutions.append(("%project_source_dir", config.tutorial_src_root))

if getattr(config, "enable_solver", "OFF") == "ON":
    config.available_features.add("solver")

# Part 6's BLAS tests link against whatever provides the CBLAS interface:
# Apple's Accelerate framework on macOS, OpenBLAS elsewhere.
import platform

if platform.system() == "Darwin":
    config.substitutions.append(("%blas_link_flags", "-framework Accelerate"))
else:
    config.substitutions.append(("%blas_link_flags", "-lopenblas"))

# Chapter 7's OpenMP runtime tests need libomp, which the wheel does not
# ship. Look for Homebrew's keg (macOS) or a system libomp (Linux); tests
# that run OpenMP code use `REQUIRES: libomp` and `%omp_link_flags`.
import ctypes.util
import shutil
import subprocess

_omp_flags = None
if shutil.which("brew"):
    _prefix = subprocess.run(["brew", "--prefix", "libomp"], capture_output=True,
                             text=True).stdout.strip()
    if _prefix and os.path.exists(os.path.join(_prefix, "lib", "libomp.dylib")):
        _omp_flags = f"-L{_prefix}/lib -lomp -Wl,-rpath,{_prefix}/lib -Wl,-w"
if _omp_flags is None and platform.system() == "Linux" and ctypes.util.find_library("omp"):
    _omp_flags = "-lomp"
if _omp_flags is not None:
    config.available_features.add("libomp")
    config.substitutions.append(("%omp_link_flags", _omp_flags))

# The wheel's shared libraries (libmlir_c_runner_utils & friends) live next
# to its tools directory.
config.substitutions.append(
    ("%mlir_libs_dir",
     os.path.normpath(os.path.join(config.llvm_tools_dir, "..", "lib")))
)

# Add tools directories to PATH
llvm_config.with_environment("PATH", config.tutorial_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
