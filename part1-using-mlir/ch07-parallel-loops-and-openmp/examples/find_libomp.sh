# Locate an OpenMP runtime and export the flags the chapter's build scripts
# need. Sourced (not executed) by examples/*/build.sh.
#
#   OMP_CFLAGS   compile flags so `#include <omp.h>` works
#   OMP_LDFLAGS  link flags for -lomp, with an rpath so the binary finds the
#                library at run time without DYLD_LIBRARY_PATH/LD_LIBRARY_PATH
#
# Search order: $LIBOMP_PREFIX if you set it; Homebrew's libomp (macOS);
# then the usual Linux locations (Debian/Ubuntu `libomp-dev`, Fedora
# `libomp-devel`), where clang finds libomp by itself.
OMP_CFLAGS=""
OMP_LDFLAGS=""
if [ -z "${LIBOMP_PREFIX:-}" ] && command -v brew &> /dev/null; then
    LIBOMP_PREFIX=$(brew --prefix libomp 2> /dev/null || true)
fi
if [ -n "${LIBOMP_PREFIX:-}" ] && [ -e "$LIBOMP_PREFIX/lib/libomp.dylib" -o -e "$LIBOMP_PREFIX/lib/libomp.so" ]; then
    OMP_CFLAGS="-I$LIBOMP_PREFIX/include"
    OMP_LDFLAGS="-L$LIBOMP_PREFIX/lib -lomp -Wl,-rpath,$LIBOMP_PREFIX/lib"
elif [ "$(uname)" = Linux ]; then
    # Debian/Ubuntu install libomp under /usr/lib/llvm-<N>/lib with a
    # libomp.so symlink on the default search path; plain -lomp works.
    OMP_LDFLAGS="-lomp"
else
    echo "error: no OpenMP runtime found. Install one with 'brew install libomp'" >&2
    echo "       (macOS) or 'apt install libomp-dev' (Linux), or set LIBOMP_PREFIX." >&2
    exit 1
fi
# Homebrew's libomp is built for the newest macOS; silence the linker's
# harmless "built for newer version" warning when clang targets an older one.
if [ "$(uname)" = Darwin ]; then
    OMP_LDFLAGS="$OMP_LDFLAGS -Wl,-w"
fi
export OMP_CFLAGS OMP_LDFLAGS
