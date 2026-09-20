#!/usr/bin/env bash
#
# Build LLVM's OpenMP runtime (libomp) for the macOS wheels.
#
# Why this exists
# ---------------
# On macOS the extension links a *dynamic* OpenMP runtime.  The Homebrew `libomp`
# that the build would otherwise pick up is compiled for the runner's own macOS
# version - on the macOS 26 runners that is a "tahoe" bottle - while the wheels we
# publish are labelled macOS 11.  Since delocate 0.11 the wheel repair step refuses
# to bundle a library whose minimum macOS version is newer than the wheel's own tag
# ("library dependencies do not satisfy target MacOS"), so those wheels cannot be
# produced at all.
#
# Building the runtime ourselves for a fixed, older deployment target solves it:
# the wheel keeps its `macosx_11_0_*` tag, bundles our libomp and stays
# self-contained.  The command line mirrors Homebrew's own libomp formula, so the
# recipe is the same one that is known to work on macOS.
#
# The result is kept in $FRGFP_LIBOMP_PREFIX (wired into actions/cache) and a stamp
# file makes this script a no-op when nothing changed.
#
# Environment:
#   FRGFP_LIBOMP_PREFIX            install prefix   (default ~/frgfp-libomp)
#   FRGFP_LIBOMP_DEPLOYMENT_TARGET libomp's minimum macOS (default 11.0)
#   FRGFP_LIBOMP_TARBALL_DIR       downloaded tarball (default ~/frgfp-libomp-tarball)
#   FRGFP_LIBOMP_WORK_DIR          extraction/build scratch (default ~/frgfp-libomp-work)
#   LLVM_VERSION                   LLVM release to build (default 23.1.1)
#
# Only the install prefix and the tarball are worth caching (see the cache steps
# in the workflows); the extracted sources and the build directory are scratch.
set -euo pipefail

LLVM_VERSION="${LLVM_VERSION:-23.1.1}"
LLVM_SHA256="ebe9be46fe8756d58c5b198ffad0fa2a766257add81a4dc52179bfacc7888ee6"

#: Must be <= the MACOSX_DEPLOYMENT_TARGET used for the wheels (see pyproject.toml).
DEPLOYMENT_TARGET="${FRGFP_LIBOMP_DEPLOYMENT_TARGET:-11.0}"
PREFIX="${FRGFP_LIBOMP_PREFIX:-$HOME/frgfp-libomp}"
TARBALL_DIR="${FRGFP_LIBOMP_TARBALL_DIR:-$HOME/frgfp-libomp-tarball}"
WORK_DIR="${FRGFP_LIBOMP_WORK_DIR:-$HOME/frgfp-libomp-work}"
STAMP="$PREFIX/.frgfp-stamp"

if [ "$(uname -s)" != "Darwin" ]; then
    echo "build_libomp: not macOS, skipping"
    exit 0
fi

stamp_value="$LLVM_VERSION-$DEPLOYMENT_TARGET-$(uname -m)"
if [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$stamp_value" ] && [ -f "$PREFIX/lib/libomp.dylib" ]; then
    echo "build_libomp: reusing $PREFIX ($stamp_value)"
else
    mkdir -p "$TARBALL_DIR"
    tarball="$TARBALL_DIR/llvm-project-$LLVM_VERSION.src.tar.xz"
    if [ ! -f "$tarball" ]; then
        echo "build_libomp: downloading LLVM $LLVM_VERSION sources"
        curl -fsSL -o "$tarball" \
            "https://github.com/llvm/llvm-project/releases/download/llvmorg-$LLVM_VERSION/llvm-project-$LLVM_VERSION.src.tar.xz"
    fi
    # Fail loudly if the download was truncated or swapped out.
    echo "$LLVM_SHA256  $tarball" | shasum -a 256 -c -

    src_root="$WORK_DIR/src"
    build_dir="$WORK_DIR/build"
    rm -rf "$WORK_DIR" "$PREFIX"
    mkdir -p "$WORK_DIR"
    tar -xf "$tarball" -C "$WORK_DIR"
    mv "$WORK_DIR/llvm-project-$LLVM_VERSION.src" "$src_root"

    # `runtimes` is the entry point for building a single runtime out of the
    # monorepo without building all of LLVM (the same source directory Homebrew
    # configures).
    echo "build_libomp: configuring OpenMP $LLVM_VERSION for macOS $DEPLOYMENT_TARGET ($(uname -m))"
    cmake -S "$src_root/runtimes" -B "$build_dir" \
        -DLLVM_ENABLE_RUNTIMES=openmp \
        -DOPENMP_ENABLE_OMPT_TOOLS=OFF \
        -DLIBOMP_INSTALL_ALIASES=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_OSX_DEPLOYMENT_TARGET="$DEPLOYMENT_TARGET" \
        -DCMAKE_INSTALL_PREFIX="$PREFIX"

    cmake --build "$build_dir" --target install -j"$(sysctl -n hw.ncpu)"
    rm -rf "$WORK_DIR"
    printf '%s' "$stamp_value" > "$STAMP"
fi

echo "build_libomp: installed into $PREFIX"
echo "build_libomp: minimum macOS / arch of the runtime:"
otool -l "$PREFIX/lib/libomp.dylib" \
    | awk '/LC_BUILD_VERSION/ {seen=1} seen && /minos|sdk|platform/ {print "  " $0; count++} count==3 {exit}'
