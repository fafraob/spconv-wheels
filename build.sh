#!/bin/bash
# Build native spconv + cumm wheels for CUDA 12.8 (incl. Blackwell / sm_120).
#
# Run inside a pixi environment of this repo (toolchain fully from conda-forge,
# reproducible on any linux-64 machine, no GPU required):
#
#   pixi run -e py311 bash build.sh      # or py310 / py312 / py313
#
# Wheels land in ./dist. See README.md for the failure modes this script
# works around — the build order and flags below are all load-bearing.

set -euo pipefail

CUMM_TAG="${CUMM_TAG:-v0.8.2}"     # commit 4c77b38d1ab57d5d1c157adddf67dad93f3a446b
SPCONV_TAG="${SPCONV_TAG:-v2.3.8}" # commit 263d6b47425ef843c82f997b12d8b714013d216c

export CUMM_CUDA_VERSION="${CUMM_CUDA_VERSION:-12.8}"
export CUMM_CUDA_ARCH_LIST="${CUMM_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0;10.0;12.0+PTX}"
# AOT build: BOTH flags are required (cumm alone silently builds a pure-python
# JIT wheel without its flag).
export CUMM_DISABLE_JIT="1"
export SPCONV_DISABLE_JIT="1"

# Make nvcc use the conda-forge host compiler, not whatever /usr/bin has.
export CUDA_HOME="$CONDA_PREFIX"
export CUDAHOSTCXX="$CXX"
export NVCC_PREPEND_FLAGS="-ccbin $CXX"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SPCONV_BUILD_DIR:-/tmp/spconv-wheel-build}"
DIST_DIR="$REPO_ROOT/dist"
PYTAG="cp$(python -c 'import sys; print(f"{sys.version_info[0]}{sys.version_info[1]}")')"
mkdir -p "$DIST_DIR"

echo "=== python $PYTAG | CUDA $CUMM_CUDA_VERSION | archs $CUMM_CUDA_ARCH_LIST ==="

# Always build from pristine clones — stale incremental pccm build state
# produces broken .so files.
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
git clone --branch "$CUMM_TAG" --depth 1 https://github.com/FindDefinition/cumm.git
git clone --branch "$SPCONV_TAG" --depth 1 https://github.com/traveller59/spconv.git

# Lift the conservative cumm pin: cumm 0.8.x is 0.7.13 + CUDA 12.8/Blackwell
# support, no API change (see cumm CHANGELOG).
sed -i 's/cumm-cu{}>=0.7.11, <0.8.0/cumm-cu{}>=0.7.11, <0.9.0/' spconv/setup.py
grep -q "cumm-cu{}>=0.7.11, <0.9.0" spconv/setup.py

# Build via setup.py directly, NOT `pip wheel`: under pip's build hooks the
# source dir is not first on sys.path and pccm mis-derives the extension
# namespaces (nested cumm.core_cc.cumm.* -> broken imports).
echo "=== building cumm ($CUMM_TAG) ==="
(cd cumm && python setup.py bdist_wheel)
cp cumm/dist/cumm_cu128-*-"$PYTAG"-*.whl "$DIST_DIR/"

# spconv must be built against the cumm wheel we just made (the two extensions
# are a pybind11-matched pair).
pip install --no-deps --force-reinstall cumm/dist/cumm_cu128-*-"$PYTAG"-*.whl

echo "=== building spconv ($SPCONV_TAG) ==="
(cd spconv && python setup.py bdist_wheel)
cp spconv/dist/spconv_cu128-*-"$PYTAG"-*.whl "$DIST_DIR/"

echo "=== import smoke test ==="
pip install --no-deps --force-reinstall spconv/dist/spconv_cu128-*-"$PYTAG"-*.whl
(cd /tmp && python -c "from cumm.core_cc import tensorview_bind; import spconv.core_cc; print('wheel imports OK')")
# NOTE: this import runs inside the build env, whose modern libstdc++ masks
# symbol-version problems — it catches broken builds (missing kernels, pybind
# mismatch), NOT portability. Portability is enforced by the gate below and
# by the clean-container install test in CI.

echo "=== portability gate (symbol versions) ==="
# The wheels bundle no libstdc++/libc, so the newest symbol version they
# reference sets the oldest system they run on. Enforce the Ubuntu 22.04
# baseline: GLIBCXX <= 3.4.30 (libstdc++ of gcc 12) and GLIBC <= 2.17
# (conda-forge sysroot). gcc 13/14 silently emit GLIBCXX_3.4.31/32 and the
# wheel then fails to import on 22.04 — this is exactly what happened once.
GATE_DIR="$WORK_DIR/gate"
for whl in "$DIST_DIR"/cumm_cu128-*-"$PYTAG"-*.whl "$DIST_DIR"/spconv_cu128-*-"$PYTAG"-*.whl; do
    rm -rf "$GATE_DIR"; mkdir -p "$GATE_DIR"
    python -m zipfile -e "$whl" "$GATE_DIR"
    mapfile -t sos < <(find "$GATE_DIR" -name 'core_cc*.so')
    if [ "${#sos[@]}" -eq 0 ]; then
        echo "FATAL: $(basename "$whl") contains no compiled core_cc extension (JIT-mode wheel?)"
        exit 1
    fi
    for so in "${sos[@]}"; do
        bad=$(readelf --dyn-syms -W "$so" | grep -oE 'GLIBCXX_3\.4\.(3[1-9]|[4-9][0-9])|GLIBC_2\.(1[8-9]|[2-9][0-9])' | sort -u || true)
        if [ -n "$bad" ]; then
            echo "FATAL: $(basename "$whl") requires symbols newer than the Ubuntu 22.04 baseline:"
            echo "$bad"
            echo "Rebuild with gcc 12 (see pixi.toml) — do NOT release this wheel."
            exit 1
        fi
        echo "OK: $(basename "$so") in $(basename "$whl") — max $(readelf --dyn-syms -W "$so" | grep -oE 'GLIBCXX_[0-9.]+' | sort -uV | tail -1), $(readelf --dyn-syms -W "$so" | grep -oE 'GLIBC_2\.[0-9]+' | sort -uV | tail -1)"
    done
done

echo "=== done ==="
ls -lh "$DIST_DIR"/*"$PYTAG"*.whl
