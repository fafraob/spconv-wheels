# spconv CUDA 12.8 wheels — native Blackwell (RTX 50-series, sm_120) support

Prebuilt [spconv](https://github.com/traveller59/spconv) v2.3.8 +
[cumm](https://github.com/FindDefinition/cumm) v0.8.2 wheels, compiled with
CUDA 12.8 for **sm_80, sm_86, sm_89, sm_90, sm_100 and sm_120** (+ PTX for
future architectures) — because no published spconv wheel supports Blackwell
GPUs natively.

**You want this if** you run spconv on an RTX 5060/5070/5080/5090, an RTX PRO
Blackwell workstation card, or B100/B200/GB200, and you hit either:

- `RuntimeError: ... no kernel image is available for execution on the device`
  (spconv-cu121/cu124 and older builds), or
- a silent **~12-second stall on the first inference** with `spconv-cu126`
  (the driver JIT-compiles CUDA 12.6 PTX for your new GPU; measured 12.4 s →
  0.6 s cold start with these native wheels, identical steady-state speed).

## Install

Linux x86_64, Python 3.10–3.13. Pick the wheels for your Python version from
the [releases page](../../releases), then (example for cp311):

```bash
pip install \
  https://github.com/fafraob/spconv-cu128-wheels/releases/download/v2.3.8/cumm_cu128-0.8.2-cp311-cp311-linux_x86_64.whl \
  https://github.com/fafraob/spconv-cu128-wheels/releases/download/v2.3.8/spconv_cu128-2.3.8-cp311-cp311-linux_x86_64.whl
```

⚠️ **Install both wheels together, exactly as released.** The two extensions
are a pybind11-matched pair: mixing this `spconv-cu128` with the `cumm-cu128`
wheel from PyPI (or vice versa) fails at import time with
`ImportError: ... type not registered yet?`.

**A torch cu128 build (torch ≥ 2.7 + cu128) is required**, and `import torch`
must run before `import spconv`. ⚠️ A plain `pip install torch` no longer
gives you that — PyPI's default torch is now a CUDA 13 (cu130) build, which
cannot load these wheels. Install torch explicitly from the cu128 index:

```bash
pip install --only-binary :all: torch --index-url https://download.pytorch.org/whl/cu128
```

(`--only-binary` guards against the cu128 index occasionally serving a
source distribution for one of torch's pure-python dependencies, which then
fails to build against that index.)
 The wheels do not link libtorch and do not
bundle CUDA libraries; they resolve `libcudart`/`libnvrtc` **12.8** from the
libraries your torch cu128 install loads into the process. Older torch CUDA
builds (cu121/cu124/cu126) ship different library sonames and the import
fails. For Blackwell you need torch cu128 anyway. Requires glibc ≥ 2.17 and
Ubuntu 22.04+ / Debian 12+ era libstdc++ (GLIBCXX ≥ 3.4.30) — every wheel is
install-tested in a clean Debian 12 container in CI before release.

These are unmodified upstream sources at the release tags, except for one
build-metadata change: spconv's `cumm<0.8.0` dependency pin is lifted to
`<0.9.0` (cumm 0.8.x is 0.7.13 plus CUDA 12.8/Blackwell arch support — no API
change, see the [cumm changelog](https://github.com/FindDefinition/cumm/blob/main/CHANGELOG.md)).
Wheels are intentionally **not** published to PyPI — the `spconv-cu1xx` /
`cumm-cu1xx` names belong to the upstream maintainers.

## Verified

- Point-level output comparison on a real LiDAR segmentation workload
  (sparse-conv U-Net, ~90M points across 248 scans): fp32 predictions agree
  with a reference Ada (sm_89) stack on 100.0000% of points; fp16 on 99.9996%
  (boundary-point noise, task metrics unchanged).
- Cold start (first inference incl. kernel selection): **~0.6 s** vs ~12.4 s
  with spconv-cu126's PTX JIT on sm_120. Steady-state throughput is the same —
  the kernel algorithms are identical, only the JIT compile is removed.

## Build it yourself

Everything is reproducible with [pixi](https://pixi.sh) — the whole toolchain
(nvcc 12.8, gcc 14, cmake/ninja) comes from conda-forge, no system CUDA, no
docker, no root, no GPU needed at build time:

```bash
pixi run -e py311 bash build.sh     # ~30-60 min; wheels land in dist/
```

Or trigger the GitHub Actions workflow (`.github/workflows/build.yml`), which
builds the full Python matrix and attaches the wheels to a release on tags.

Arch list, CUDA version and source tags are overridable via env vars
(`CUMM_CUDA_ARCH_LIST`, `CUMM_CUDA_VERSION`, `CUMM_TAG`, `SPCONV_TAG`) — see
`build.sh`.

## The traps (why this repo exists)

Building AOT spconv wheels is documented nowhere and has several silent
failure modes, hit and solved here so you don't have to:

1. **`CUMM_DISABLE_JIT=1` is required in addition to `SPCONV_DISABLE_JIT=1`.**
   Without it, cumm quietly builds a pure-Python (JIT-mode) wheel with no
   compiled kernels, and the spconv build fails later with a confusing
   `TENSORVIEW_INCLUDE_PATH` assertion.
2. **Build with `python setup.py bdist_wheel`, not `pip wheel`.** Under pip's
   PEP-517 build hooks the source directory is not first on `sys.path`, and
   pccm then mis-derives the extension namespaces: the resulting `core_cc`
   gets a nested module layout (`cumm.core_cc.cumm.*`) and fails with
   `ImportError: cannot import name 'tensorview_bind'`.
3. **Both wheels must be built in one run, in one environment.** spconv's
   extension resolves pybind11 types registered by cumm's extension; wheels
   built with different pybind11 internals fail with
   `arg(): could not convert default argument 'workspace: tv::Tensor' ...
   (type not registered yet?)`.
4. **Always build from pristine clones.** Stale incremental pccm/ccimport
   build state from a previous run (or a different environment) produces
   broken `.so` files that import but miss symbols.
5. **Pin the host compiler to gcc 12, and never trust an in-build-env import
   test for portability.** The wheels bundle no libstdc++, so the newest
   GLIBCXX symbol they reference sets the oldest system they run on: gcc 13/14
   emit `GLIBCXX_3.4.31/32` references and the wheel then fails to import on
   Ubuntu 22.04 (`GLIBCXX_3.4.32 not found`) — while importing fine inside the
   build env, whose modern conda libstdc++ masks the problem. `build.sh` gates
   every wheel on `readelf` symbol versions, and CI install-tests each wheel
   in a clean Debian 12 container before a release can be published.

Bonus, unrelated to building but easy to hit at runtime:

- **fp16 inference: use `model.half()`, not autocast.** spconv's kernel tuner
  crashes on autocast's mixed dtypes
  (`ConvTunerSimple ... can't find suitable algorithm for 0`).
- **Never zero-pad a layer's *output* channels** (e.g. 6 → 8 "for tensor-core
  alignment"): spconv 2.3.8 silently computes wrong results downstream.
  Zero-padding *input* channels (with correspondingly zero-padded weights) is
  safe and can move unaligned stems onto tensor-core kernels.

## Blackwell notes

Workstation/consumer Blackwell (sm_120) keeps the Ada-style `mma.sync` tensor
core pipeline for fp16 — the 5th-generation `tcgen05` instructions are
datacenter (sm_100a) only. These wheels therefore already use the fastest
fp16 instructions available on RTX 50-series hardware; native compilation
buys cold-start time and forward reproducibility, not steady-state throughput
over the PTX-JIT path.

## License & attribution

[spconv](https://github.com/traveller59/spconv) and
[cumm](https://github.com/FindDefinition/cumm) are © their authors, licensed
under Apache-2.0. This repo (build scripts, CI, docs) is Apache-2.0 as well.
Not affiliated with or endorsed by the upstream projects — upstream has been
inactive since late 2024, which is the reason these wheels exist.
