# spconv-wheels — prebuilt spconv for CUDA 12.8 / 13.0, native Blackwell (RTX 50-series, sm_120) support

Prebuilt [spconv](https://github.com/traveller59/spconv) v2.3.8 +
[cumm](https://github.com/FindDefinition/cumm) v0.8.2 wheels, compiled with
**CUDA 12.8 (`cu128`) and CUDA 13.0 (`cu130`)** for **sm_80, sm_86, sm_89,
sm_90, sm_100 and sm_120** (+ PTX for future architectures).

Upstream's newest wheel is `spconv-cu126`, which has no native Blackwell
kernels (it runs on RTX 50-series only through the driver's PTX JIT, with a
~12 s stall) and drags its own bundled CUDA 12.6 runtime into a process that
already loads torch's. These wheels are the **unmodified upstream v2.3.8
code** — exactly what `pip install spconv-cu126` gives you — compiled for
current CUDA and current GPUs, using the CUDA libraries your torch already
ships, from a build you can reproduce in one command and whose portability is
gated in CI. Other community builds exist; see [Alternatives](#alternatives).

**You want this if** you run spconv on an RTX 5060/5070/5080/5090, an RTX PRO
Blackwell workstation card, or B100/B200/GB200, and you hit either:

- `RuntimeError: ... no kernel image is available for execution on the device`
  (spconv-cu121/cu124 and older builds), or
- a silent **~12-second stall on the first inference** with `spconv-cu126`
  (the driver JIT-compiles CUDA 12.6 PTX for your new GPU; measured 12.4 s →
  0.6 s cold start with these native wheels, identical steady-state speed).

## Install

Linux x86_64, Python 3.10–3.13. **Pick the wheel flavor that matches your
torch build's CUDA version** — the wheels bundle no CUDA libraries; they
resolve `libcudart`/`libnvrtc` from the libraries your torch install loads
into the process, so the sonames must match, and `import torch` must run
before `import spconv`. Consequently *these* wheels do not work with torch
cu121/cu124/cu126 builds (unlike upstream's PyPI wheels, which bundle their
own CUDA runtime) — with such a torch, stay on `spconv-cu126`.

### cu130 — for the default PyPI torch (recommended)

Recent torch from PyPI (≥ 2.11) is a CUDA 13 build, so this just works:

```bash
pip install torch
pip install \
  https://github.com/fafraob/spconv-wheels/releases/download/v2.3.8/cumm_cu130-0.8.2-cp311-cp311-linux_x86_64.whl \
  https://github.com/fafraob/spconv-wheels/releases/download/v2.3.8/spconv_cu130-2.3.8-cp311-cp311-linux_x86_64.whl
```

### cu128 — for torch +cu128 builds (torch 2.7–2.11 from the cu128 index)

```bash
python -m pip install -U pip   # old pips reject current typing_extensions wheels
pip install --only-binary :all: torch --index-url https://download.pytorch.org/whl/cu128
pip install \
  https://github.com/fafraob/spconv-wheels/releases/download/v2.3.8/cumm_cu128-0.8.2-cp311-cp311-linux_x86_64.whl \
  https://github.com/fafraob/spconv-wheels/releases/download/v2.3.8/spconv_cu128-2.3.8-cp311-cp311-linux_x86_64.whl
```

(`--only-binary` keeps pip from falling back to a source distribution of a
torch dependency, which fails to build against the pytorch-only index.)

Swap `cp311` for your Python version — all wheels are on the
[releases page](../../releases).

⚠️ **Install both wheels of one flavor together, exactly as released.** The
two extensions are a pybind11-matched pair: mixing this `spconv-cu1xx` with
the `cumm-cu1xx` wheel from PyPI (or vice versa) fails at import time with
`ImportError: ... type not registered yet?`.

Requires glibc ≥ 2.17 and Ubuntu 22.04+ / Debian 12+ era libstdc++
(GLIBCXX ≥ 3.4.30) — every wheel is install-tested in a clean Debian 12
container in CI before release.

These are unmodified upstream sources at the release tags, except for two
build-level changes (no code changes): spconv's `cumm<0.8.0` dependency pin
is lifted to `<0.9.0` (cumm 0.8.x is 0.7.13 plus CUDA 12.8/Blackwell arch
support — no API change, see the
[cumm changelog](https://github.com/FindDefinition/cumm/blob/main/CHANGELOG.md)),
and for the cu130 flavor cumm's hardcoded `-std=c++14` is raised to `c++17`
(nvcc 13 requires ≥ C++17; this is
[spconv#765](https://github.com/traveller59/spconv/issues/765) — the code is
already C++17-clean, upstream builds it as C++17 on macOS and spconv selects
C++17 itself for CUDA ≥ 11).
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
- All 8 released wheel pairs (cu128 × cu130, Python 3.10–3.13) GPU-verified
  on sm_89 (RTX 4090, Ubuntu 22.04), installed from the release URLs into
  fresh venvs with the install commands above (torch 2.11+cu128 / plain-PyPI
  torch 2.14+cu130): `SubMConv3d` and strided `SparseConv3d` match an fp64
  dense `conv3d` reference to ~4e-07; fp16 forward and fp32 backward clean.
- Every wheel is installed and imported in a clean Debian 12 container in CI
  (with the exact torch install commands above) before a release can publish.

## Alternatives

Other community routes to spconv on Blackwell / recent CUDA, in case these
wheels don't fit your setup:

- [rathaROG/cumm-spconv](https://ratharog.github.io/cumm-spconv/) — a pip
  index of wheels built from rathaROG's **forks** of spconv (2.3.9–2.4.1) and
  cumm (0.8.3–0.9.1), which carry additional fixes on top of upstream.
  cu126/cu128/cu130, Linux **and Windows**, Python 3.9–3.14, manylinux_2_28;
  the cu130 wheel contains native sm_120 kernels (checked with `cuobjdump`).
  Pick it if you need Windows, Python 3.14 or the fork's fixes; pick this repo
  if you want the unmodified upstream 2.3.8 code with an auditable build.
- [RayYoh/spconv-12.8](https://github.com/RayYoh/spconv-12.8) and
  [davidzha712/spconv-blackwell-cu128](https://github.com/davidzha712/spconv-blackwell-cu128)
  — step-by-step guides for building from source for CUDA 12.8 / sm_120.
- [L-Reichardt/spconv-triton](https://github.com/L-Reichardt/spconv-triton) —
  an experimental Triton reimplementation, architecture-independent by
  construction.
- Plain `pip install spconv-cu126` does still work — also next to a cu130
  torch, because the upstream wheel bundles its own `libcudart`/`libnvrtc`
  12.6 (verified on an RTX 4090 with torch 2.14+cu130). On Blackwell it runs
  through the driver's PTX JIT, i.e. with the ~12 s cold start described
  above, but it works.

## Build it yourself

Everything is reproducible with [pixi](https://pixi.sh) — the whole toolchain
(nvcc 12.8 or 13.0, gcc 12, cmake/ninja) comes from conda-forge, no system
CUDA, no docker, no root, no GPU needed at build time:

```bash
pixi run -e py311 bash build.sh           # cu128; or py310 / py312 / py313
pixi run -e py311-cu130 bash build.sh     # cu130 (CUDA version auto-detected)
```

~30–60 min each; wheels land in `dist/`. Or trigger the GitHub Actions
workflow (`.github/workflows/build.yml`), which builds the full
Python × CUDA matrix, install-tests every wheel in a clean container, and
attaches the wheels to a release on tags.

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
6. **CUDA 13 needs one flag: raise cumm's hardcoded `-std=c++14` to `c++17`.**
   nvcc 13 requires ≥ C++17 and the build otherwise dies immediately
   ([spconv#765](https://github.com/traveller59/spconv/issues/765)). The code
   is already C++17-clean — with that one change, all 775 kernels compile
   under CUDA 13.0 unmodified.
7. **Export `PYTHONNOUSERSITE=1` when building in a conda-style env.** User
   site-packages (`~/.local`) take precedence over the env's, so a stale
   cumm/pccm there silently shadows the freshly built one — the spconv build
   then fails against the wrong cumm (e.g. `Unknown CUDA arch (10.0)` from an
   old cumm that predates Blackwell).

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
Not affiliated with or endorsed by the upstream projects — upstream spconv has
had no commits since December 2024, which is the reason these wheels exist.
