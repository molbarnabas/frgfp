# Releasing FRGfp

Everything in this document is driven by **git tags**. The version is never edited
by hand: `setuptools_scm` derives it from the tag that is being built, and the
release workflow refuses to continue when the tag, the branch and the built
version do not agree.

## Branch and tag model

```
develop ──► rc ──► main
             │       │
             │       └── tag vX.Y.Z      final release     → docs /latest/ + PyPI
             └────────── tag vX.Y.ZrcN   release candidate → docs /X.Y.ZrcN/ + PyPI (pre-release)
```

| Tag | Branch | `/latest/` docs | PyPI |
|---|---|---|---|
| `vX.Y.ZrcN` (any `N`) | `rc` | **never touched** | uploaded as a pre-release |
| `vX.Y.Z` | `main` | updated | uploaded as the current release |

`latest` in the documentation is *always* the newest final release. Release
candidates get their own documentation directory and their own switcher entry
(marked *pre-release*), but they never become `latest`.

Promoting code between branches is part of the release flow, not a prerequisite of
the pipeline:

* release candidates are tagged on `rc` and need **no merge into `main`** — a tag
  push runs `release.yml` from the tagged commit, even while the default branch
  does not contain the workflow yet;
* the final release is tagged on `main`, so `rc` **does** have to be merged into
  `main` first; the tag guard refuses a final tag that is not reachable from
  `origin/main`. That is also the merge which puts the workflows on the default
  branch and thereby enables the manual *Run workflow* buttons.

## Workflows

| Workflow | Trigger | What it does |
|---|---|---|
| `.github/workflows/ci.yml` | PR into `rc`/`main`, push to `rc`/`main` | Builds and tests a wheel for **every** supported platform × CPython combination (Linux x86_64 & aarch64, macOS Apple Silicon, Windows AMD64 × 3.10–3.14 — 20 wheels), builds the sdist and installs/tests it, builds the docs with `-W`. The single aggregate job **`ci`** is the status check to require in branch protection. |
| `.github/workflows/release.yml` | tag `vX.Y.Z` / `vX.Y.ZrcN`, or manual dispatch | Validates the tag, builds/tests all wheels + sdist, publishes the docs to GitHub Pages, creates the GitHub release, and — after a human approves the `pypi` environment — uploads the wheels and sdist to PyPI. |
| `.github/workflows/docs-dev.yml` | push to `develop` | Publishes the development preview to `/dev/`. Delete this file if you do not want it. |
| `.github/workflows/wheel-debug.yml` | manual | Builds and tests **one** wheel for a chosen runner/CPython using the same `pyproject.toml` configuration, so a problem can be iterated on in minutes without re-running the full matrix. It never posts the required `ci` check. |

The tests that gate a PR are exactly the tests that gate a release: cibuildwheel
installs the freshly built wheel into a clean virtual environment and runs

```
python -m pytest --pyargs frgfp -m "not benchmark"
```

from a temporary directory, so the *installed* wheel is what gets tested.

The 19 tests marked `benchmark` (the large 1D/2D grids, the O(8) order-100 case and
the cold solver constructions) are **not part of any workflow**: they report timings
that are only comparable on one machine, and the release pipeline must not depend on
them. They still carry correctness assertions, so run them on demand before a
release if you touched the solver or the large-grid paths:

```bash
python -m pytest --pyargs frgfp -m benchmark -q --benchmark-sort=name
```

Linux wheels are built in the `manylinux_2_28` image (AlmaLinux 8, GCC 14) and
therefore require **glibc ≥ 2.28** (RHEL/Alma/Rocky 8+, Debian 10+, Ubuntu 18.10+).
The older `manylinux2014` image is not usable here: its GCC 10.2 cannot build the
current NumPy, which the test environment has to install.

macOS wheels are labelled **macOS 11+** and bundle their own OpenMP runtime: Homebrew's
`libomp` is compiled for the runner's own macOS (a 26.x bottle on the macOS 26
runners) and delocate refuses to bundle a library whose minimum macOS is newer than
the wheel's tag. `tools/ci/build_libomp.sh` therefore builds LLVM's OpenMP runtime
for the wheel's deployment target during `before-all` (mirroring Homebrew's own
recipe, pinned by version and checksum) and the build links and bundles that copy.
The result is cached between jobs, so the ~2 minute build happens once per runner
and cache key.

On macOS exactly **one** OpenMP runtime may be present in a test process. The wheel
already loads its bundled copy, so the macOS test steps pin numba to its
dependency-free `workqueue` threading layer (`NUMBA_THREADING_LAYER=workqueue`)
instead of letting it `dlopen` a second copy from Homebrew: the OpenMP runtime
treats multiple copies as fatal and aborts the process, which the runners report as
exit code `-6` (SIGABRT, `OMP: Error #15 … already initialized`). The C++ OpenMP
backend is still exercised — only numba's parallel kernels run serially in CI. To
also exercise numba's OpenMP layer there, point `DYLD_FALLBACK_LIBRARY_PATH` at the
`frgfp/.dylibs` directory *inside the installed wheel*, so that numba resolves
`libomp.dylib` to the same file the extension already loaded and dyld keeps a single
image.

### Why there is no Intel (x86_64) macOS wheel

Apple Silicon is the only macOS target: since **numba 0.63** and **llvmlite 0.46**
those projects publish `macosx_*_arm64` wheels only, so a test environment for an
Intel wheel cannot even be created from PyPI — `pip` reaches for the newest numba,
finds an arm64 wheel plus an sdist, and the sdist build fails in LLVM. (The last
PyPI versions with Intel-macOS wheels are numba 0.62.1 and llvmlite 0.45.1,
cp310–cp313, which is why the matrix used to fail on exactly those cells.) Intel
wheel columns are therefore not built; the pipeline that once did is documented in
the git history.

Intel macOS users are not stranded:

* **conda-forge** still builds numba for `osx-64` (0.67.0 for Python 3.10–3.14), so
  `conda install -c conda-forge numba` followed by `pip install frgfp` works; and
* the source distribution installs anywhere the [build
  requirements](../doc/source/installation.rst) are met (CMake, a C++17 compiler,
  `brew install libomp` on macOS).

Windows, Linux x86_64, Linux aarch64 and macOS Apple Silicon remain fully built and
tested. The platform table in `doc/source/installation.rst` is the user-facing
version of this and must be kept in sync when the matrix changes.

## One-time setup

### 1. GitHub repository settings

These are needed **before the first tag**, and none of them requires a merge into
`main`. What *does* depend on the default branch:

* **Tag pushes run the workflow from the tagged commit**, whether or not that
  commit is on the default branch (GitHub's `push` event explicitly covers
  "workflows that are not merged into the default branch"). `rc` can therefore tag
  and publish release candidates with the pipeline living only on `rc`.
* The **Run workflow** button for `workflow_dispatch` workflows (Release, Wheel
  debug, Docs preview) appears only once those files are on the default branch, so
  manual re-runs and rehearsals start working after the first `rc` → `main`
  merge.
* The **final** `vX.Y.Z` tag must be reachable from `origin/main` — the tag guard
  enforces that, which is why promoting `rc` → `main` is required *for the final
  release only*.

The settings themselves:

* **Settings → Actions → General → Workflow permissions**: **Read and write
  permissions**. This is what allows the docs jobs to push `gh-pages` and the
  release job to create the GitHub release; with the restricted setting they fail
  with a 403.
* **Settings → Branches → rulesets** for `main` and `rc`: require a pull request,
  require the status check **`ci`**, require branches to be up to date, and block
  force-pushes. The `ci` check only appears in the list after `ci.yml` has run
  once, so open any pull request first.
* **Settings → Environments** → create **`pypi`** and **`testpypi`**, and add
  yourself under *Required reviewers*. This is the manual approval that makes the
  publication "half automatic": everything is built, tested and documented
  automatically, and the upload waits for one click.

### 2. First-time GitHub Pages setup

`gh-pages` does not exist until the first documentation deployment creates it, and
the Pages branch dropdown only lists branches that already exist. Two orders work;
pick one:

**A. Let the first release candidate create it (no merge, no extra step)**

1. Push the first `vX.Y.ZrcN` tag (see *Cutting a release* below) — its docs job
   creates the orphan `gh-pages` branch containing `index.html`, `switcher.json`,
   `versions.json` and `.nojekyll`, and publishes `/X.Y.ZrcN/`.
2. Refresh *Settings → Pages* → *Build and deployment* → Source: **Deploy from a
   branch**, branch **`gh-pages`**, folder **`/ (root)`**.
3. Check <https://molbarnabas.github.io/frgfp/> — before any final release it
   redirects to the release-candidate directory. From then on every release adds
   its own directory and `/latest/` follows the newest final release; the Pages
   settings never need touching again.

**B. Bootstrap it with the development preview (before any tag)**

*Push to `develop`* (no `main` involvement): `docs-dev.yml` triggers on pushes to
`develop`, creates the branch the same way and publishes `/dev/`, so the site root
redirects to `/dev/` until the first final release exists. Then do steps 2 and 3
above.
Alternatively, once the workflows are on the default branch, run *Actions → **Docs
preview (dev)** → Run workflow* for the same effect without touching `develop` —
remember that this button only exists after the first `rc` → `main` merge.

If you prefer to keep the branch's history under your control, create it from a
throwaway clone instead (your working tree stays untouched):

```bash
git clone --no-checkout git@github.com:molbarnabas/frgfp.git /tmp/frgfp-pages
cd /tmp/frgfp-pages
git checkout --orphan gh-pages
printf '<!doctype html><meta http-equiv="refresh" content="0;url=latest/"><title>FRGfp docs</title>\n' > index.html
: > .nojekyll
git add index.html .nojekyll
git commit -m "docs: initialise gh-pages"
git push origin gh-pages
```

Do **not** create `gh-pages` from the GitHub UI off `main`/`rc`: the UI branches
from an existing ref, so the published branch would contain the whole source tree
(served as part of your documentation site).

### 3. PyPI account and trusted publishing (no tokens)

1. Create an account on <https://pypi.org> and one on <https://test.pypi.org>
   (they are separate), and verify both e-mail addresses.
2. Enable **two-factor authentication** on both (Account settings → *Two-factor
   authentication*). Uploading is impossible without 2FA; save the recovery codes.
3. Register the **pending publisher** (Account settings → *Publishing* →
   *Add a pending publisher*). A project name is only reserved by the first
   upload, so this is the way to authorise the very first release. The fields must
   match the workflow exactly:

   | Field | Value |
   |---|---|
   | PyPI Project Name | `frgfp` |
   | Owner | `molbarnabas` |
   | Repository name | `frgfp` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` (use `testpypi` on TestPyPI) |

   A typo in any of them produces `403 invalid-publisher` on upload.
4. Nothing else is needed: the workflow requests an `id-token` and PyPI mints a
   short-lived upload token. **No API token or repository secret is stored.**

<details>
<summary>Alternative: API token instead of trusted publishing</summary>

1. PyPI → *Account settings → API tokens → Add API token*, scope *Project:
   `frgfp`*, and copy the token once.
2. GitHub → *Settings → Secrets and variables → Actions → New repository secret*
   named `PYPI_API_TOKEN`.
3. In `release.yml`, replace the two publish steps' `with:` body with

   ```yaml
   with:
     packages-dir: dist/
     password: ${{ secrets.PYPI_API_TOKEN }}
   ```

   (adding `repository-url: https://test.pypi.org/legacy/` for the TestPyPI
   step), and drop the `id-token: write` permission.
</details>

## Cutting a release

```bash
# ------------------------------------------------------------ release candidate
git switch rc
git merge --no-ff develop            # bring in the changes to be released
git push
# no merge into main is involved here: the tag push runs release.yml from the
# tagged commit, even while the workflow exists only on rc
git tag -a v0.2.0rc1 -m "0.2.0rc1"   # annotated tag; the message is free text
git push origin v0.2.0rc1
# -> builds and tests 20 wheels + the sdist, publishes /0.2.0rc1/ docs, creates a
#    GitHub pre-release, then waits for your approval to upload to PyPI.

# ---------------------------------------------------------------- final release
git switch main
git merge --no-ff rc                 # required: a final tag must be reachable from
git push                             # origin/main (the guard checks this), and
                                     # this merge is what brings the workflows to
                                     # the default branch
git tag -a v0.2.0 -m "0.2.0"
git push origin v0.2.0
# -> same build/test run, docs also become /latest/, the GitHub release is marked
#    "Latest", and PyPI receives the final artefacts after your approval.
```

To rebuild an existing tag, or to do a dry run without publishing: *Actions →
Release → Run workflow*, enter the tag, choose `target: testpypi` and leave
`publish` off for a plain build (or turn it on to exercise the approval-gated
TestPyPI upload). That button only exists once the workflows are on the default
branch, i.e. after the first `rc` → `main` merge. Until then, release candidates go
straight from a tag — and if a publication is rejected, fix the publisher on PyPI
and use *Re-run failed jobs*, which reuses the artefacts of that run instead of
rebuilding them.

### Rules the pipeline enforces for you

* Only `vX.Y.Z` and `vX.Y.ZrcN` tags trigger a release (alpha/beta-style tags are
  ignored). A candidate must be reachable from `rc`, a final release from `main`.
* A `vX.Y.ZrcN` tag is refused when the final `vX.Y.Z` tag already exists — a
  candidate published behind its own release would only confuse `pip` users.
* A tag whose commit is reachable from both branches is accepted with a warning:
  that is what a fast-forwarded `main` looks like. (Use `git merge --no-ff` when
  promoting `rc` to `main` if you want the branch tips to stay distinct.)
* The version `setuptools_scm` derives for the commit must equal the tag, and
  every wheel filename is checked against it before it becomes an artefact.
* A version can never be uploaded twice, so a broken release needs a new tag
  (for example `v0.2.0rc2`).
* An older tag can never move `/latest/` backwards; the deploy script warns and
  leaves the newer release in place.
* `/latest/` is never written from a pre-release.

## Verifying locally before pushing

```bash
# the exact build CI does on Linux: CIBW_BUILD is what decides the platform
# identifiers - `--only cp312-manylinux_x86_64` would hide musllinux and any
# other identifier, so always check the list first
cibuildwheel --platform linux --print-build-identifiers
CIBW_BUILD=cp312-* cibuildwheel --platform linux .

# the sdist and the wheel
python -m build
python -m pytest --pyargs frgfp -m "not benchmark"

# documentation, with the switcher entry for this version highlighted
DOC_VERSION_MATCH=0.2.0rc1 sphinx-build -b html doc/source /tmp/frgfp-docs

# the Pages assembly (writes files only, does not touch git)
python tools/ci/deploy_docs.py --pages-dir /tmp/pages --html-dir /tmp/frgfp-docs \
    --version 0.2.0rc1 --dry-run
```

To iterate on a platform-specific problem, run *Actions → **Wheel debug (single
target)** → Run workflow* with e.g. `os: macos-latest`, `archs: arm64`,
`python: 310`. It uses the same `pyproject.toml` settings, so fixing it there
fixes it for the real matrix too.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `403 invalid-publisher` on upload | The pending publisher's owner/repo/workflow/environment do not match `release.yml`, or the job did not request `id-token: write`. |
| `tag ... is not a release tag` | Tags must be `vX.Y.Z` or `vX.Y.ZrcN`. |
| `must be reachable from the 'rc' branch` | The tag points at a commit that is not an ancestor of `origin/rc`; tag the branch tip instead. |
| `vX.Y.ZrcN` refused because `vX.Y.Z` already exists | The candidate would be published behind its final release; cut a new version instead. |
| `could not push to the documentation branch` (403) | *Settings → Actions → General → Workflow permissions* must be **Read and write permissions**; also check that no branch ruleset blocks `gh-pages`, and that the `--branch` name matches the Pages setting. |
| Wheel version mismatch | The tag is not on the checked-out commit (`fetch-depth: 0` + `fetch-tags: true` matter), or a stale `build/` directory leaked into the build. |
| macOS build fails on `omp.h` / `libomp` | `brew install libomp` — `CMakeLists.txt` locates the keg by itself, and the workflow installs it. |
| macOS fails in `delocate-wheel` | The repair command echoes delocate's error as a `::error::` annotation, so the message appears on the pull request. macOS labels the wheel macOS 11+ and `tools/ci/build_libomp.sh` builds the OpenMP runtime for exactly that target; if the annotation reports a *different* version, align `MACOSX_DEPLOYMENT_TARGET` in `[tool.cibuildwheel.macos].environment` with `FRGFP_LIBOMP_DEPLOYMENT_TARGET`. |
| macOS test step fails with exit code `-6` | SIGABRT from the OpenMP runtime (`OMP: Error #15 … already initialized`): a second OpenMP runtime was loaded. The macOS steps already pin `NUMBA_THREADING_LAYER=workqueue` for exactly this reason; do not put another `libomp` on the dynamic loader path. |
| Test failures are hard to see | The macOS `test-command` echoes the failing run as a `::error::pytest failed: …` annotation, and the repair step does the same for delocate — both are visible on the pull request without downloading logs. |
| macOS `build_libomp.sh` fails | The script is cached and idempotent: delete `~/frgfp-libomp` and `~/frgfp-libomp-tarball` (or change `hashFiles('tools/ci/build_libomp.sh')` in the cache key) to force a rebuild; the LLVM tarball is checksum-verified, so a truncated download fails loudly. |
| `pip install …musllinux…` fails / numba has no wheels | musllinux builds are skipped on purpose: numba publishes no musllinux wheels, so the test environment would have to compile llvmlite and LLVM. Do not remove `*-musllinux*` from `skip` without a full LLVM toolchain. |
| Only some matrix cells fail, and the log is long | Use *Wheel debug (single target)* with the failing runner/CPython; it runs the same configuration in isolation. |
| `No threading layer could be loaded` on macOS | numba needs an OpenMP runtime at run time; the macOS test environment already adds Homebrew's `libomp` to `DYLD_FALLBACK_LIBRARY_PATH`. |
| `/latest/` did not update | The tag was a release candidate, or it is older than the recorded `latest` (see the warning in the docs job log). |
| `gh-pages` is missing from the Pages branch dropdown | It does not exist yet — see *First-time GitHub Pages setup* above; the first docs deployment creates it (a tag run does that without touching `main`). |
| The *Run workflow* button is missing for Release / Docs preview | `workflow_dispatch` only lists workflows that exist on the **default branch** (`main`); tag-driven releases still work without that merge, and the buttons appear after the first `rc` → `main` promotion. |
| The publish job did not wait for approval | The `pypi`/`testpypi` environment referenced by the workflow did not exist yet, so GitHub created it *without* protection rules. Create it under *Settings → Environments* with yourself as a required reviewer, then re-run the job. |
| The PyPI publish step is skipped | It waits for the `pypi`/`testpypi` environment approval (add yourself as a required reviewer), and it only runs after the wheels, sdist, docs and GitHub release all succeeded. If the docs deploy failed, fix it and use *Re-run failed jobs*. |

