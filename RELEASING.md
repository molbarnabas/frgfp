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

## Workflows

| Workflow | Trigger | What it does |
|---|---|---|
| `.github/workflows/ci.yml` | PR into `rc`/`main`, push to `rc`/`main` | Builds and tests a wheel for **every** platform × CPython combination (Linux x86_64 & aarch64, macOS arm64 & x86_64, Windows AMD64 × 3.10–3.14), builds the sdist and installs/tests it, builds the docs with `-W`. The single aggregate job **`ci`** is the status check to require in branch protection. |
| `.github/workflows/release.yml` | tag `vX.Y.Z` / `vX.Y.ZrcN`, or manual dispatch | Validates the tag, builds/tests all wheels + sdist, runs the benchmark suite, publishes the docs to GitHub Pages, creates the GitHub release, and — after a human approves the `pypi` environment — uploads the wheels and sdist to PyPI. |
| `.github/workflows/docs-dev.yml` | push to `develop` | Publishes the development preview to `/dev/`. Delete this file if you do not want it. |

The tests that gate a PR are exactly the tests that gate a release: cibuildwheel
installs the freshly built wheel into a clean virtual environment and runs

```
python -m pytest --pyargs frgfp -m "not benchmark"
```

from a temporary directory, so the *installed* wheel is what gets tested. The
benchmark suite (the large 1D/2D grids) runs once per release on Linux and its
JSON report is attached to the GitHub release.

Linux wheels are built in the `manylinux_2_28` image (AlmaLinux 8, GCC 14) and
therefore require **glibc ≥ 2.28** (RHEL/Alma/Rocky 8+, Debian 10+, Ubuntu 18.10+).
The older `manylinux2014` image is not usable here: its GCC 10.2 cannot build the
current NumPy, which the test environment has to install.


## One-time setup

### 1. GitHub repository settings

* **Settings → Pages** → *Build and deployment* → Source: **Deploy from a branch**,
  branch **`gh-pages`**, folder **`/ (root)`**. The branch is created
  automatically by the first docs deployment.
* **Settings → Actions → General → Workflow permissions**: *Read repository
  contents*. The docs jobs opt into `contents: write` themselves, which is enough
  to push `gh-pages`.
* **Settings → Branches → rulesets** for `main` and `rc`: require a pull request,
  require the status check **`ci`**, require branches to be up to date, and block
  force-pushes.
* **Settings → Environments** → create **`pypi`** and **`testpypi`**, and add
  yourself under *Required reviewers*. This is the manual approval that makes the
  publication "half automatic": everything is built, tested and documented
  automatically, and the upload waits for one click.

### 2. PyPI account and trusted publishing (no tokens)

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
git tag -a v0.2.0rc1 -m "0.2.0rc1"   # annotated tag; the message is free text
git push origin v0.2.0rc1
# -> builds and tests 25 wheels + the sdist, publishes /0.2.0rc1/ docs, creates a
#    GitHub pre-release, then waits for your approval to upload to PyPI.

# ---------------------------------------------------------------- final release
git switch main
git merge --no-ff rc
git push
git tag -a v0.2.0 -m "0.2.0"
git push origin v0.2.0
# -> same build/test run, docs also become /latest/, the GitHub release is marked
#    "Latest", and PyPI receives the final artefacts after your approval.
```

To rebuild an existing tag, or to do a dry run without publishing: *Actions →
Release → Run workflow*, enter the tag, choose `target: testpypi` and leave
`publish` off for a plain build (or turn it on to exercise the approval-gated
TestPyPI upload).

### Rules the pipeline enforces for you

* Only `vX.Y.Z` and `vX.Y.ZrcN` tags trigger a release (alpha/beta-style tags are
  ignored). The tag must be reachable from `rc` (pre-release) or `main` (final),
  and must not already be part of the other branch.
* The version `setuptools_scm` derives for the commit must equal the tag, and
  every wheel filename is checked against it before it becomes an artefact.
* A version can never be uploaded twice, so a broken release needs a new tag
  (for example `v0.2.0rc2`).
* An older tag can never move `/latest/` backwards; the deploy script warns and
  leaves the newer release in place.
* `/latest/` is never written from a pre-release.

## Verifying locally before pushing

```bash
# build and test exactly the way CI does, on this machine, in the build container
pip install cibuildwheel
cibuildwheel --only cp312-manylinux_x86_64 .

# the sdist and the wheel
python -m build
python -m pytest --pyargs frgfp -m "not benchmark"

# documentation, with the switcher entry for this version highlighted
DOC_VERSION_MATCH=0.2.0rc1 sphinx-build -b html doc/source /tmp/frgfp-docs

# the Pages assembly (writes files only, does not touch git)
python tools/ci/deploy_docs.py --pages-dir /tmp/pages --html-dir /tmp/frgfp-docs \
    --version 0.2.0rc1 --dry-run
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `403 invalid-publisher` on upload | The pending publisher's owner/repo/workflow/environment do not match `release.yml`, or the job did not request `id-token: write`. |
| `tag ... is not a release tag` | Tags must be `vX.Y.Z` or `vX.Y.ZrcN`. |
| `must be reachable from the 'rc' branch` | The tag points at a commit that is not an ancestor of `origin/rc`; tag the branch tip instead. |
| Wheel version mismatch | The tag is not on the checked-out commit (`fetch-depth: 0` + `fetch-tags: true` matter), or a stale `build/` directory leaked into the build. |
| macOS build fails on `omp.h` / `libomp` | `brew install libomp` — `CMakeLists.txt` locates the keg by itself, and the workflow installs it. |
| `No threading layer could be loaded` on macOS | numba needs an OpenMP runtime at run time; the macOS test environment already adds Homebrew's `libomp` to `DYLD_FALLBACK_LIBRARY_PATH`. |
| `/latest/` did not update | The tag was a release candidate, or it is older than the recorded `latest` (see the warning in the docs job log). |
| The docs job cannot push `gh-pages` | Pages is not configured for the `gh-pages` branch, or the job lost its `contents: write` permission. |
| The PyPI publish step is skipped | It waits for the `pypi`/`testpypi` environment approval (add yourself as a required reviewer), and it only runs after the wheels, sdist, docs and GitHub release all succeeded. If the docs deploy failed, fix it and use *Re-run failed jobs*. |

