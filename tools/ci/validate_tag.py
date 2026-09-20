#!/usr/bin/env python3
"""Validate a release tag before any artefact is built.

The release pipeline is driven entirely by git tags, so this script is the single
place where "is this tag a legitimate release?" is decided.  It fails fast (and
therefore stops the whole workflow) when

* the tag does not have the form ``vX.Y.Z`` or ``vX.Y.ZrcN``,
* a ``vX.Y.ZrcN`` tag is not an ancestor of the ``rc`` branch, or is already part
  of ``main`` (release candidates belong to the ``rc`` branch),
* a final ``vX.Y.Z`` tag is not an ancestor of ``main``,
* the version derived from the tag does not match the version ``setuptools_scm``
  reports for the checked-out commit (a tag that says ``0.2.0`` but builds a
  ``0.1.0`` wheel is the classic silent release bug).

Usage::

    python tools/ci/validate_tag.py --tag v0.2.0rc1 --repo-dir .

When running inside GitHub Actions the derived values are also written to
``$GITHUB_OUTPUT`` (``version``, ``is_prerelease``, ``docs_dir``).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

TAG_RE = re.compile(r"^v(?P<base>\d+\.\d+\.\d+)(?P<pre>rc(?P<rc>\d+))?$")

#: Branch a pre-release tag must live on, and the branch a final release needs.
PRERELEASE_BRANCH = "rc"
RELEASE_BRANCH = "main"


class ValidationError(RuntimeError):
    """Raised for every condition that must abort the release."""


def _git(repo: Path, *args: str, check: bool = True) -> str:
    """Run ``git <args>`` in *repo* and return the stripped stdout."""
    try:
        proc = subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=False
        )
    except OSError as exc:  # pragma: no cover - defensive
        raise ValidationError(f"could not run git {' '.join(args)}: {exc}") from exc
    if check and proc.returncode != 0:
        raise ValidationError(
            f"git {' '.join(args)} failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return proc.stdout.strip()


def _is_ancestor(repo: Path, commit: str, ref: str) -> bool:
    """Return True when *commit* is contained in *ref* (branch or remote ref)."""
    if not _git(repo, "rev-parse", "--verify", "--quiet", ref, check=False):
        raise ValidationError(
            f"reference '{ref}' is not available - fetch it first "
            f"(actions/checkout with fetch-depth: 0 and fetch-tags: true)"
        )
    try:
        return (
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", commit, ref],
                cwd=repo,
                capture_output=True,
                check=False,
            ).returncode
            == 0
        )
    except OSError as exc:  # pragma: no cover - defensive
        raise ValidationError(f"could not run git merge-base: {exc}") from exc


def _scm_version(repo: Path) -> str | None:
    """Return the setuptools_scm version of the working tree, if available."""
    try:
        from setuptools_scm import get_version  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover - setuptools_scm is a dev dependency
        return None
    try:
        return get_version(root=str(repo))
    except Exception:
        return None


def _write_output(name: str, value: str) -> None:
    """Append a ``key=value`` pair to ``$GITHUB_OUTPUT`` when running in Actions."""
    target = os.environ.get("GITHUB_OUTPUT")
    if not target:
        return
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(f"{name}={value}\n")


def _fail(message: str) -> None:
    """Report *message* as a GitHub annotation and exit non-zero."""
    print(f"::error::{message}")
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def parse_tag(tag: str) -> dict[str, object]:
    """Split a release tag into its components.

    Returns
    -------
    dict
        ``base`` (``x.y.z``), ``prerelease`` (``rcN`` or ``None``),
        ``is_prerelease`` and the PEP 440 ``version``.
    """
    match = TAG_RE.match(tag)
    if match is None:
        _fail(
            f"tag '{tag}' is not a release tag: expected vX.Y.Z or vX.Y.ZrcN "
            f"(e.g. v1.0.0 or v1.0.0rc2)"
        )
    base = match.group("base")
    pre = match.group("pre")
    return {
        "base": base,
        "prerelease": pre,
        "is_prerelease": pre is not None,
        "version": f"{base}{pre}" if pre else base,
    }



def _version_key(value: str):
    """Return a sortable PEP 440 key for a version-ish string."""
    try:
        from packaging.version import InvalidVersion, Version
    except Exception:  # pragma: no cover - packaging ships with pip tooling
        return value
    try:
        return Version(value.lstrip("v"))
    except InvalidVersion:
        return Version("0")


def newer_tags(repo: Path, tag: str) -> list[str]:
    """Return release tags that sort strictly above *tag* (informational)."""
    current = _version_key(str(parse_tag(tag)["version"]))
    found = [
        raw
        for raw in _git(repo, "tag", "--list", "v*", check=False).splitlines()
        if re.match(r"^v\d+\.\d+\.\d+(rc\d+)?$", raw) and _version_key(raw) > current
    ]
    return sorted(found, key=_version_key)


def validate(tag: str, repo: Path, branch_check: bool = True) -> dict[str, str]:
    """Run every release-tag check and return the derived metadata."""
    info = parse_tag(tag)
    version = str(info["version"])
    is_prerelease = bool(info["is_prerelease"])

    # The tag must exist, and the built commit must be exactly the tagged one.
    tag_commit = _git(repo, "rev-list", "-n", "1", tag)
    if not tag_commit:
        _fail(f"tag '{tag}' does not exist in {repo}")
    head = _git(repo, "rev-parse", "HEAD")
    if tag_commit != head:
        _fail(
            f"HEAD ({head[:12]}) is not the tagged commit ({tag_commit[:12]}); "
            f"check out the tag before building"
        )
    print(f"tag            : {tag}")
    print(f"tag commit     : {tag_commit[:12]}")
    print(f"version        : {version}")
    print(f"pre-release    : {is_prerelease}")

    # The version the build will produce must equal the tag.
    scm = _scm_version(repo)
    if scm is None:
        print("setuptools_scm : not installed, skipping the version cross-check")
    else:
        local = scm.split("+", 1)[0]
        if local != version:
            _fail(
                f"setuptools_scm derives version '{scm}' from this commit but the "
                f"tag is '{tag}'; the wheel would be published as '{local}'"
            )
        print(f"setuptools_scm : {scm} (matches the tag)")

    if branch_check:
        wanted = PRERELEASE_BRANCH if is_prerelease else RELEASE_BRANCH
        other = RELEASE_BRANCH if is_prerelease else PRERELEASE_BRANCH
        kind = "Release candidates" if is_prerelease else "Final releases"
        if not _is_ancestor(repo, tag_commit, f"origin/{wanted}"):
            _fail(
                f"tag '{tag}' must be reachable from the '{wanted}' branch, "
                f"but it is not; tag the tip of '{wanted}'"
            )
        if _is_ancestor(repo, tag_commit, f"origin/{other}"):
            # An rc tagged on a commit that `main` already contains would be
            # released *behind* the final release, so that is an error.  A final
            # release whose commit is also on `rc` is harmless: it simply means
            # `main` was fast-forwarded from `rc` instead of merged.
            message = (
                f"tag '{tag}' is already part of '{other}'"
                + ("" if is_prerelease else " (main was fast-forwarded from rc?)")
            )
            if is_prerelease:
                _fail(f"{message}; {kind} are tagged on '{wanted}'")
            print(f"::warning::{message}")
        print(f"branch         : reachable from origin/{wanted}, not from origin/{other}")

        stale = newer_tags(repo, tag)
        if stale:
            print(
                f"::warning::newer release tags already exist ({', '.join(stale)}); "
                f"the docs deploy step will refuse to downgrade /latest"
            )

    return {
        "version": version,
        "is_prerelease": "true" if is_prerelease else "false",
        "docs_dir": version,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True, help="release tag, e.g. v1.2.0rc1")
    parser.add_argument(
        "--repo-dir", type=Path, default=Path("."), help="repository checkout to inspect"
    )
    parser.add_argument(
        "--no-branch-check",
        action="store_true",
        help="skip the rc/main reachability checks (emergency re-release only)",
    )
    args = parser.parse_args(argv)

    info = validate(args.tag, args.repo_dir.resolve(), branch_check=not args.no_branch_check)
    for key, value in info.items():
        _write_output(key, str(value))
    print("validation     : OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
