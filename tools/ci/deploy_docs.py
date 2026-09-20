#!/usr/bin/env python3
"""Assemble the versioned GitHub Pages tree for the FRGfp documentation.

The ``gh-pages`` branch holds every published documentation version next to each
other, the way the SciPy docs do::

    /                  redirect to /latest/ , switcher.json , versions.json , .nojekyll
    /latest/           the newest *final* release (never a release candidate)
    /0.2.0/            each final release, pinned
    /0.2.0rc1/         each release candidate, pinned
    /dev/              optional preview of the development branch

The build that fills ``/latest/`` must highlight the ``latest`` switcher entry
while the pinned copy of the same version highlights the version number, so
``--target`` selects which of the two this invocation fills in:

``--target version`` (default)
    Replace ``/<version>/``, refresh ``switcher.json`` / ``versions.json`` and the
    root redirect.  ``latest`` is only re-pointed when *version* is the newest
    final release, so re-tagging an older release can never move ``/latest/``
    backwards (a ``::warning::`` is emitted instead).

``--target latest``
    Replace ``/latest/`` with the given build.  Refused unless the version is the
    newest final release recorded in ``versions.json`` (``--force`` overrides).

Usage::

    python tools/ci/deploy_docs.py --pages-dir _pages --html-dir doc/build/html \\
        --version 0.2.0 --base-url https://molbarnabas.github.io/frgfp/
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

#: Directory name of the "newest final release" alias and of the dev preview.
LATEST_DIR = "latest"
DEV_DIR = "dev"

#: Matches the directory names that count as released versions.
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(rc\d+)?$")

SWITCHER_FILE = "switcher.json"
VERSIONS_FILE = "versions.json"
INDEX_FILE = "index.html"

#: Printed when the push to the documentation branch is rejected.  A read-only
#: workflow token is by far the most common cause.
PUSH_HINT = (
    "could not push to the documentation branch. Check that (1) Settings -> "
    "Actions -> General -> Workflow permissions is set to 'Read and write "
    "permissions' so that the workflow token may push, (2) the branch is not "
    "protected by a ruleset that forbids pushes, and (3) the branch name passed "
    "to --branch matches the branch configured under Settings -> Pages "
    "('Deploy from a branch')."
)


def _fail(message: str) -> None:
    """Report *message* as a GitHub annotation and exit non-zero."""
    print(f"::error::{message}")
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def _version_key(value: str):
    """Return a sortable PEP 440 key for a version string."""
    from packaging.version import InvalidVersion, Version

    try:
        return Version(value)
    except InvalidVersion:
        _fail(f"'{value}' is not a valid PEP 440 version")


def is_prerelease(version: str) -> bool:
    """Return True when *version* is a release candidate (or any pre-release)."""
    return bool(_version_key(version).is_prerelease)


def _git(pages_dir: Path, *args: str) -> str:
    """Run ``git <args>`` inside the pages work tree."""
    proc = subprocess.run(
        ["git", *args], cwd=pages_dir, capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        _fail(f"git {' '.join(args)} failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return proc.stdout.strip()


def _load_json(path: Path, default):
    """Load *path* as JSON, returning *default* when it is missing or invalid."""
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"::warning::{path.name} is not valid JSON, regenerating it")
        return default


def _write_json(path: Path, payload) -> None:
    """Write *payload* as pretty-printed JSON."""
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _replace_tree(source: Path, target: Path) -> None:
    """Replace *target* with a copy of *source*, removing stale files."""
    if not source.is_dir():
        _fail(f"built documentation directory '{source}' does not exist")
    if not (source / INDEX_FILE).is_file():
        _fail(f"'{source}' does not look like a Sphinx HTML build (no {INDEX_FILE})")
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def discover_versions(pages_dir: Path) -> list[str]:
    """Return every version directory currently published on the site."""
    return sorted(
        (
            entry.name
            for entry in pages_dir.iterdir()
            if entry.is_dir() and VERSION_RE.match(entry.name)
        ),
        key=_version_key,
        reverse=True,
    )


def switcher_entries(
    base_url: str, versions: list[str], latest: str | None, dev: bool
) -> list[dict]:
    """Build the ``switcher.json`` payload consumed by pydata-sphinx-theme.

    The ``version`` field of an entry is what ``conf.py`` passes as
    ``version_match``, so it is exactly ``latest`` for the ``/latest/`` build,
    exactly the version string for a pinned build and exactly ``dev`` for the
    development preview.
    """
    finals = [v for v in versions if not is_prerelease(v)]
    pres = [v for v in versions if is_prerelease(v)]
    entries: list[dict] = []
    if latest:
        entries.append(
            {
                "name": f"latest (v{latest})",
                "version": LATEST_DIR,
                "url": f"{base_url}{LATEST_DIR}/",
                "preferred": True,
            }
        )
    entries.extend(
        {"name": f"v{version}", "version": version, "url": f"{base_url}{version}/"}
        for version in finals
    )
    entries.extend(
        {
            "name": f"v{version} (pre-release)",
            "version": version,
            "url": f"{base_url}{version}/",
        }
        for version in pres
    )
    if dev:
        entries.append(
            {"name": "dev", "version": DEV_DIR, "url": f"{base_url}{DEV_DIR}/"}
        )
    return entries


def update_metadata(pages_dir: Path, base_url: str, version: str) -> dict:
    """Refresh ``versions.json`` and ``switcher.json`` after a version deploy."""
    versions_file = pages_dir / VERSIONS_FILE
    previous = _load_json(versions_file, {})
    versions = discover_versions(pages_dir)
    dev = (pages_dir / DEV_DIR).is_dir()

    finals = [v for v in versions if not is_prerelease(v)]
    newest_final = finals[0] if finals else None
    previous_latest = previous.get("latest")
    if previous_latest and previous_latest not in finals:
        previous_latest = None

    if (
        previous_latest
        and version != DEV_DIR
        and _version_key(previous_latest) > _version_key(version)
    ):
        print(
            f"::warning::{version} is older than the current latest "
            f"({previous_latest}); /latest/ stays at {previous_latest}"
        )
        latest = previous_latest
    else:
        latest = newest_final

    metadata = {
        "latest": latest,
        "versions": versions,
        "prereleases": [v for v in versions if is_prerelease(v)],
        "dev": dev,
        "base_url": base_url,
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    _write_json(versions_file, metadata)
    _write_json(
        pages_dir / SWITCHER_FILE, switcher_entries(base_url, versions, latest, dev)
    )
    return metadata


def write_root_index(pages_dir: Path, base_url: str, metadata: dict) -> None:
    """Write the site root: a redirect to the newest docs plus a version list."""
    versions = metadata.get("versions") or []
    latest = metadata.get("latest")
    dev = bool(metadata.get("dev"))
    # Prefer the /latest/ alias so the bookmarked URL always follows the newest
    # final release; fall back to the newest published docs, then to the
    # development preview on a site where nothing has been released yet.
    if latest:
        target, label = f"{LATEST_DIR}/", f"latest (v{latest})"
    elif versions:
        target, label = f"{versions[0]}/", versions[0]
    elif dev:
        target, label = f"{DEV_DIR}/", "development"
    else:
        _fail("no documentation version is published yet, nothing to redirect to")
    links = "\n".join(
        f'      <li><a href="{base_url}{version}/">v{version}'
        f'{" (pre-release)" if is_prerelease(version) else ""}</a></li>'
        for version in versions
    )
    if metadata.get("dev"):
        links += f'\n      <li><a href="{base_url}{DEV_DIR}/">dev</a></li>'
    redirect = f"{base_url}{target}"
    html = f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="0; url={redirect}">
    <link rel="canonical" href="{redirect}">
    <title>FRGfp documentation</title>
  </head>
  <body>
    <p>Redirecting to the <a href="{redirect}">{label} FRGfp documentation</a>&hellip;</p>
    <h1>FRGfp documentation versions</h1>
    <ul>
{links}
    </ul>
  </body>
</html>
"""
    (pages_dir / INDEX_FILE).write_text(html, encoding="utf-8")
    # Keep GitHub Pages from running Jekyll over the generated site.
    (pages_dir / ".nojekyll").write_text("", encoding="utf-8")



def commit_and_push(
    pages_dir: Path,
    message: str,
    *,
    branch: str,
    git_name: str,
    git_email: str,
    dry_run: bool,
    no_commit: bool,
    no_push: bool,
) -> None:
    """Record the freshly assembled tree in git and push it to *branch*."""
    if dry_run or no_commit:
        print("git            : skipped (dry run / --no-commit)")
        return
    _git(pages_dir, "config", "user.name", git_name)
    _git(pages_dir, "config", "user.email", git_email)
    _git(pages_dir, "add", "-A")
    if not _git(pages_dir, "status", "--porcelain"):
        print("git            : nothing to commit")
        return
    _git(pages_dir, "commit", "-m", message)
    print(f"git            : committed {_git(pages_dir, 'rev-parse', '--short', 'HEAD')}")
    if no_push:
        print("git            : push skipped (--no-push)")
        return
    push = subprocess.run(
        ["git", "push", "origin", f"HEAD:{branch}"],
        cwd=pages_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if push.returncode != 0:
        print(push.stderr.strip() or push.stdout.strip())
        _fail(PUSH_HINT)
    print(f"git            : pushed to origin/{branch}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pages-dir", type=Path, required=True, help="gh-pages checkout")
    parser.add_argument("--html-dir", type=Path, required=True, help="built Sphinx HTML")
    parser.add_argument(
        "--version", required=True, help="version being deployed, or 'dev'"
    )
    parser.add_argument(
        "--base-url",
        default="https://molbarnabas.github.io/frgfp/",
        help="public URL of the docs root (must end with '/')",
    )
    parser.add_argument(
        "--target",
        choices=("version", "latest"),
        default="version",
        help="'version' fills /<version>/, 'latest' fills the /latest/ alias",
    )
    parser.add_argument(
        "--force", action="store_true", help="skip the newest-final-release guard"
    )
    parser.add_argument("--dry-run", action="store_true", help="write files, skip git")
    parser.add_argument("--no-commit", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--branch", default="gh-pages", help="branch to push to")
    parser.add_argument("--git-name", default="github-actions[bot]")
    parser.add_argument(
        "--git-email", default="41898282+github-actions[bot]@users.noreply.github.com"
    )
    parser.add_argument("--message", default=None, help="commit message override")
    args = parser.parse_args(argv)

    if not args.base_url.endswith("/"):
        _fail("--base-url must end with '/'")
    pages_dir = args.pages_dir.expanduser().resolve()
    html_dir = args.html_dir.expanduser().resolve()
    if not pages_dir.is_dir():
        _fail(f"pages work tree '{pages_dir}' does not exist")

    version = args.version
    prerelease = version != DEV_DIR and is_prerelease(version)

    if args.target == "latest":
        if version == DEV_DIR or prerelease:
            _fail(
                "--target latest refuses pre-releases and dev builds: /latest/ is "
                "always the newest final release"
            )
        recorded = _load_json(pages_dir / VERSIONS_FILE, {}).get("latest")
        if recorded != version and not args.force:
            _fail(
                f"--target latest needs '{version}' to be the newest final release "
                f"recorded in {VERSIONS_FILE} (found {recorded!r}); run "
                f"--target version for this release first"
            )
        _replace_tree(html_dir, pages_dir / LATEST_DIR)
        print(f"pages          : /{LATEST_DIR}/ <- {html_dir}")
    else:
        directory = DEV_DIR if version == DEV_DIR else version
        _replace_tree(html_dir, pages_dir / directory)
        print(f"pages          : /{directory}/ <- {html_dir}")
        metadata = update_metadata(pages_dir, args.base_url, version)
        write_root_index(pages_dir, args.base_url, metadata)
        print(
            f"pages          : versions={metadata['versions']} "
            f"latest={metadata['latest']}"
        )

    commit_and_push(
        pages_dir,
        args.message or f"docs: publish {version}",
        branch=args.branch,
        git_name=args.git_name,
        git_email=args.git_email,
        dry_run=args.dry_run,
        no_commit=args.no_commit,
        no_push=args.no_push,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

