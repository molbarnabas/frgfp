#!/usr/bin/env bash
# Convenience wrapper around `docker compose` for the development container.
#
# It forwards the host UID/GID to the image build so that files created inside
# the container (build outputs, generated documentation, notebook outputs) stay
# owned by the host user.
#
#   ./docker/dev.sh build       # build the image
#   ./docker/dev.sh up -d       # start the container
#   ./docker/dev.sh exec dev bash
#   ./docker/dev.sh down        # stop (add -v to drop the build cache volume)
#
# USER_ID/GROUP_ID already present in the environment win over `id -u`/`id -g`.
# The script refuses to run under sudo (that would forward 0/0) and, when the
# login session predates the `docker` group grant, re-executes itself under
# `sg docker` with the host ids pinned.  See `docker-group-check` for the why.
set -e

self="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$self")/.."

# The ids must be the *host* ids: docker/Dockerfile does
# `groupadd --gid "$GROUP_ID" dev` and `useradd --uid "$USER_ID"`, so wrong
# values surface much later as an obscure build failure.
uid="$(id -u)"
gid="$(id -g)"
docker_gid="$(getent group docker | cut -d: -f3 || true)"

if [ "$uid" -eq 0 ]; then
    echo "docker/dev.sh: refusing to run as root." >&2
    echo "               sudo forwards 'id -u/-g' as 0/0, so the image build fails with" >&2
    echo "               'groupadd --gid 0 dev' (gid 0 is the existing root group), and" >&2
    echo "               bind-mounted outputs end up owned by root." >&2
    exit 1
fi

if [ -n "$docker_gid" ] && [ "$gid" = "$docker_gid" ]; then
    # Primary group is the docker group, i.e. we are inside newgrp/sg.  That is
    # only acceptable when the real login ids were handed to us (which the
    # automatic re-exec below does); otherwise `id -g` would poison GROUP_ID.
    if [ -z "${USER_ID:-}" ] || [ -z "${GROUP_ID:-}" ]; then
        echo "docker/dev.sh: your primary group is the docker group (gid $docker_gid), i.e." >&2
        echo "               the script is running inside 'newgrp docker'/'sg docker'.  GROUP_ID" >&2
        echo "               would then be $docker_gid instead of your login gid, which bakes" >&2
        echo "               the wrong group into the container's 'dev' user.  Run the script" >&2
        echo "               from a plain shell instead - it acquires the group itself." >&2
        exit 1
    fi
fi

if [ -n "$docker_gid" ] && ! id -G | tr ' ' '\n' | grep -qx "$docker_gid"; then
    if printf '%s' ",$(getent group docker | cut -d: -f4)," | grep -q ",$USER,"; then
        echo "docker/dev.sh: this session predates the 'docker' group grant; re-running" >&2
        echo "               under 'sg docker' with the host ids pinned to $uid/$gid." >&2
        echo "               (VS Code: 'Remote-SSH: Kill VS Code Server on Host' and" >&2
        echo "               reconnect to make this unnecessary.)" >&2
        # `sg group cmd ...` is not usable here: sg drops everything after the
        # first word, so the arguments have to travel inside a `-c` command line.
        # shellq() single-quotes a word POSIX-safely for that.
        shellq() { printf "'%s'" "${1//\'/\'\\\'\'}"; }
        reexec="$(shellq "$self")"
        for arg in "$@"; do
            reexec="$reexec $(shellq "$arg")"
        done
        USER_ID="$uid" GROUP_ID="$gid" exec sg docker -c "$reexec"
    fi
    echo "docker/dev.sh: $USER is not in the 'docker' group; run" >&2
    echo "                 sudo usermod -aG docker $USER" >&2
    echo "               and log in again." >&2
    exit 1
fi

export USER_ID="${USER_ID:-$uid}"
export GROUP_ID="${GROUP_ID:-$gid}"

exec docker compose "$@"
