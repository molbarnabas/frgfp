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
set -e

cd "$(dirname "$0")/.."

export USER_ID="$(id -u)"
export GROUP_ID="$(id -g)"

exec docker compose "$@"
