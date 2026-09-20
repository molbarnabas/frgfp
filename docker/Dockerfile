# Development image for FRGfp.
#
# Provides a Linux (Debian bookworm) toolchain with every dependency needed to
# build the C++ extensions, run the tests and generate the documentation.
#
# The project dependencies are baked in at build time; the package itself is
# installed in editable mode from the bind-mounted repository on first start
# (see docker/entrypoint.sh), so source edits are picked up immediately.

FROM python:3.10-slim-bookworm

ARG USER_ID=1000
ARG GROUP_ID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# C++17 toolchain, CMake (>= 3.15), Ninja, Eigen3, OpenMP (via GCC) and git
# (required by setuptools_scm to derive the version from the git tags).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        libeigen3-dev \
        git \
        make \
        less \
        vim \
    && rm -rf /var/lib/apt/lists/*

# Non-root user matching the host UID/GID, so bind-mounted files keep their
# host ownership (no root-owned artifacts, git works without complaints).
RUN groupadd --gid "${GROUP_ID}" dev \
    && useradd --uid "${USER_ID}" --gid "${GROUP_ID}" --create-home --shell /bin/bash dev

# A virtual environment owned by `dev`: editable installs must write into
# site-packages, which the system prefix does not allow for a normal user.
RUN python -m venv /opt/venv && chown -R dev:dev /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
ENV VIRTUAL_ENV="/opt/venv"
RUN pip install --upgrade pip

# Python dependencies first (better layer caching). requirements.txt mirrors the
# `dev` extra of pyproject.toml: runtime, build backend, tests and documentation.
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Convenience extras: plotting for examples/, live docs and a notebook server.
RUN pip install matplotlib sphinx-autobuild jupyterlab

# The repository is bind-mounted here; tell git it is safe even if the container
# user does not happen to match the owner of the mounted files.
RUN git config --system --add safe.directory /workspace

# Pre-create the CMake/scikit-build directory owned by `dev`: the named volume
# mounted at this path inherits this ownership when it is first created, which
# keeps the editable install writable for the non-root user.
RUN mkdir -p /workspace/build && chown dev:dev /workspace/build

# Debian's /etc/profile resets PATH, which would hide the virtualenv in login
# shells (VS Code's integrated terminal, `bash -l`); re-apply it for those.
RUN printf '%s\n' 'export VIRTUAL_ENV=/opt/venv' 'export PATH="/opt/venv/bin:$PATH"' \
    > /etc/profile.d/venv.sh

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /workspace
USER dev

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["sleep", "infinity"]
