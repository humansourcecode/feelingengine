# Feeling Engine — isolated reproducibility test image.
#
# Goal: prove the repo installs and passes tests from a clean state,
# with only what we explicitly install. If `docker build .` succeeds,
# a fresh user on a matching OS will have the same experience.
#
# Usage:
#   docker build -t feeling-engine-test .
#
# If the build completes, all checks passed. No API keys required —
# this image never hits external APIs. E2E tests are opt-in locally
# via `pytest --run-e2e` and are not run here.

FROM python:3.11-slim

# System dependencies declared in README.
# - ffmpeg (ffprobe) for audio duration probing in the ElevenLabs adapter
# - git is required by pip for editable installs
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only packaging + source first for layer caching.
COPY pyproject.toml LICENSE README.md ./
COPY feeling_engine ./feeling_engine
COPY tests ./tests
COPY examples ./examples
COPY deploy ./deploy
COPY docs ./docs
COPY .env.example ./

# Install with all provider extras + dev tooling.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[all,dev]"

# Unit tests — 22 tests, no external API calls.
RUN python -m pytest tests/unit -v

# Mode 1 smoke test against the bundled profile fixture.
# Catches CLI-level regressions that unit tests don't hit.
RUN python examples/analyze_speech.py \
        --profiles tests/unit/fixtures/tiny_profiles.json \
        --no-layer4 --no-fire \
        > /tmp/mode1_output.txt && \
    grep -q "Emotional Arc" /tmp/mode1_output.txt && \
    echo "Mode 1 smoke test passed"

# Default entry: drop the user into an interactive shell with the
# package installed, so they can explore.
CMD ["bash"]
