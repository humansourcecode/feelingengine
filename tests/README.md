# Tests

## Directory layout

```
tests/
  unit/                     # fast, isolated, no external APIs
  e2e/                      # hits live APIs (ElevenLabs, Modal, Anthropic)
    fixtures/               # committed — test inputs (text, profiles)
    output/                 # gitignored — generated artifacts (MP3, JSON)
    .gitignore              # ignores output/
  README.md                 # you are here
```

## The fixtures / output pattern

**All tests that generate artifacts must use this pattern:**

| Path | Committed? | What goes here |
|---|---|---|
| `tests/<suite>/fixtures/` | ✅ yes | Small, stable inputs: text files, pre-computed profiles, expected arcs. Version-controlled so tests are reproducible. |
| `tests/<suite>/output/` | ❌ no (gitignored) | Generated artifacts: MP3s from TTS, TRIBE prediction JSON, rendered arcs. Regeneratable; don't bloat the repo. |

**Never** write test artifacts to `/tmp`. Ephemeral, un-reproducible, lost on reboot.

**Never** commit `output/` contents. MP3s and prediction arrays are large and regenerable.

### Example

```python
# tests/e2e/test_text_to_arc.py
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def test_text_to_arc():
    text = (FIXTURES / "jobs_short.txt").read_text()
    audio_path = OUTPUT / "jobs_short.mp3"
    # ... synthesize, predict, assert ...
```

## E2E tests and cost

Live-API tests cost real money per run (~$0.03-0.11 depending on clip length).
- Keep fixtures short (10-20 seconds of audio).
- Mark E2E tests with a pytest marker so they can be skipped in CI:
  `@pytest.mark.e2e` → skipped unless `--run-e2e` is passed.
- Record cost expectations in the test docstring.

## Running tests

```bash
# Unit tests only (default, fast, free)
pytest

# Include E2E tests (costs money — requires .env with live keys)
pytest --run-e2e

# Just one E2E test
pytest tests/e2e/test_text_to_arc.py --run-e2e -v
```
