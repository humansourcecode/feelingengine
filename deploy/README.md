# Deployment scripts

This directory contains standalone scripts that deploy Feeling Engine's
compute backends. They are **not** part of the `feeling_engine` Python
package — they're operator scripts you run once per environment.

## TRIBE v2 on Modal

Runs TRIBE v2 brain prediction on Modal's serverless A100 GPUs.
`feeling_engine/adapters/compute/modal_tribe.py` calls the deployed
function; this script is what puts the function in Modal to begin with.

### One-time setup

```bash
# 1. Install Modal
pip install modal

# 2. Authenticate (browser-based)
modal setup

# 3. Get a HuggingFace access token with TRIBE v2 model access
#    Request access: https://huggingface.co/facebook/tribev2
#    Then create a token: https://huggingface.co/settings/tokens
#    Export it in your shell:
export HUGGINGFACE_ACCESS_TOKEN=hf_...

# 4. Deploy TRIBE to Modal (first deploy takes ~10 min, image build is cached)
modal deploy deploy/tribe_modal.py
```

After deploy, the function is live at `tribe-v2.predict_brain` in your
Modal account. Feeling Engine's `ModalTRIBEAdapter` picks it up automatically.

### Verify deployment

```bash
# Direct CLI call — smoke test the Modal function without Feeling Engine
python deploy/tribe_modal.py predict path/to/short_audio.mp3
```

Or via the adapter:

```python
from feeling_engine.adapters.compute.modal_tribe import ModalTRIBEAdapter

ModalTRIBEAdapter.is_deployed()  # → True
```

### Costs

- First deploy: ~$0 (only build time, no GPU use)
- Per prediction: ~$0.08 for 60 seconds of audio on A100-40GB
- Modal Starter plan includes $30/month free credits

### Teardown (if you want to stop paying)

```bash
modal app stop tribe-v2
```

## License note

These scripts are MIT (part of Feeling Engine). TRIBE v2 itself — the
model weights downloaded at deploy time — is CC BY-NC 4.0 from Meta FAIR.
You need your own HuggingFace access to the weights. Feeling Engine
does not redistribute them.
