"""Brain-to-Emotion Translator — the core of Feeling Engine.

Converts brain region activation time series into human-readable
emotional arcs with confidence scores and neuroscience grounding.

Five layers:
1. Brain data (region activations per timestep)
2. Change detection (where brain states shift significantly)
3. Neuroscience mapping (dimensional: valence/arousal/body-focus)
4. LLM synthesis (constrained interpretation of measurement data)
5. Confidence scoring (per-label, based on activation strength + framework agreement)
"""
