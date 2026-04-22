"""Content pre-processing for mechanism detection on non-text content.

Extends Tier 2/3 detectors beyond linguistic patterns by using a multimodal
LLM (Claude Sonnet vision) to describe visual + audio + prosodic cues in
content, producing an enriched transcript that linguistic detectors can then
pattern-match against.

Commercial-safe: uses commercial API (Anthropic), produces our IP output.
No TRIBE dependency.
"""
