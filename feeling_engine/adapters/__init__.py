"""Pluggable adapters for TTS, GPU compute, and brain models.

Each adapter wraps a specific service (ElevenLabs, Modal, TRIBE v2, etc.)
behind a common interface. Users bring their own API keys and choose
their preferred providers.
"""
