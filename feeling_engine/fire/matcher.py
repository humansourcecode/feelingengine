"""Fire — Cross-domain emotional pattern matching.

Given an emotional arc (from the Translator), finds content in the
corpus whose emotional signature most closely matches. Surfaces
cross-cultural and cross-domain precedents.

"This speech shares an emotional structure with a 16th-century
Japanese death poem and a soldier's last letter from WWII."

Matching uses two signals:
1. Vocabulary vector similarity (cosine): do the same feelings appear?
2. Sequence similarity (Levenshtein): do the feelings appear in the same ORDER?

Combined: 0.7 * cosine + 0.3 * sequence = match score.

Usage:
    from feeling_engine.fire.matcher import FireMatcher

    matcher = FireMatcher()  # loads corpus from corpus/sample/
    matches = matcher.match_arc(emotional_arc, top_k=5)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from feeling_engine.translator.brain_to_emotion import EmotionalArc


CORPUS_DIR = Path(__file__).parent / "corpus"


@dataclass
class CorpusEntry:
    """A single piece of content in the Fire corpus."""
    id: str
    title: str
    domain: str  # literature, speeches, poetry, letters, etc.
    feelings: list[str]  # ordered emotional arc terms
    vector: np.ndarray  # vocabulary vector (computed from feelings)
    source_text: Optional[str] = None
    excerpt: Optional[str] = None
    context: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class MatchResult:
    """A single match between a query arc and a corpus entry."""
    entry: CorpusEntry
    score: float  # combined similarity (0-1)
    cosine_score: float
    sequence_score: float
    shared_feelings: list[str]  # feelings that appear in both
    is_cross_domain: bool  # does this match cross a domain boundary?


class FireMatcher:
    """Cross-domain emotional pattern matcher.

    Loads a corpus of content with pre-computed emotional arcs,
    then matches new arcs against the corpus to find precedents.

    Args:
        corpus_path: directory containing corpus YAML files.
            Defaults to the bundled sample corpus.
        vocabulary_path: path to vocabulary.yaml for vector computation.
    """

    def __init__(
        self,
        corpus_path: Path | None = None,
        vocabulary_path: Path | None = None,
    ):
        self._corpus_path = corpus_path or CORPUS_DIR
        self._vocab_path = vocabulary_path or (
            Path(__file__).parent.parent / "translator" / "vocabulary.yaml"
        )
        self._vocabulary = self._load_vocabulary()
        self._corpus: list[CorpusEntry] = []
        self._load_corpus()

    def match_arc(
        self,
        arc: EmotionalArc,
        top_k: int = 5,
        min_score: float = 0.1,
    ) -> list[MatchResult]:
        """Match an emotional arc against the corpus.

        Args:
            arc: emotional arc from the Translator
            top_k: number of top matches to return
            min_score: minimum similarity threshold

        Returns:
            list of MatchResult, sorted by score descending
        """
        # Extract feelings sequence from the arc
        query_feelings = self._arc_to_feelings(arc)
        return self.match_feelings(query_feelings, top_k=top_k, min_score=min_score)

    def match_feelings(
        self,
        feelings: list[str],
        top_k: int = 5,
        min_score: float = 0.1,
    ) -> list[MatchResult]:
        """Match a feelings sequence against the corpus.

        Args:
            feelings: ordered list of emotional vocabulary terms
            top_k: number of top matches
            min_score: minimum threshold

        Returns:
            list of MatchResult, sorted by score descending
        """
        if not feelings or not self._corpus:
            return []

        query_vector = self._feelings_to_vector(feelings)
        query_domain = None  # query has no inherent domain

        results = []
        for entry in self._corpus:
            cos_sim = self._cosine_similarity(query_vector, entry.vector)
            seq_sim = self._sequence_similarity(feelings, entry.feelings)
            combined = 0.7 * cos_sim + 0.3 * seq_sim

            if combined < min_score:
                continue

            shared = set(feelings) & set(entry.feelings)

            results.append(MatchResult(
                entry=entry,
                score=combined,
                cosine_score=cos_sim,
                sequence_score=seq_sim,
                shared_feelings=sorted(shared),
                is_cross_domain=True,  # all corpus matches are cross-domain by nature
            ))

        results.sort(key=lambda r: -r.score)
        return results[:top_k]

    def match_feelings_string(
        self,
        feelings_string: str,
        top_k: int = 5,
    ) -> list[MatchResult]:
        """Match from an arrow-separated feelings string.

        Accepts: "tenderness -> longing -> catharsis"
        """
        feelings = self._parse_feelings_string(feelings_string)
        return self.match_feelings(feelings, top_k=top_k)

    def add_entry(
        self,
        id: str,
        title: str,
        domain: str,
        feelings: list[str],
        source_text: Optional[str] = None,
        excerpt: Optional[str] = None,
        context: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> CorpusEntry:
        """Add a new entry to the in-memory corpus."""
        vector = self._feelings_to_vector(feelings)
        entry = CorpusEntry(
            id=id,
            title=title,
            domain=domain,
            feelings=feelings,
            vector=vector,
            source_text=source_text,
            excerpt=excerpt,
            context=context,
            metadata=metadata or {},
        )
        self._corpus.append(entry)
        return entry

    def corpus_stats(self) -> dict:
        """Return corpus statistics."""
        domains = {}
        for entry in self._corpus:
            domains[entry.domain] = domains.get(entry.domain, 0) + 1
        return {
            "total_entries": len(self._corpus),
            "domains": domains,
            "vocabulary_size": len(self._vocabulary),
        }

    def format_matches(self, matches: list[MatchResult]) -> str:
        """Render matches as human-readable text."""
        if not matches:
            return "No matches found."

        # Check for cross-domain flag
        domains = set(m.entry.domain for m in matches)
        cross_domain = len(domains) > 1

        lines = []
        if cross_domain:
            lines.append(f"CROSS-DOMAIN MATCHES (spans {', '.join(sorted(domains))})")
        lines.append(f"Top {len(matches)} matches:")
        lines.append("=" * 60)

        for i, m in enumerate(matches, 1):
            lines.append(f"\n#{i} — {m.entry.title}")
            lines.append(f"   Domain: {m.entry.domain}")
            lines.append(f"   Score: {m.score:.3f} "
                         f"(cosine={m.cosine_score:.3f}, "
                         f"sequence={m.sequence_score:.3f})")
            lines.append(f"   Shared feelings: {', '.join(m.shared_feelings[:8])}")
            if m.entry.excerpt:
                lines.append(f"   Excerpt: \"{m.entry.excerpt[:150]}...\"")

        return "\n".join(lines)

    # ─── Internal methods ───

    def _load_vocabulary(self) -> list[str]:
        """Load vocabulary terms from vocabulary.yaml."""
        if self._vocab_path.exists():
            with open(self._vocab_path) as f:
                data = yaml.safe_load(f)
            return list(data.get("terms", {}).keys())
        return []

    def _load_corpus(self):
        """Load corpus entries from YAML/JSON files in corpus directory."""
        self._corpus = []

        if not self._corpus_path.exists():
            return

        for path in sorted(self._corpus_path.rglob("*.yaml")):
            self._load_corpus_file(path)
        for path in sorted(self._corpus_path.rglob("*.json")):
            self._load_corpus_file(path)

    def _load_corpus_file(self, path: Path):
        """Load entries from a single corpus file."""
        try:
            text = path.read_text()
            if path.suffix == ".yaml":
                data = yaml.safe_load(text)
            else:
                data = json.loads(text)

            if isinstance(data, list):
                entries = data
            elif isinstance(data, dict) and "entries" in data:
                entries = data["entries"]
            elif isinstance(data, dict) and "id" in data:
                entries = [data]
            else:
                return

            for entry_data in entries:
                feelings = entry_data.get("feelings", [])
                if isinstance(feelings, str):
                    feelings = self._parse_feelings_string(feelings)

                if not feelings:
                    continue

                vector = self._feelings_to_vector(feelings)
                self._corpus.append(CorpusEntry(
                    id=entry_data.get("id", path.stem),
                    title=entry_data.get("title", path.stem),
                    domain=entry_data.get("domain", "unknown"),
                    feelings=feelings,
                    vector=vector,
                    source_text=entry_data.get("source_text"),
                    excerpt=entry_data.get("excerpt"),
                    context=entry_data.get("context"),
                    metadata=entry_data.get("metadata", {}),
                ))
        except Exception:
            pass  # skip malformed files silently

    def _arc_to_feelings(self, arc: EmotionalArc) -> list[str]:
        """Extract ordered feelings list from an EmotionalArc."""
        feelings = []
        for te in arc.timesteps:
            if te.primary:
                top_term = te.primary[0].term
                # Avoid consecutive duplicates
                if not feelings or feelings[-1] != top_term:
                    feelings.append(top_term)
        return feelings

    def _feelings_to_vector(self, feelings: list[str]) -> np.ndarray:
        """Convert feelings list to vocabulary vector.

        Each dimension = one vocabulary term. Value = position-weighted
        presence (earlier feelings get slightly higher weight).
        L2-normalized for cosine similarity.
        """
        vec = np.zeros(len(self._vocabulary))
        vocab_index = {term: i for i, term in enumerate(self._vocabulary)}
        total = len(feelings)

        for pos, term in enumerate(feelings):
            if term in vocab_index:
                weight = 1.0 + 0.1 * (total - pos) if total > 0 else 1.0
                idx = vocab_index[term]
                vec[idx] = max(vec[idx], weight)

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two L2-normalized vectors."""
        return float(np.clip(np.dot(a, b), -1.0, 1.0))

    @staticmethod
    def _sequence_similarity(seq_a: list[str], seq_b: list[str]) -> float:
        """Normalized Levenshtein distance between two feeling sequences.

        Returns 0.0 (completely different) to 1.0 (identical).
        """
        if not seq_a and not seq_b:
            return 1.0
        if not seq_a or not seq_b:
            return 0.0

        m, n = len(seq_a), len(seq_b)
        dp = list(range(n + 1))

        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                if seq_a[i - 1] == seq_b[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                prev = temp

        distance = dp[n]
        return 1.0 - (distance / max(m, n))

    @staticmethod
    def _parse_feelings_string(text: str) -> list[str]:
        """Parse arrow-separated feelings: 'tenderness -> longing -> catharsis'"""
        text = text.replace("\u2192", "->")
        terms = [t.strip().lower().replace("-", "_").replace(" ", "_")
                 for t in text.split("->")]
        return [t for t in terms if t]
