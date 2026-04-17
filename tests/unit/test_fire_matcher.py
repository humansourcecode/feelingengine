"""Fire matcher — cross-domain emotional pattern matching."""
from feeling_engine.fire.matcher import FireMatcher


def test_corpus_loads():
    matcher = FireMatcher()
    stats = matcher.corpus_stats()
    assert stats["total_entries"] >= 6
    assert stats["vocabulary_size"] > 0


def test_corpus_spans_multiple_domains():
    matcher = FireMatcher()
    stats = matcher.corpus_stats()
    # Sample corpus intentionally covers multiple domains
    assert len(stats["domains"]) >= 4


def test_mortality_feelings_match_death_content():
    """Feelings centered on mortality should match mortality-themed entries."""
    matcher = FireMatcher()
    feelings = ["composure", "mortality_awareness", "body_response", "reverence", "resolve"]
    matches = matcher.match_feelings(feelings, top_k=5)

    assert matches, "Expected at least one match for mortality feelings"
    # The sample corpus has Jobs death pivot, Marcus Aurelius, Ballou letter, Keats —
    # at least one of these should surface in the top matches
    expected_titles = {
        "Steve Jobs — Stanford Commencement (Death Pivot)",
        "Marcus Aurelius — Meditations, Book II",
        "Sullivan Ballou — Last Letter to Wife (Civil War)",
        "John Keats — Ode to a Nightingale",
    }
    top_titles = {m.entry.title for m in matches}
    assert top_titles & expected_titles, (
        f"None of {expected_titles} surfaced in matches: {top_titles}"
    )


def test_empty_feelings_returns_no_matches():
    matcher = FireMatcher()
    assert matcher.match_feelings([]) == []


def test_arrow_separated_parsing():
    matcher = FireMatcher()
    matches = matcher.match_feelings_string(
        "composure -> mortality_awareness -> body_response"
    )
    assert matches, "Arrow-separated string should parse and match"


def test_match_result_shape():
    matcher = FireMatcher()
    matches = matcher.match_feelings(["mortality_awareness"], top_k=3)
    for m in matches:
        assert 0.0 <= m.score <= 1.0
        assert 0.0 <= m.cosine_score <= 1.0
        assert 0.0 <= m.sequence_score <= 1.0
        assert isinstance(m.shared_feelings, list)
