"""Feeling Engine — QA toolkit for content review.

Produces a machine-readable review ledger identifying context gaps,
missing citations, placeholder text, and unexplained jargon in a written
article. Ledger format is compatible with the review_wizard.py interactive
tool.

Usage:

    # Generate a QA ledger for an article
    python -m feeling_engine.qa.context_qa article.md > qa_ledger.md

    # Walk through the ledger interactively
    python review_wizard.py qa_ledger.md

    # Apply amended text back into the article
    python -m feeling_engine.qa.apply_amendments qa_ledger.md article.md

    # Loop until zero issues remain.

See docs/methodology.md for the broader content-methodology context.
"""

from feeling_engine.qa.context_qa import run_qa, Issue

__all__ = ["run_qa", "Issue"]
