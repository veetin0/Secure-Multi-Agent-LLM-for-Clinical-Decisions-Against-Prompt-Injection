"""Capability layer for MDAgents (the thesis's protection layer).

Adapted from CaMeL (Debenedetti et al. 2025). Three ideas carry over:

  provenance   every value records where it came from. The clinician's
               question is trusted; each retrieved passage is untrusted; a
               model's output inherits the sources of everything in its
               context.
  quarantine   an agent that has read a passage may pass on only a
               structured value, an option letter, never its free text
               (property P3). The letter is extracted from its reply by a
               call whose output is restricted to the option letters.
  policy       the final answer is computed from structured values under an
               explicit rule (property P2), instead of by a model reading
               free text.

Routing (P1) needs nothing here: the classifier and recruiter never see
passages (thesis, Section 4.4).

The rule at the decision point compares two majorities:
  tainted  the experts' choices after reading the passages and deliberating
  trusted  the same experts' choices on the question alone
If they agree, that answer stands. If they differ, the passages changed the
outcome, so the trusted answer is returned and the question is flagged for
human review.
"""

from collections import Counter
from dataclasses import dataclass

TRUSTED = frozenset({"question"})


class PolicyViolation(Exception):
    """A value was about to flow somewhere the policy forbids."""


@dataclass(frozen=True)
class Value:
    data: object
    sources: frozenset
    structured: bool = False 

    @property
    def trusted(self):
        return self.sources <= TRUSTED


def question(text):
    return Value(text, TRUSTED)


def passages(items):
    """The retrieved passages; each is its own untrusted source."""
    return Value(items, frozenset(f"passage:{i}" for i in range(len(items or []))))


def output_of(text, *context):
    """A model's free-text output: it carries the sources of its context."""
    return Value(text, frozenset().union(*(v.sources for v in context)))


def choice(letter, source):
    """A structured choice extracted from `source`, keeping its provenance."""
    return Value(letter, source.sources, structured=True)


def to_context(value):
    """P3: what may enter another agent's context.

    Trusted text and structured values may; untrusted free text may not.
    """
    if value.trusted or value.structured:
        return value.data
    raise PolicyViolation(
        f"untrusted free text from {sorted(value.sources)} may not enter another agent's context")


def majority(choices):
    """Most common letter among structured choices; a tie goes to the letter
    chosen first in roster order. None if there are no usable choices."""
    letters = [c.data for c in choices if c.structured and c.data]
    if not letters:
        return None
    counts = Counter(letters)
    top = max(counts.values())
    return next(L for L in letters if counts[L] == top)


def decide(tainted, trusted):
    """P2: the final answer from structured choices only.

    Returns a dict: answer, flagged, and both majorities, for the record.
    """
    for c in list(tainted) + list(trusted):
        if not c.structured:
            raise PolicyViolation("the decision may only use structured choices")
    t, s = majority(tainted), majority(trusted)
    if s is None:
        return {"answer": t, "flagged": t is not None, "tainted": t, "trusted": s}
    if t is None or t == s:
        return {"answer": s, "flagged": t is None, "tainted": t, "trusted": s}
    return {"answer": s, "flagged": True, "tainted": t, "trusted": s}
