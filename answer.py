"""Turn a system's free-text decision into one answer letter.

MDAgents ends with free text, not a choice. Scoring that text with a loose
regex was unreliable with MedGemma 4B (see the thesis, Section 5.3), so the
letter is determined in a fixed order:

  1. stated   the decision says "the answer is (B)" / "Answer: B)" near its end
  2. model    a second, short model call is asked which option the decision picks
              (done in medgemma_backend.pick_letter, stored as `picked_letter`)
  3. nothing  neither worked: the question is counted as unscored

This file has no dependencies so that analyze.py can use it without a model.
"""

import re

# "answer is (B)", "Answer: B)", "Final Answer: **D**". The trailing
# (?![A-Za-z]) stops the pattern matching the "A" of a following "Answer".
STATED = re.compile(
    r'answer\s*(?:is)?\s*[:\-]?\s*\*{0,2}\s*\(?([A-E])\)?(?![A-Za-z])',
    re.IGNORECASE,
)


def text_of(decision):
    """A decision is a string, {temp: string} or {'majority': {temp: string}}."""
    if decision is None:
        return ""
    if isinstance(decision, str):
        return decision
    if isinstance(decision, dict):
        if "majority" in decision:
            return text_of(decision["majority"])
        return "\n".join(text_of(v) for v in decision.values())
    return str(decision)


def stated_letter(decision, letters):
    """The letter the decision explicitly states, or None."""
    tail = text_of(decision).strip()[-400:]
    found = STATED.findall(tail)
    if found and found[-1].upper() in letters:
        return found[-1].upper()
    return None


def final_letter(record):
    """The letter that gets scored for one result record."""
    letters = record["options"]
    return stated_letter(record["response"], letters) or record.get("picked_letter")
