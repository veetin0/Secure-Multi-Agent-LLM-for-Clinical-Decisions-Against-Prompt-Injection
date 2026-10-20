"""MedGemma behind an OpenAI-compatible server (vLLM, Ollama, ...).

    export MEDGEMMA_BASE_URL=http://localhost:8000/v1
    export MEDGEMMA_MODEL=google/medgemma-27b-text-it
    export MEDGEMMA_API_KEY=not-needed-for-local

Optional:
    MEDGEMMA_MAX_TOKENS   output cap per call (default 2048)
    MEDGEMMA_GUIDED=1     vLLM only: force the answer letter with guided decoding
    MEDGEMMA_TIMEOUT      seconds per request (default 300)
"""

import os
import re

from openai import OpenAI

from answer import stated_letter, text_of

TRUNCATED = 0     # completions cut off at the token cap, counted over the run
_client = None
_guided_works = True


def is_medgemma(model_info):
    return model_info is not None and "medgemma" in str(model_info).lower()


def model_name():
    return os.environ.get("MEDGEMMA_MODEL", "google/medgemma-27b-text-it")


def get_client():
    global _client
    if _client is None:
        base_url = os.environ.get("MEDGEMMA_BASE_URL")
        if not base_url:
            raise RuntimeError("MEDGEMMA_BASE_URL is not set. See the top of "
                               "medgemma_backend.py.")
        _client = OpenAI(
            base_url=base_url,
            api_key=os.environ.get("MEDGEMMA_API_KEY", "not-needed-for-local"),
            timeout=float(os.environ.get("MEDGEMMA_TIMEOUT", "300")),
        )
    return _client


def chat_completion(messages):
    """One model call. Counts completions that were cut off at the token cap."""
    global TRUNCATED
    max_tokens = int(os.environ.get("MEDGEMMA_MAX_TOKENS", "2048"))
    response = get_client().chat.completions.create(
        model=model_name(), messages=messages,
        temperature=0.0, max_tokens=max_tokens,
    )
    choice = response.choices[0]
    if choice.finish_reason == "length":
        TRUNCATED += 1
        print(f"[warn] completion hit max_tokens={max_tokens} "
              f"(truncated so far: {TRUNCATED})")
    return choice.message.content or ""


def pick_letter(decision, options):
    """Name the option a decision selects. Returns (letter, how).

    how is 'stated' (read directly from the text), 'model' (a short extra
    model call), 'unparsed' (the call returned no letter) or 'failed'.
    """
    letters = sorted(options)

    letter = stated_letter(decision, letters)
    if letter:
        return letter, "stated"

    text = text_of(decision).strip()
    menu = "\n".join(f"({L}) {options[L]}" for L in letters)
    prompt = (
        f"A medical panel reached this conclusion:\n\n{text}\n\n"
        f"The answer choices were:\n{menu}\n\n"
        "Which option does the conclusion select? Report what the conclusion "
        "already says; do not re-evaluate the case. "
        f"Reply with one letter ({', '.join(letters)}) and nothing else."
    )
    def ask(extra):
        return get_client().chat.completions.create(
            model=model_name(), temperature=0.0, max_tokens=8,
            messages=[{"role": "user", "content": prompt}], **extra,
        ).choices[0].message.content or ""

    global _guided_works
    try:
        if os.environ.get("MEDGEMMA_GUIDED") == "1" and _guided_works:
            try:
                reply = ask({"extra_body": {"guided_choice": letters}})   # vLLM only
            except Exception:
                # Newer vLLM versions renamed this option. Fall back to a
                # plain call rather than leaving the question unscored.
                _guided_works = False
                print("[warn] server rejected guided_choice; asking without it")
                reply = ask({})
        else:
            reply = ask({})
    except Exception as error:
        print(f"[warn] pick_letter call failed: {error}")
        return None, "failed"

    match = re.search(rf"\b([{''.join(letters)}])\b", reply.strip().upper())
    if match:
        return match.group(1), "model"
    print(f"[warn] pick_letter got no letter from: {reply!r}")
    return None, "unparsed"
