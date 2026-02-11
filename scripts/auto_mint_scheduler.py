#!/usr/bin/env python3
import argparse
import ast
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone

import requests

DEFAULT_BASE_URL = "https://www.moltbook.com/api/v1"
PROFILE_DIR = os.path.join(os.path.expanduser("~"), ".config", "moltbook-wizard")
PROFILES_PATH = os.path.join(PROFILE_DIR, "profiles.json")
DEFAULT_STATE_PATH = os.path.join(PROFILE_DIR, "auto_mint_state.json")
CONTENT_ID_LOG_PATH = os.path.join(PROFILE_DIR, "content_ids.log")
LLM_CONFIG_PATH = os.path.join(PROFILE_DIR, "llm_config.json")
OPENROUTER_CONFIG_PATH = os.path.join(PROFILE_DIR, "openrouter.json")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
DEFAULT_OPENROUTER_MODEL = "openrouter/auto"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
LOCAL_ONLY_VERIFICATION = True
LLM_PROVIDERS = ("openrouter", "openai", "gemini")

_SMALL_NUMBERS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS_NUMBERS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_SCALE_NUMBERS = {
    "hundred": 100,
    "thousand": 1000,
    "million": 1000000,
}
_NUMBER_WORDS = set(_SMALL_NUMBERS) | set(_TENS_NUMBERS)
def _build_number_variants():
    variants = {}

    def add(value, text):
        key = re.sub(r"[^a-z]", "", text.lower())
        if not key:
            return
        variants.setdefault(key, value)

    for word, value in _SMALL_NUMBERS.items():
        if 1 <= value <= 19:
            add(value, word)
    for word, value in _TENS_NUMBERS.items():
        add(value, word)
    for tens_word, tens_value in _TENS_NUMBERS.items():
        for ones_word, ones_value in _SMALL_NUMBERS.items():
            if ones_value == 0:
                continue
            value = tens_value + ones_value
            add(value, f"{tens_word}{ones_word}")
            add(value, f"{tens_word}-{ones_word}")
            add(value, f"{tens_word} {ones_word}")
    add(100, "hundred")
    add(100, "one hundred")
    add(100, "one-hundred")
    add(100, "onehundred")
    return variants

_NUMBER_VARIANTS = _build_number_variants()
_OP_WORDS = {
    "plus": "+",
    "add": "+",
    "added": "+",
    "minus": "-",
    "subtract": "-",
    "subtracted": "-",
    "times": "*",
    "multiplied": "*",
    "multiply": "*",
    "x": "*",
    "divide": "/",
    "divided": "/",
    "over": "/",
}
_SUM_HINTS = {
    "total",
    "sum",
    "together",
    "altogether",
    "combined",
    "overall",
    "in all",
    "accelerates",
    "new velocity",
    "new speed",
    "velocity",
    "speed",
    "force",
    "gains",
    "gain",
    "increase",
    "increases",
    "added",
    "add",
    "plus",
}
_SUBTRACT_HINTS = {
    "difference",
    "minus",
    "subtract",
    "subtracted",
    "less",
    "decrease",
    "decreases",
    "reduced",
    "reduce",
    "reduces",
    "drop",
    "drops",
    "dropped",
    "slow",
    "slows",
    "slowed",
    "slowing",
    "decelerate",
    "decelerates",
    "decelerated",
}
_MULTIPLY_HINTS = {
    "product",
    "times",
    "multiplied",
    "multiply",
    "multiplies",
    "multiplier",
    "multipliers",
    "multiplication",
}
_DIVIDE_HINTS = {
    "divide",
    "divided",
    "over",
    "quotient",
}
_RATE_UNITS = {
    "per second",
    "per minute",
    "per hour",
    "per day",
    "per week",
    "per month",
    "per year",
}
_TIME_UNITS = {
    "second",
    "seconds",
    "minute",
    "minutes",
    "hour",
    "hours",
    "day",
    "days",
    "week",
    "weeks",
    "month",
    "months",
    "year",
    "years",
}
_DURATION_HINTS = {"for", "during", "over", "travel", "traveled", "travelled", "distance", "moved"}
_COUNT_REPEAT_HINTS = {
    "touches",
    "hits",
    "strikes",
    "times",
    "repeats",
    "repetitions",
    "occurrences",
    "instances",
}
_COUNT_ENTITY_HINTS = {
    "claws",
    "lobsters",
    "people",
}
_COUNT_HINTS = _COUNT_REPEAT_HINTS | _COUNT_ENTITY_HINTS
_COUNT_VERBS = {"are", "is", "were", "was"}


def _utc_now():
    return datetime.now(timezone.utc)


def _parse_iso(ts):
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def _save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.chmod(path, 0o600)


def _record_content_id(content_id, label=None, content_type=None):
    if not content_id:
        return
    timestamp = _utc_now().isoformat()
    link = f"https://www.moltbook.com/post/{content_id}"
    label = label or "unknown"
    content_type = content_type or "post"
    line = f"{timestamp} | {label} | {content_type} | {content_id} | {link}\n"
    os.makedirs(PROFILE_DIR, exist_ok=True)
    with open(CONTENT_ID_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line)


def _load_llm_config():
    data = {}
    if os.path.exists(LLM_CONFIG_PATH):
        try:
            with open(LLM_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            data = {}
    if not isinstance(data, dict):
        data = {}
    if "openrouter" not in data and os.path.exists(OPENROUTER_CONFIG_PATH):
        try:
            with open(OPENROUTER_CONFIG_PATH, "r", encoding="utf-8") as f:
                legacy = json.load(f)
        except (OSError, json.JSONDecodeError):
            legacy = {}
        if isinstance(legacy, dict) and legacy.get("api_key"):
            data["openrouter"] = {
                "api_key": legacy.get("api_key"),
                "model": legacy.get("model", DEFAULT_OPENROUTER_MODEL),
            }
    return data


def _get_llm_provider_config(provider):
    return _load_llm_config().get(provider, {})


def _get_preferred_provider():
    preferred = _load_llm_config().get("preferred_provider")
    if preferred in LLM_PROVIDERS:
        return preferred
    return None


def _get_openrouter_key():
    env_key = os.getenv("OPENROUTER_API_KEY")
    if env_key:
        return env_key
    return _get_llm_provider_config("openrouter").get("api_key")


def _get_openrouter_model():
    env_model = os.getenv("OPENROUTER_MODEL")
    if env_model:
        return env_model
    return _get_llm_provider_config("openrouter").get("model", DEFAULT_OPENROUTER_MODEL)


def _get_openai_key():
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key
    return _get_llm_provider_config("openai").get("api_key")


def _get_openai_model():
    env_model = os.getenv("OPENAI_MODEL")
    if env_model:
        return env_model
    return _get_llm_provider_config("openai").get("model", DEFAULT_OPENAI_MODEL)


def _get_gemini_key():
    env_key = os.getenv("GEMINI_API_KEY")
    if env_key:
        return env_key
    return _get_llm_provider_config("gemini").get("api_key")


def _get_gemini_model():
    env_model = os.getenv("GEMINI_MODEL")
    if env_model:
        return env_model
    return _get_llm_provider_config("gemini").get("model", DEFAULT_GEMINI_MODEL)


def _extract_number(text):
    if not text:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return f"{float(match.group(0)):.2f}"
    except ValueError:
        return None


def _format_answer(value):
    if value is None:
        return None
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return value


def _clean_challenge_for_llm(challenge):
    if not challenge:
        return ""
    text = challenge.lower()
    text = re.sub(r"([a-z])[^a-z0-9\s]+([a-z])", r"\1\2", text)
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    tokens = _merge_numeric_fragments(text.split())
    parts = []
    for token in tokens:
        if token.isalpha():
            parts.append(_normalize_number_token(token))
        else:
            parts.append(token)
    return " ".join(parts)


def _merge_numeric_fragments(tokens):
    merged = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.isalpha():
            norm = _normalize_word(token)
            max_span = min(6, len(tokens) - i)
            if len(token) <= 2:
                max_span = min(12, len(tokens) - i)
            best_word = None
            best_span = None
            best_dist = None
            for span in range(2, max_span + 1):
                if not all(tokens[i + j].isalpha() for j in range(span)):
                    continue
                span_tokens = tokens[i : i + span]
                all_short = all(len(t) <= 2 for t in span_tokens)
                combined = "".join(_normalize_word(t) for t in span_tokens)
                strict = _strict_number_word(combined)
                if strict:
                    if best_word is None or best_dist is None or 0 < best_dist or span < best_span:
                        best_word = strict
                        best_span = span
                        best_dist = 0
                    continue
                if not all_short:
                    continue
                candidate = _normalize_number_token(combined)
                if (
                    candidate not in _NUMBER_WORDS
                    and candidate not in _SCALE_NUMBERS
                    and candidate not in _NUMBER_VARIANTS
                    and candidate != "and"
                ):
                    continue
                dist = _edit_distance_limited(re.sub(r"(.)\1+", r"\1", combined), candidate, 2)
                if dist is None:
                    dist = 2
                if best_word is None or dist < best_dist or (dist == best_dist and span < best_span):
                    best_word = candidate
                    best_span = span
                    best_dist = dist
            if best_word:
                merged.append(best_word)
                i += best_span
                continue
            merged.append(_normalize_number_token(norm))
        else:
            merged.append(token)
        i += 1
    return merged

def _prepare_hint_text(challenge):
    cleaned = _clean_challenge_for_llm(challenge).lower()
    compact = re.sub(r"[^a-z]+", "", cleaned)
    compact_norm = _normalize_word(compact)
    return cleaned, compact, compact_norm


def _has_hint(hints, cleaned, compact, compact_norm):
    for hint in hints:
        if hint in cleaned:
            return True
        if hint.replace(" ", "") in compact:
            return True
        if hint.replace(" ", "") in compact_norm:
            return True
    return False


def _has_word(cleaned, word):
    return bool(re.search(rf"\b{re.escape(word)}\b", cleaned))


def _has_any_word(cleaned, words):
    return any(_has_word(cleaned, word) for word in words)


def _has_duration_hint(cleaned, compact, compact_norm):
    if _has_any_word(cleaned, _DURATION_HINTS):
        return True
    if cleaned:
        number_words = "|".join(sorted(_NUMBER_WORDS | set(_SCALE_NUMBERS)))
        time_words = "|".join(sorted(_TIME_UNITS))
        pattern = rf"\bin\s+(?:\d+|{number_words})(?:\s+(?:{number_words}))?\s+(?:{time_words})\b"
        if re.search(pattern, cleaned):
            return True
    if not compact:
        return False
    for hint in _DURATION_HINTS:
        for unit in _TIME_UNITS:
            combo = f"{hint}{unit}"
            if combo in compact or combo in compact_norm:
                return True
    return False


def _should_multiply_rate(cleaned, compact, compact_norm):
    if not cleaned:
        return False
    if not _has_hint(_RATE_UNITS, cleaned, compact, compact_norm):
        return False
    if not _has_hint(_TIME_UNITS, cleaned, compact, compact_norm):
        return False
    return _has_duration_hint(cleaned, compact, compact_norm)


def _should_multiply_count(cleaned, compact, compact_norm, numbers):
    if not cleaned or len(numbers) != 2:
        return False
    has_repeat = _has_hint(_COUNT_REPEAT_HINTS, cleaned, compact, compact_norm)
    has_entity = _has_hint(_COUNT_ENTITY_HINTS, cleaned, compact, compact_norm)
    if not (has_repeat or has_entity):
        return False
    if has_repeat:
        if _has_any_word(cleaned, _COUNT_VERBS) or _has_hint(_SUM_HINTS, cleaned, compact, compact_norm):
            return True
    if has_entity:
        for phrase in ("there are", "there is", "number of", "count of", "total of", "in total"):
            if phrase in cleaned or phrase.replace(" ", "") in compact or phrase.replace(" ", "") in compact_norm:
                return True
    return False


def _should_multiply_each(cleaned, compact, compact_norm):
    if not cleaned:
        return False
    if "each" not in cleaned and "per" not in cleaned:
        return False
    for unit in _RATE_UNITS:
        if unit in cleaned or unit.replace(" ", "") in compact or unit.replace(" ", "") in compact_norm:
            return False
    return any(
        phrase in cleaned or phrase.replace(" ", "") in compact or phrase.replace(" ", "") in compact_norm
        for phrase in (
            "there are",
            "there is",
            "number of",
            "how many",
            "total of",
            "in total",
            "overall",
            "altogether",
        )
    )


def _extract_message_content(message):
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if text:
                    parts.append(text)
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts).strip()
    return ""


def _build_math_prompt(challenge):
    expr = _build_llm_expression(challenge)
    if expr:
        return f"Calculate: {expr}. Return only the number with exactly two decimal places."
    cleaned = _clean_challenge_for_llm(challenge)
    return (
        "Calculate the result and return only the number with exactly two decimal places: "
        f"{cleaned or challenge}"
    )


def _solve_with_openrouter(challenge):
    api_key = _get_openrouter_key()
    if not api_key:
        return None, None
    model = _get_openrouter_model()
    prompt = _build_math_prompt(challenge)
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a precise calculator."},
            {"role": "user", "content": prompt},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://moltbook-wizard.local",
        "X-Title": "Moltbook Wizard",
    }
    try:
        resp = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=20)
    except requests.RequestException as exc:
        return None, f"OpenRouter request failed: {exc}"
    if resp.status_code not in (200, 201):
        return None, f"OpenRouter HTTP {resp.status_code}: {resp.text}"
    try:
        data = resp.json()
    except ValueError:
        return None, "OpenRouter response was not JSON"
    content = ""
    choices = data.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        content = _extract_message_content(message)
    answer = _extract_number(content)
    if not answer:
        return None, "OpenRouter returned no usable number"
    return answer, None


def _solve_with_openai(challenge):
    api_key = _get_openai_key()
    if not api_key:
        return None, None
    model = _get_openai_model()
    prompt = _build_math_prompt(challenge)
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a precise calculator."},
            {"role": "user", "content": prompt},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=20)
    except requests.RequestException as exc:
        return None, f"OpenAI request failed: {exc}"
    if resp.status_code not in (200, 201):
        return None, f"OpenAI HTTP {resp.status_code}: {resp.text}"
    try:
        data = resp.json()
    except ValueError:
        return None, "OpenAI response was not JSON"
    content = ""
    choices = data.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        content = _extract_message_content(message)
    answer = _extract_number(content)
    if not answer:
        return None, "OpenAI returned no usable number"
    return answer, None


def _extract_gemini_text(data):
    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    content = candidates[0].get("content")
    if isinstance(content, dict):
        parts = content.get("parts") or []
        texts = []
        for part in parts:
            if isinstance(part, dict) and part.get("text"):
                texts.append(part["text"])
            elif isinstance(part, str):
                texts.append(part)
        if texts:
            return "\n".join(texts).strip()
    if isinstance(content, str):
        return content.strip()
    for key in ("text", "output"):
        value = candidates[0].get(key)
        if isinstance(value, str):
            return value.strip()
    for key in ("output_text", "outputText"):
        value = candidates[0].get(key)
        if isinstance(value, str):
            return value.strip()
    return ""


def _solve_with_gemini(challenge):
    api_key = _get_gemini_key()
    if not api_key:
        return None, None
    model = _get_gemini_model()
    prompt = _build_math_prompt(challenge)
    candidates = [model]
    if model == DEFAULT_GEMINI_MODEL:
        candidates += [
            "gemini-1.5-flash-latest",
            "gemini-1.5-flash",
            "gemini-1.5-pro-latest",
            "gemini-1.5-pro",
        ]
    last_error = None
    for candidate in candidates:
        url = GEMINI_API_URL.format(model=candidate, api_key=api_key)
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": 64,
                "responseMimeType": "text/plain",
            },
        }
        try:
            resp = requests.post(url, json=payload, timeout=20)
        except requests.RequestException as exc:
            return None, f"Gemini request failed: {exc}"
        if resp.status_code not in (200, 201):
            if resp.status_code == 404 and candidate != candidates[-1]:
                last_error = f"Gemini HTTP 404 for model {candidate}"
                continue
            return None, f"Gemini HTTP {resp.status_code}: {resp.text}"
        try:
            data = resp.json()
        except ValueError:
            return None, "Gemini response was not JSON"
        content = _extract_gemini_text(data)
        answer = _extract_number(content)
        if not answer:
            return None, "Gemini returned no usable number"
        return answer, None
    if last_error:
        return None, last_error
    return None, "Gemini returned no usable number"


def _load_profiles(path):
    data = _load_json(path, {"active": None, "profiles": {}})
    return data.get("profiles", {})


def _normalize_word(word):
    return re.sub(r"(.)\1{2,}", r"\1", word)


def _collapse_duplicate_number_words(words):
    collapsed = []
    prev = None
    for word in words:
        if word == prev and (word in _NUMBER_WORDS or word in _SCALE_NUMBERS):
            continue
        collapsed.append(word)
        prev = word
    return collapsed


def _normalize_number_token(token):
    if not token:
        return token
    if token.isalpha():
        token = token.lower()
        strict = _strict_number_word(token)
        if strict:
            return strict
        softened = _normalize_word(token)
        collapsed = re.sub(r"(.)\1+", r"\1", softened)
        fixed = _fix_number_word(collapsed)
        if fixed in _NUMBER_WORDS or fixed in _SCALE_NUMBERS or fixed == "and":
            return fixed
        fixed = _fix_number_word(softened)
        if fixed in _NUMBER_WORDS or fixed in _SCALE_NUMBERS or fixed == "and":
            return fixed
        return collapsed
    return token


def _strict_number_word(token):
    token = token.lower()
    if token in _NUMBER_WORDS or token in _SCALE_NUMBERS or token == "and":
        return token
    if token in _NUMBER_VARIANTS:
        return token
    softened = _normalize_word(token)
    if softened in _NUMBER_WORDS or softened in _SCALE_NUMBERS or softened == "and":
        return softened
    if softened in _NUMBER_VARIANTS:
        return softened
    collapsed = re.sub(r"(.)\1+", r"\1", softened)
    if collapsed in _NUMBER_WORDS or collapsed in _SCALE_NUMBERS or collapsed == "and":
        return collapsed
    if collapsed in _NUMBER_VARIANTS:
        return collapsed
    return None


def _fix_number_word(token):
    if token in _NUMBER_WORDS or token in _SCALE_NUMBERS or token == "and":
        return token
    if len(token) >= 4:
        candidates = [
            word
            for word in (_NUMBER_WORDS | set(_SCALE_NUMBERS))
            if word.startswith(token) and len(word) == len(token) + 1
        ]
        if len(candidates) == 1:
            return candidates[0]
        if 4 <= len(token) <= 8:
            for word in (_NUMBER_WORDS | set(_SCALE_NUMBERS)):
                if len(word) == len(token) + 1 and word[0] == token[0]:
                    if _is_subsequence(token, word):
                        return word
    fuzzy = _fuzzy_number_word(token)
    if fuzzy:
        return fuzzy
    return token


def _is_subsequence(shorter, longer):
    idx = 0
    for ch in longer:
        if idx < len(shorter) and shorter[idx] == ch:
            idx += 1
    return idx == len(shorter)


def _edit_distance_limited(a, b, max_dist):
    if abs(len(a) - len(b)) > max_dist:
        return None
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        row_min = curr[0]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
            row_min = min(row_min, curr[-1])
        if row_min > max_dist:
            return None
        prev = curr
    dist = prev[-1]
    if dist > max_dist:
        return None
    return dist


def _fuzzy_number_word(token):
    if not token or len(token) < 4:
        return None
    best = None
    best_dist = None
    for word in (_NUMBER_WORDS | set(_SCALE_NUMBERS)):
        if token[0] != word[0]:
            continue
        if abs(len(token) - len(word)) > 2:
            continue
        max_dist = 1 if len(word) <= 5 else 2
        dist = _edit_distance_limited(token, word, max_dist)
        if dist is None:
            continue
        if best_dist is None or dist < best_dist:
            best = word
            best_dist = dist
    return best


def _format_number(value):
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _is_number_str(token):
    return bool(re.match(r"^-?\d+(?:\.\d+)?$", str(token)))


def _extract_explicit_expr(tokens):
    expr_tokens = []
    i = 0
    while i + 2 < len(tokens):
        if _is_number_str(tokens[i]) and tokens[i + 1] in "+-*/" and _is_number_str(tokens[i + 2]):
            if not expr_tokens:
                expr_tokens.extend([tokens[i], tokens[i + 1], tokens[i + 2]])
            else:
                expr_tokens.extend([tokens[i + 1], tokens[i + 2]])
            i += 2
            continue
        i += 1
    return " ".join(expr_tokens).strip()


def _build_llm_expression(challenge):
    expr, numbers, has_operator = _challenge_to_expr(challenge)
    if not has_operator and len(numbers) < 2:
        return ""
    expr = (expr or "").strip()
    if expr and any(op in expr for op in "+-*/"):
        try:
            _safe_eval(expr)
            return expr
        except Exception:
            pass
    cleaned, compact, compact_norm = _prepare_hint_text(challenge)
    op = None
    if len(numbers) >= 2 and _should_multiply_rate(cleaned, compact, compact_norm):
        op = "*"
    elif _should_multiply_count(cleaned, compact, compact_norm, numbers):
        op = "*"
    elif len(numbers) >= 2 and _should_multiply_each(cleaned, compact, compact_norm):
        op = "*"
    elif len(numbers) >= 2 and _has_hint(_MULTIPLY_HINTS, cleaned, compact, compact_norm):
        op = "*"
    elif len(numbers) >= 2 and _has_hint(_DIVIDE_HINTS, cleaned, compact, compact_norm):
        op = "/"
    elif len(numbers) >= 2 and _has_hint(_SUBTRACT_HINTS, cleaned, compact, compact_norm):
        op = "-"
    elif numbers and _has_hint(_SUM_HINTS, cleaned, compact, compact_norm):
        op = "+"
    if numbers:
        formatted = [_format_number(n) for n in numbers]
        if op == "+":
            return " + ".join(formatted)
        if op in ("-", "*", "/") and len(formatted) >= 2:
            return f"{formatted[0]} {op} {formatted[1]}"
        if len(formatted) >= 2:
            return " + ".join(formatted)
        return formatted[0]
    return ""


def _words_to_number(words):
    total = 0
    current = 0
    has_tens = False
    for word in words:
        if word == "and":
            continue
        if word in _NUMBER_VARIANTS:
            value = _NUMBER_VARIANTS[word]
            if value == 100:
                current = max(1, current) * 100 if current else 100
                has_tens = False
                continue
            if value >= 10 and has_tens:
                continue
            current += value
            if value >= 20:
                has_tens = True
            continue
        if word in _SMALL_NUMBERS:
            value = _SMALL_NUMBERS[word]
            if value >= 10 and has_tens:
                continue
            current += value
        elif word in _TENS_NUMBERS:
            if has_tens:
                continue
            current += _TENS_NUMBERS[word]
            has_tens = True
        elif word == "hundred":
            current = max(1, current) * _SCALE_NUMBERS[word]
            has_tens = False
        elif word in ("thousand", "million"):
            scale = _SCALE_NUMBERS[word]
            total += max(1, current) * scale
            current = 0
            has_tens = False
    return total + current


def _is_number_token(token):
    if re.match(r"\d", token or ""):
        return True
    if token and token.isalpha():
        token = _normalize_number_token(token)
        return token in _NUMBER_WORDS or token in _SCALE_NUMBERS or token in _NUMBER_VARIANTS or token == "and"
    return False


def _has_number_ahead(tokens, start):
    for idx in range(start, len(tokens)):
        token = tokens[idx]
        if _is_number_token(token):
            return True
        if token and token.isalpha() and idx + 1 < len(tokens) and tokens[idx + 1].isalpha():
            left = _normalize_word(token)
            right = _normalize_word(tokens[idx + 1])
            if left + right in _NUMBER_WORDS or left + right in _SCALE_NUMBERS:
                return True
    return False


def _challenge_to_expr(challenge):
    if not challenge:
        return "", [], False
    text = challenge.lower()
    text = re.sub(r"([a-z])[^a-z0-9\s]+([a-z])", r"\1\2", text)
    text = re.sub(r"(?<=[a-z])[+*/-]+", " ", text)
    text = re.sub(r"[+*/-]+(?=[a-z])", " ", text)
    text = text.replace("-", " ")
    tokens = _merge_numeric_fragments(re.findall(r"[a-z]+|\d+(?:\.\d+)?|[+*/()\-]", text))
    out = []
    numbers = []
    has_operator = False
    prev_number = False
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in _OP_WORDS:
            out.append(_OP_WORDS[token])
            has_operator = True
            prev_number = False
            i += 1
            continue
        if re.match(r"\d", token):
            out.append(token)
            try:
                numbers.append(float(token))
            except ValueError:
                pass
            prev_number = True
            i += 1
            continue
        if token.isalpha():
            token = _normalize_number_token(token)
        if token in _NUMBER_WORDS or token in _SCALE_NUMBERS or token in _NUMBER_VARIANTS or token == "and":
            words = []
            j = i
            while j < len(tokens):
                nxt = tokens[j]
                if nxt.isalpha():
                    nxt = _normalize_number_token(nxt)
                if nxt in _NUMBER_WORDS or nxt in _SCALE_NUMBERS or nxt in _NUMBER_VARIANTS or nxt == "and":
                    words.append(nxt)
                    j += 1
                    continue
                break
            words = _collapse_duplicate_number_words(words)
            has_numeric = any(
                word in _SMALL_NUMBERS
                or word in _TENS_NUMBERS
                or word in _SCALE_NUMBERS
                or word in _NUMBER_VARIANTS
                for word in words
            )
            if has_numeric:
                value = _words_to_number(words)
                out.append(str(value))
                numbers.append(float(value))
                prev_number = True
            i = j
            continue
        if token in "+-*/()":
            if token in "+-*/":
                if not prev_number:
                    i += 1
                    continue
                if not _has_number_ahead(tokens, i + 1):
                    i += 1
                    continue
                out.append(token)
                has_operator = True
                prev_number = False
                i += 1
                continue
            out.append(token)
            i += 1
            continue
        i += 1
    cleaned, compact, compact_norm = _prepare_hint_text(challenge)
    if _has_hint(_RATE_UNITS, cleaned, compact, compact_norm) and not _has_hint(
        _DIVIDE_HINTS, cleaned, compact, compact_norm
    ):
        out = [tok for tok in out if tok != "/"]
    if _has_hint(_SUM_HINTS, cleaned, compact, compact_norm) and not _has_hint(
        _DIVIDE_HINTS, cleaned, compact, compact_norm
    ):
        out = [tok for tok in out if tok != "/"]
    if _has_hint(_SUM_HINTS, cleaned, compact, compact_norm) and not _has_hint(
        _SUBTRACT_HINTS, cleaned, compact, compact_norm
    ):
        out = [tok for tok in out if tok != "-"]
    has_operator = any(tok in "+-*/" for tok in out)
    explicit_expr = _extract_explicit_expr(out)
    if explicit_expr:
        return explicit_expr, numbers, True
    return " ".join(out).strip(), numbers, has_operator


def _safe_eval(expr):
    node = ast.parse(expr, mode="eval")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Num):
            return node.n
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.UAdd):
                return _eval(node.operand)
            if isinstance(node.op, ast.USub):
                return -_eval(node.operand)
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
        raise ValueError("unsupported expression")

    return _eval(node)


def _solve_local_math(challenge):
    expr, numbers, has_operator = _challenge_to_expr(challenge)
    if not expr and not numbers:
        return None
    if expr:
        try:
            result = _safe_eval(expr)
            return f"{result:.2f}"
        except Exception:
            pass
    cleaned, compact, compact_norm = _prepare_hint_text(challenge)
    if len(numbers) >= 2 and _should_multiply_rate(cleaned, compact, compact_norm):
        return f"{numbers[0] * numbers[1]:.2f}"
    if _should_multiply_count(cleaned, compact, compact_norm, numbers):
        return f"{numbers[0] * numbers[1]:.2f}"
    if len(numbers) >= 2 and _should_multiply_each(cleaned, compact, compact_norm):
        return f"{numbers[0] * numbers[1]:.2f}"
    if len(numbers) >= 2 and _has_hint(_MULTIPLY_HINTS, cleaned, compact, compact_norm):
        return f"{numbers[0] * numbers[1]:.2f}"
    if len(numbers) >= 2 and _has_hint(_DIVIDE_HINTS, cleaned, compact, compact_norm):
        if numbers[1] == 0:
            return None
        return f"{numbers[0] / numbers[1]:.2f}"
    if len(numbers) >= 2 and _has_hint(_SUBTRACT_HINTS, cleaned, compact, compact_norm):
        return f"{numbers[0] - numbers[1]:.2f}"
    if numbers and _has_hint(_SUM_HINTS, cleaned, compact, compact_norm):
        return f"{sum(numbers):.2f}"
    if len(numbers) >= 2 and not has_operator:
        return f"{sum(numbers):.2f}"
    return None


def _solve_math_challenge(challenge):
    expression = _build_llm_expression(challenge)
    local_answer = _solve_local_math(challenge)
    if LOCAL_ONLY_VERIFICATION:
        details = {
            "llm": None,
            "local": local_answer,
            "llm_provider": None,
            "expression": expression,
        }
        if local_answer:
            return local_answer, "local", None, details
        return None, "local", "local solver failed", details
    solvers = {
        "openrouter": _solve_with_openrouter,
        "openai": _solve_with_openai,
        "gemini": _solve_with_gemini,
    }
    preferred = _get_preferred_provider()
    if preferred:
        solver = solvers.get(preferred)
        if solver:
            llm_answer, llm_error = solver(challenge)
            local_answer = _solve_local_math(challenge)
            details = {
                "llm": llm_answer,
                "local": local_answer,
                "llm_provider": preferred,
                "expression": expression,
            }
            if llm_answer and local_answer and llm_answer != local_answer:
                return local_answer, "local", f"LLM mismatch ({llm_answer})", details
            if llm_answer:
                return llm_answer, preferred, None, details
            if local_answer:
                return local_answer, "local", None, details
            return None, preferred, llm_error or "LLM failed", details
        return None, preferred, "preferred provider not supported", {
            "llm": None,
            "local": None,
            "llm_provider": preferred,
            "expression": expression,
        }

    local_answer = _solve_local_math(challenge)
    if local_answer:
        return local_answer, "local", None, {
            "llm": None,
            "local": local_answer,
            "llm_provider": None,
            "expression": expression,
        }
    for name in LLM_PROVIDERS:
        solver = solvers.get(name)
        if not solver:
            continue
        answer, error = solver(challenge)
        if answer:
            return answer, name, None, {
                "llm": answer,
                "local": local_answer,
                "llm_provider": name,
                "expression": expression,
            }
    return None, None, "unable to solve challenge", {
        "llm": None,
        "local": local_answer,
        "llm_provider": None,
        "expression": expression,
    }


def _request(method, url, api_key=None, json_body=None):
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if json_body is not None:
        headers["Content-Type"] = "application/json"
    try:
        resp = requests.request(method, url, headers=headers, json=json_body, timeout=30)
    except requests.RequestException as exc:
        return None, f"request failed: {exc}"
    if resp.status_code not in (200, 201):
        return None, f"HTTP {resp.status_code}: {resp.text}"
    try:
        return resp.json(), None
    except ValueError:
        return None, "response was not JSON"


def _latest_post_time(profile_data):
    if not isinstance(profile_data, dict):
        return None
    posts = profile_data.get("recentPosts") or profile_data.get("posts") or []
    latest = None
    for post in posts:
        created_at = _parse_iso(post.get("created_at"))
        if created_at and (latest is None or created_at > latest):
            latest = created_at
    return latest


def _parse_mint_from_content(content):
    if not content:
        return None
    match = re.search(r"\{.*?\}", content, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    if data.get("p") != "mbc-20" or data.get("op") != "mint":
        return None
    return data


def _remote_has_matching_mint(profile_data, tick, amount):
    if not isinstance(profile_data, dict):
        return False
    posts = profile_data.get("recentPosts") or profile_data.get("posts") or []
    for post in posts:
        content = post.get("content") or ""
        data = _parse_mint_from_content(content)
        if not data:
            continue
        if str(data.get("tick")) == str(tick) and str(data.get("amt")) == str(amount):
            return True
    return False


def _get_agent_name(base_url, api_key):
    data, error = _request("GET", f"{base_url}/agents/me", api_key=api_key)
    if not data:
        return None, error
    agent = data.get("agent", data)
    name = agent.get("name")
    if not name:
        return None, "missing agent name"
    return name, None


def _get_remote_last_post_time(base_url, api_key, agent_name):
    data, error = _request("GET", f"{base_url}/agents/profile?name={agent_name}", api_key=api_key)
    if not data:
        return None, error
    return _latest_post_time(data), None, data


def _signature_for(tick, amount, submolt, prefix):
    return f"{str(tick).lower()}|{amount}|{submolt}|{prefix}"


def _get_state_entry(state, profile, signature, legacy_key=None):
    profile_state = state.setdefault(profile, {})
    if signature in profile_state:
        return profile_state[signature]
    if legacy_key and legacy_key in profile_state:
        profile_state[signature] = profile_state.pop(legacy_key)
        return profile_state[signature]
    profile_state[signature] = {}
    return profile_state[signature]


def _get_state_times(state, profile, signature, legacy_key=None):
    entry = _get_state_entry(state, profile, signature, legacy_key=legacy_key)
    last_status = entry.get("last_status")
    last_post_at = _parse_iso(entry.get("last_post_at"))
    if last_status == "post_failed":
        last_post_at = None
    next_allowed_at = _parse_iso(entry.get("next_allowed_at"))
    sent_once = bool(entry.get("sent_once"))
    suspended_until = _parse_iso(entry.get("suspended_until"))
    return last_post_at, next_allowed_at, sent_once, suspended_until


def _pick_last_post(local_last, remote_last):
    if local_last and remote_last:
        return max(local_last, remote_last)
    return local_last or remote_last


def _combine_next_allowed(last_post, override_next, interval_minutes):
    base_next = None
    if last_post:
        base_next = last_post + timedelta(minutes=interval_minutes)
    if override_next and base_next:
        return max(override_next, base_next)
    return override_next or base_next


def _extract_retry_after_minutes(error_text):
    if not error_text or "HTTP 429:" not in error_text:
        return None
    payload = error_text.split("HTTP 429:", 1)[1].strip()
    try:
        data = json.loads(payload)
    except (TypeError, json.JSONDecodeError):
        return None
    retry_minutes = data.get("retry_after_minutes")
    if retry_minutes is not None:
        try:
            return float(retry_minutes)
        except (TypeError, ValueError):
            return None
    retry_seconds = data.get("retry_after_seconds")
    if retry_seconds is not None:
        try:
            return float(retry_seconds) / 60.0
        except (TypeError, ValueError):
            return None
    return None


def _record_post(state, profile, signature, status, post_id=None, post_time=None, next_allowed_at=None):
    profile_state = state.setdefault(profile, {})
    profile_state.setdefault(signature, {})
    if post_time is None:
        post_time = _utc_now()
    profile_state[signature]["last_post_at"] = post_time.isoformat()
    profile_state[signature]["last_status"] = status
    if post_id:
        profile_state[signature]["last_post_id"] = post_id
    if next_allowed_at:
        profile_state[signature]["next_allowed_at"] = next_allowed_at.isoformat()
    else:
        profile_state[signature].pop("next_allowed_at", None)


def _record_rate_limit(state, profile, signature, next_allowed_at):
    profile_state = state.setdefault(profile, {})
    profile_state.setdefault(signature, {})
    profile_state[signature]["last_status"] = "rate_limited"
    if next_allowed_at:
        profile_state[signature]["next_allowed_at"] = next_allowed_at.isoformat()


def _record_failure(state, profile, signature, status, error=None, next_allowed_at=None):
    profile_state = state.setdefault(profile, {})
    profile_state.setdefault(signature, {})
    profile_state[signature]["last_status"] = status
    if error:
        profile_state[signature]["last_error"] = str(error)[:500]
    if next_allowed_at:
        profile_state[signature]["next_allowed_at"] = next_allowed_at.isoformat()


def _record_suspension(state, profile, signature, suspended_until):
    if not suspended_until:
        return
    profile_state = state.setdefault(profile, {})
    profile_state.setdefault(signature, {})
    profile_state[signature]["suspended_until"] = suspended_until.isoformat()


def _parse_suspension_until(error_text):
    if not error_text:
        return None
    payload = None
    if "HTTP" in error_text and ":" in error_text:
        payload = error_text.split(":", 1)[1].strip()
    if payload:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            hint = data.get("hint") or data.get("error") or ""
            if data.get("suspended_until"):
                parsed = _parse_iso(data.get("suspended_until"))
                if parsed:
                    return parsed
            if hint:
                match = re.search(
                    r"Suspension ends in\\s+(\\d+)\\s+(minute|minutes|hour|hours|day|days)",
                    hint,
                    re.IGNORECASE,
                )
                if match:
                    value = int(match.group(1))
                    unit = match.group(2).lower()
                    if unit.startswith("minute"):
                        return _utc_now() + timedelta(minutes=value)
                    if unit.startswith("hour"):
                        return _utc_now() + timedelta(hours=value)
                    if unit.startswith("day"):
                        return _utc_now() + timedelta(days=value)
    return None


def _post_mint(base_url, api_key, tick, amount, submolt, prefix, dry_run):
    payload = {"p": "mbc-20", "op": "mint", "tick": tick, "amt": str(amount)}
    payload_str = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    timestamp = _utc_now().strftime("%Y-%m-%d %H:%M:%S")
    title = f"mint task {timestamp}"
    content_lines = []
    if prefix:
        content_lines.append(prefix)
    content_lines.append(f"mint task {timestamp}")
    content_lines.append(payload_str)
    content = "\n".join(content_lines)
    body = {"submolt": submolt, "title": title, "content": content}
    if dry_run:
        return {"success": True, "dry_run": True, "body": body}, None
    return _request("POST", f"{base_url}/posts", api_key=api_key, json_body=body)


def _content_id_from_error(error_text):
    if not error_text:
        return None
    payload = error_text
    if "HTTP" in error_text and ":" in error_text:
        payload = error_text.split(":", 1)[1].strip()
    try:
        data = json.loads(payload)
    except (TypeError, json.JSONDecodeError):
        return None
    if isinstance(data, dict):
        return data.get("content_id") or data.get("post_id")
    return None


def _auto_verify(base_url, api_key, data, label):
    post = data.get("post") or data.get("data") or {}
    verification_required = data.get("verification_required") or post.get("verification_status") == "pending"
    if not verification_required:
        return True, None
    verification = data.get("verification") or {}
    code = verification.get("code") or verification.get("verification_code")
    challenge = verification.get("challenge")
    if not code or not challenge:
        return False, "verification required but missing code or challenge"
    print(f"Verification challenge: {challenge}")
    answer, provider, error, details = _solve_math_challenge(challenge)
    llm_answer = (details or {}).get("llm")
    local_answer = (details or {}).get("local")
    llm_provider = (details or {}).get("llm_provider") or provider
    expression = (details or {}).get("expression")
    if expression:
        print(f"Verification expression: {expression}")
    if llm_answer and local_answer and llm_answer != local_answer:
        print(f"Verification LLM answer ({llm_provider}): {llm_answer}")
        print(f"Verification local answer: {local_answer}")
        print("Using local answer to avoid incorrect verification.")
    if not answer:
        return False, error or "unable to solve verification challenge"
    answer = _format_answer(answer)
    provider_label = provider or "solver"
    print(f"Verification answer ({provider_label}): {answer}")
    payload = {"verification_code": code, "answer": answer}
    verify_data, error = _request("POST", f"{base_url}/verify", api_key=api_key, json_body=payload)
    if not verify_data:
        content_id = _content_id_from_error(error)
        if content_id:
            print(f"\"content_id\": \"{content_id}\"")
            _record_content_id(content_id, label=label, content_type="post")
        return False, error
    content_id = verify_data.get("content_id")
    if content_id:
        print(f"\"content_id\": \"{content_id}\"")
        _record_content_id(content_id, label=label, content_type=verify_data.get("content_type"))
    return True, None


def _ensure_claimed(base_url, api_key):
    data, error = _request("GET", f"{base_url}/agents/status", api_key=api_key)
    if not data:
        return False, error
    if data.get("status") != "claimed":
        return False, f"status={data.get('status')}"
    return True, None


def _run_once(base_url, profiles, state, args):
    next_times = []
    only = None
    if args.only:
        only = {name.strip() for name in args.only.split(",") if name.strip()}

    for name, profile in profiles.items():
        if only and name not in only:
            continue
        api_key = profile.get("api_key")
        if not api_key:
            print(f"[{name}] missing api_key, skipping")
            continue

        signature = _signature_for(args.tick, args.amount, args.submolt, args.prefix)
        legacy_key = str(args.tick).lower()
        state_last, state_next, _sent_once, suspended_until = _get_state_times(
            state, name, signature, legacy_key=legacy_key
        )
        now = _utc_now()

        next_allowed_local = _combine_next_allowed(state_last, state_next, args.interval_minutes)
        if next_allowed_local and now < next_allowed_local:
            print(f"[{name}] next allowed at {next_allowed_local.isoformat()}")
            next_times.append(next_allowed_local)
            continue

        if suspended_until and now < suspended_until:
            print(f"[{name}] suspended until {suspended_until.isoformat()}")
            next_times.append(suspended_until)
            continue

        remote_last = None
        agent_name = None
        agent_name, error = _get_agent_name(base_url, api_key)
        if error:
            suspended_until = _parse_suspension_until(error)
            if suspended_until:
                _record_suspension(state, name, signature, suspended_until)
                print(f"[{name}] suspended until {suspended_until.isoformat()}")
                next_times.append(suspended_until)
                continue
            backoff_at = now + timedelta(minutes=args.interval_minutes)
            _record_failure(state, name, signature, "agent_lookup_failed", error=error, next_allowed_at=backoff_at)
            print(f"[{name}] unable to fetch agent name: {error}")
            print(f"[{name}] retry after {backoff_at.isoformat()}")
            next_times.append(backoff_at)
            continue
        elif agent_name:
            remote_last, error, profile_data = _get_remote_last_post_time(base_url, api_key, agent_name)
            if error:
                suspended_until = _parse_suspension_until(error)
                if suspended_until:
                    _record_suspension(state, name, signature, suspended_until)
                    print(f"[{name}] suspended until {suspended_until.isoformat()}")
                    next_times.append(suspended_until)
                    continue
                backoff_at = now + timedelta(minutes=args.interval_minutes)
                _record_failure(state, name, signature, "profile_fetch_failed", error=error, next_allowed_at=backoff_at)
                print(f"[{name}] unable to fetch last post: {error}")
                print(f"[{name}] retry after {backoff_at.isoformat()}")
                next_times.append(backoff_at)
                continue

        last_post = _pick_last_post(state_last, remote_last)
        next_allowed = _combine_next_allowed(last_post, state_next, args.interval_minutes)
        if next_allowed and now < next_allowed:
            print(f"[{name}] next allowed at {next_allowed.isoformat()}")
            next_times.append(next_allowed)
            continue

        if args.require_claimed:
            ok, error = _ensure_claimed(base_url, api_key)
            if not ok:
                suspended_until = _parse_suspension_until(error)
                if suspended_until:
                    _record_suspension(state, name, signature, suspended_until)
                    print(f"[{name}] suspended until {suspended_until.isoformat()}")
                    next_times.append(suspended_until)
                    continue
                backoff_at = now + timedelta(minutes=args.interval_minutes)
                _record_failure(state, name, signature, "not_claimed", error=error, next_allowed_at=backoff_at)
                print(f"[{name}] not claimed ({error}), skipping")
                print(f"[{name}] retry after {backoff_at.isoformat()}")
                next_times.append(backoff_at)
                continue

        data, error = _post_mint(
            base_url,
            api_key,
            args.tick,
            args.amount,
            args.submolt,
            args.prefix,
            args.dry_run,
        )
        if not data:
            retry_minutes = _extract_retry_after_minutes(error)
            if retry_minutes:
                retry_at = _utc_now() + timedelta(minutes=retry_minutes)
                _record_rate_limit(state, name, signature, retry_at)
                print(f"[{name}] rate limited, next allowed at {retry_at.isoformat()}")
                next_times.append(retry_at)
                continue
            suspended_until = _parse_suspension_until(error)
            if suspended_until:
                _record_suspension(state, name, signature, suspended_until)
                print(f"[{name}] suspended until {suspended_until.isoformat()}")
                next_times.append(suspended_until)
                continue
            print(f"[{name}] post failed: {error}")
            backoff_at = now + timedelta(minutes=args.interval_minutes)
            _record_failure(state, name, signature, "post_failed", error=error, next_allowed_at=backoff_at)
            print(f"[{name}] retry after {backoff_at.isoformat()}")
            next_times.append(backoff_at)
            continue

        if args.dry_run:
            print(f"[{name}] dry run: would post {args.tick} amt {args.amount}")
            next_times.append(_utc_now() + timedelta(minutes=args.interval_minutes))
            continue

        post = data.get("post") or data.get("data") or {}
        post_id = post.get("id")
        post_time = _parse_iso(post.get("created_at")) or _utc_now()
        verified, verify_error = _auto_verify(base_url, api_key, data, name)
        if verified:
            print(f"[{name}] posted (id={post_id})")
            _record_post(state, name, signature, "posted", post_id=post_id, post_time=post_time)
        else:
            print(f"[{name}] posted but verification failed: {verify_error}")
            _record_post(state, name, signature, "verify_failed", post_id=post_id, post_time=post_time)
        next_times.append(post_time + timedelta(minutes=args.interval_minutes))

    _save_json(args.state, state)
    return next_times


def _sleep_seconds(next_times, min_sleep_seconds):
    now = _utc_now()
    if not next_times:
        return min_sleep_seconds
    earliest = min(next_times)
    delta = (earliest - now).total_seconds()
    if delta < min_sleep_seconds:
        return min_sleep_seconds
    return delta


def main():
    parser = argparse.ArgumentParser(description="Auto-mint MBC-20 for all profiles")
    parser.add_argument("--tick", required=True, help="token ticker")
    parser.add_argument("--amount", required=True, help="mint amount")
    parser.add_argument("--submolt", default="mbc20")
    parser.add_argument("--prefix", default="mbc20.xyz")
    parser.add_argument("--interval-minutes", type=int, default=30)
    parser.add_argument("--profiles", default=PROFILES_PATH, help="profiles.json path")
    parser.add_argument("--state", default=DEFAULT_STATE_PATH, help="state file path")
    parser.add_argument("--only", help="comma-separated profile names to include")
    parser.add_argument("--require-claimed", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--loop", action="store_true", help="run continuously")
    parser.add_argument("--min-sleep-seconds", type=int, default=10)
    args = parser.parse_args()

    base_url = os.getenv("MOLTBOOK_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    if base_url.startswith("https://moltbook.com"):
        print("Error: use https://www.moltbook.com to avoid auth header stripping")
        return 1

    profiles = _load_profiles(args.profiles)
    if not profiles:
        print("No profiles found. Use the wizard to save API keys.")
        return 1

    state = _load_json(args.state, {})
    while True:
        next_times = _run_once(base_url, profiles, state, args)
        if not args.loop:
            break
        sleep_seconds = _sleep_seconds(next_times, max(1, args.min_sleep_seconds))
        next_check = _utc_now() + timedelta(seconds=sleep_seconds)
        print(f"Next check at {next_check.isoformat()} (sleep {int(sleep_seconds)}s)")
        time.sleep(sleep_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
