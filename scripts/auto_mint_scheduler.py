#!/usr/bin/env python3
import argparse
import ast
import json
import os
import re
from datetime import datetime, timedelta, timezone

import requests

DEFAULT_BASE_URL = "https://www.moltbook.com/api/v1"
PROFILE_DIR = os.path.join(os.path.expanduser("~"), ".config", "moltbook-wizard")
PROFILES_PATH = os.path.join(PROFILE_DIR, "profiles.json")
DEFAULT_STATE_PATH = os.path.join(PROFILE_DIR, "auto_mint_state.json")

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
}


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


def _load_profiles(path):
    data = _load_json(path, {"active": None, "profiles": {}})
    return data.get("profiles", {})


def _normalize_word(word):
    return re.sub(r"(.)\\1+", r"\\1", word)


def _words_to_number(words):
    total = 0
    current = 0
    for word in words:
        if word == "and":
            continue
        if word in _SMALL_NUMBERS:
            current += _SMALL_NUMBERS[word]
        elif word in _TENS_NUMBERS:
            current += _TENS_NUMBERS[word]
        elif word == "hundred":
            current = max(1, current) * _SCALE_NUMBERS[word]
        elif word in ("thousand", "million"):
            scale = _SCALE_NUMBERS[word]
            total += max(1, current) * scale
            current = 0
    return total + current


def _challenge_to_expr(challenge):
    if not challenge:
        return "", [], False
    text = challenge.lower().replace("-", " ")
    tokens = re.findall(r"[a-z]+|\d+(?:\.\d+)?|[+*/()\-]", text)
    out = []
    numbers = []
    has_operator = False
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in _OP_WORDS:
            out.append(_OP_WORDS[token])
            has_operator = True
            i += 1
            continue
        if re.match(r"\d", token):
            out.append(token)
            try:
                numbers.append(float(token))
            except ValueError:
                pass
            i += 1
            continue
        if token.isalpha():
            token = _normalize_word(token)
        if token in _NUMBER_WORDS or token in _SCALE_NUMBERS or token == "and":
            words = []
            j = i
            while j < len(tokens):
                nxt = tokens[j]
                if nxt.isalpha():
                    nxt = _normalize_word(nxt)
                if nxt in _NUMBER_WORDS or nxt in _SCALE_NUMBERS or nxt == "and":
                    words.append(nxt)
                    j += 1
                    continue
                break
            value = _words_to_number(words)
            out.append(str(value))
            numbers.append(float(value))
            i = j
            continue
        if token in "+-*/()":
            out.append(token)
            has_operator = True
            i += 1
            continue
        i += 1
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


def _solve_math_challenge(challenge):
    expr, numbers, has_operator = _challenge_to_expr(challenge)
    if not expr and not numbers:
        return None
    if expr:
        try:
            result = _safe_eval(expr)
            return f"{result:.2f}"
        except Exception:
            pass
    challenge_lower = (challenge or "").lower()
    if numbers and (not has_operator) and any(hint in challenge_lower for hint in _SUM_HINTS):
        return f"{sum(numbers):.2f}"
    return None


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


def _should_post(state, profile, tick, interval_minutes):
    profile_state = state.get(profile, {})
    key = tick.lower()
    last_post_at = _parse_iso(profile_state.get(key, {}).get("last_post_at"))
    if not last_post_at:
        return True, None
    next_allowed = last_post_at + timedelta(minutes=interval_minutes)
    if _utc_now() >= next_allowed:
        return True, None
    return False, next_allowed


def _record_post(state, profile, tick, status, post_id=None):
    profile_state = state.setdefault(profile, {})
    key = tick.lower()
    profile_state.setdefault(key, {})
    profile_state[key]["last_post_at"] = _utc_now().isoformat()
    profile_state[key]["last_status"] = status
    if post_id:
        profile_state[key]["last_post_id"] = post_id


def _post_mint(base_url, api_key, tick, amount, submolt, prefix, dry_run):
    payload = {"p": "mbc-20", "op": "mint", "tick": tick, "amt": str(amount)}
    payload_str = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    content = f"{prefix}\n{payload_str}" if prefix else payload_str
    title = f"mbc-20 mint {tick}"
    body = {"submolt": submolt, "title": title, "content": content}
    if dry_run:
        return {"success": True, "dry_run": True, "body": body}, None
    return _request("POST", f"{base_url}/posts", api_key=api_key, json_body=body)


def _auto_verify(base_url, api_key, data):
    post = data.get("post") or data.get("data") or {}
    verification_required = data.get("verification_required") or post.get("verification_status") == "pending"
    if not verification_required:
        return True, None
    verification = data.get("verification") or {}
    code = verification.get("code") or verification.get("verification_code")
    challenge = verification.get("challenge")
    if not code or not challenge:
        return False, "verification required but missing code or challenge"
    answer = _solve_math_challenge(challenge)
    if not answer:
        return False, "unable to solve verification challenge"
    payload = {"verification_code": code, "answer": answer}
    verify_data, error = _request("POST", f"{base_url}/verify", api_key=api_key, json_body=payload)
    if not verify_data:
        return False, error
    return True, None


def _ensure_claimed(base_url, api_key):
    data, error = _request("GET", f"{base_url}/agents/status", api_key=api_key)
    if not data:
        return False, error
    if data.get("status") != "claimed":
        return False, f"status={data.get('status')}"
    return True, None


def main():
    parser = argparse.ArgumentParser(description="Auto-mint MBC-20 for all profiles")
    parser.add_argument("--tick", required=True, help="token ticker")
    parser.add_argument("--amount", required=True, help="mint amount")
    parser.add_argument("--submolt", default="mbc20")
    parser.add_argument("--prefix", default="mbc20.xyz")
    parser.add_argument("--interval-minutes", type=int, default=120)
    parser.add_argument("--profiles", default=PROFILES_PATH, help="profiles.json path")
    parser.add_argument("--state", default=DEFAULT_STATE_PATH, help="state file path")
    parser.add_argument("--only", help="comma-separated profile names to include")
    parser.add_argument("--require-claimed", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_url = os.getenv("MOLTBOOK_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    if base_url.startswith("https://moltbook.com"):
        print("Error: use https://www.moltbook.com to avoid auth header stripping")
        return 1

    profiles = _load_profiles(args.profiles)
    if not profiles:
        print("No profiles found. Use the wizard to save API keys.")
        return 1

    only = None
    if args.only:
        only = {name.strip() for name in args.only.split(",") if name.strip()}

    state = _load_json(args.state, {})

    for name, profile in profiles.items():
        if only and name not in only:
            continue
        api_key = profile.get("api_key")
        if not api_key:
            print(f"[{name}] missing api_key, skipping")
            continue
        allowed, next_allowed = _should_post(state, name, args.tick, args.interval_minutes)
        if not allowed:
            print(f"[{name}] next allowed at {next_allowed.isoformat()}")
            continue
        if args.require_claimed:
            ok, error = _ensure_claimed(base_url, api_key)
            if not ok:
                print(f"[{name}] not claimed ({error}), skipping")
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
            print(f"[{name}] post failed: {error}")
            _record_post(state, name, args.tick, "post_failed")
            continue
        if args.dry_run:
            print(f"[{name}] dry run: would post {args.tick} amt {args.amount}")
            continue
        post = data.get("post") or data.get("data") or {}
        post_id = post.get("id")
        verified, verify_error = _auto_verify(base_url, api_key, data)
        if verified:
            print(f"[{name}] posted (id={post_id})")
            _record_post(state, name, args.tick, "posted", post_id=post_id)
        else:
            print(f"[{name}] posted but verification failed: {verify_error}")
            _record_post(state, name, args.tick, "verify_failed", post_id=post_id)

    _save_json(args.state, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
