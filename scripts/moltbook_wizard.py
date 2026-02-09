#!/usr/bin/env python3
import ast
import json
import os
import re
import shlex
import sys
import subprocess
from datetime import datetime, timezone
from getpass import getpass

import requests

DEFAULT_BASE_URL = "https://www.moltbook.com/api/v1"
MBC20_TOKEN_URLS = (
    "https://mbc20.xyz/api/stats",
    "https://mbc20.xyz/api/tokens",
    "https://mbc20.xyz/api/tokens?limit=100",
)
PROFILE_DIR = os.path.join(os.path.expanduser("~"), ".config", "moltbook-wizard")
PROFILES_PATH = os.path.join(PROFILE_DIR, "profiles.json")
VERIFICATION_LOG_PATH = os.path.join(PROFILE_DIR, "verification_log.json")

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


class ExitWizard(Exception):
    pass


def _configure_stdio():
    try:
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


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


def _print_header(text):
    print("\n" + "=" * len(text))
    print(text)
    print("=" * len(text))


def _prompt(text, default=None, required=True, secret=False):
    while True:
        label = text
        if default is not None:
            label = f"{text} [{default}]"
        label += ": "
        try:
            value = getpass(label) if secret else input(label)
        except UnicodeDecodeError:
            print("Input encoding error. Please enter plain UTF-8 text.")
            continue
        except (EOFError, KeyboardInterrupt):
            raise ExitWizard()
        value = value.strip()
        if not value and default is not None:
            return str(default)
        if value:
            return value
        if not required:
            return ""
        print("Please enter a value.")


def _prompt_yes_no(text, default=True):
    suffix = "Y/n" if default else "y/N"
    while True:
        value = input(f"{text} ({suffix}): ").strip().lower()
        if not value:
            return default
        if value in ("y", "yes"):
            return True
        if value in ("n", "no"):
            return False
        print("Please enter y or n.")


def _prompt_multiline(text):
    print(text)
    print("Finish with an empty line.")
    lines = []
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            raise ExitWizard()
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _get_base_url():
    base_url = os.getenv("MOLTBOOK_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    if base_url.startswith("https://moltbook.com"):
        print("Error: use https://www.moltbook.com to avoid auth header stripping")
        raise ExitWizard()
    return base_url


def _load_profiles():
    if not os.path.exists(PROFILES_PATH):
        return {"active": None, "profiles": {}}
    try:
        with open(PROFILES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"active": None, "profiles": {}}
    if not isinstance(data, dict):
        return {"active": None, "profiles": {}}
    data.setdefault("active", None)
    data.setdefault("profiles", {})
    return data


def _save_profiles(data):
    os.makedirs(PROFILE_DIR, exist_ok=True)
    with open(PROFILES_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.chmod(PROFILES_PATH, 0o600)


def _load_verification_log():
    if not os.path.exists(VERIFICATION_LOG_PATH):
        return []
    try:
        with open(VERIFICATION_LOG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(data, list):
        return data
    return []


def _save_verification_log(entries):
    os.makedirs(PROFILE_DIR, exist_ok=True)
    with open(VERIFICATION_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, sort_keys=True)
    os.chmod(VERIFICATION_LOG_PATH, 0o600)


def _record_verification(entry):
    if not entry:
        return
    log = _load_verification_log()
    code = entry.get("code")
    if code:
        for existing in log:
            if existing.get("code") == code:
                existing.update(entry)
                _save_verification_log(log)
                return
    log.insert(0, entry)
    _save_verification_log(log[:50])


def _update_verification_status(code, status, answer=None):
    if not code:
        return
    log = _load_verification_log()
    updated = False
    for entry in log:
        if entry.get("code") == code:
            entry["status"] = status
            entry["verified_at"] = _utc_now().isoformat()
            if answer is not None:
                entry["answer"] = answer
            updated = True
            break
    if updated:
        _save_verification_log(log)


def _pending_verification_entries():
    log = _load_verification_log()
    if not log:
        return []
    now = _utc_now()
    changed = False
    pending = []
    for entry in log:
        status = entry.get("status", "pending")
        expires_at = _parse_iso(entry.get("expires_at"))
        if status == "pending" and expires_at and now > expires_at:
            entry["status"] = "expired"
            changed = True
            continue
        if entry.get("status", "pending") == "pending":
            pending.append(entry)
    if changed:
        _save_verification_log(log)
    return pending


def _choose_profile_name(profiles, prompt_text="Select profile"):
    names = sorted(profiles.keys())
    if not names:
        return None
    print("Available profiles:")
    for idx, name in enumerate(names, 1):
        print(f"{idx}) {name}")
    choice = _prompt(prompt_text, required=True)
    if choice.isdigit():
        index = int(choice)
        if 1 <= index <= len(names):
            return names[index - 1]
    if choice in profiles:
        return choice
    print("Invalid selection.")
    return None


def _get_api_key(state):
    api_key = state.get("api_key") or os.getenv("MOLTBOOK_API_KEY")
    if api_key:
        return api_key
    profiles = _load_profiles()
    profiles_map = profiles.get("profiles", {})
    active = profiles.get("active")
    if active and active in profiles_map:
        api_key = profiles_map[active].get("api_key")
        if api_key:
            state["api_key"] = api_key
            state["profile_name"] = active
            return api_key
    if profiles_map:
        name = _choose_profile_name(profiles_map)
        if name:
            profiles["active"] = name
            _save_profiles(profiles)
            api_key = profiles_map[name].get("api_key")
            if api_key:
                state["api_key"] = api_key
                state["profile_name"] = name
                return api_key
    api_key = _prompt("Moltbook API key", secret=True)
    state["api_key"] = api_key
    if _prompt_yes_no("Save this API key as a profile?", default=True):
        profiles = _load_profiles()
        profile_name = _prompt("Profile name", default="default")
        profiles["profiles"][profile_name] = {"api_key": api_key}
        profiles["active"] = profile_name
        _save_profiles(profiles)
        state["profile_name"] = profile_name
    return api_key


def _request(method, url, api_key=None, json_body=None):
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if json_body is not None:
        headers["Content-Type"] = "application/json"
    try:
        resp = requests.request(method, url, headers=headers, json=json_body, timeout=30)
    except requests.RequestException as exc:
        print(f"Request failed: {exc}")
        return None
    if resp.status_code not in (200, 201):
        print(f"HTTP {resp.status_code}: {resp.text}")
        return None
    try:
        return resp.json()
    except ValueError:
        print("Response was not JSON")
        return None


def _request_public(url, silent=False):
    try:
        resp = requests.get(url, timeout=30)
    except requests.RequestException as exc:
        if not silent:
            print(f"Request failed: {exc}")
        return None, str(exc)
    if resp.status_code not in (200, 201):
        error = f"HTTP {resp.status_code}: {resp.text}"
        if not silent:
            print(error)
        return None, error
    try:
        return resp.json(), None
    except ValueError:
        error = "Response was not JSON"
        if not silent:
            print(error)
        return None, error


def _pretty_json(data):
    print(json.dumps(data, indent=2, sort_keys=True))


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
        return None, "empty challenge"
    if expr:
        try:
            result = _safe_eval(expr)
            return f"{result:.2f}", None
        except Exception:
            pass
    challenge_lower = (challenge or "").lower()
    if numbers and (not has_operator) and any(hint in challenge_lower for hint in _SUM_HINTS):
        return f"{sum(numbers):.2f}", None
    return None, "unable to solve challenge automatically"


def _handle_verification(base_url, api_key, data):
    post = data.get("post") or data.get("data") or {}
    verification_required = data.get("verification_required") or post.get("verification_status") == "pending"
    if not verification_required:
        return

    verification = data.get("verification") or {}
    code = verification.get("code") or verification.get("verification_code")
    challenge = verification.get("challenge")
    expires_at = verification.get("expires_at")
    _record_verification(
        {
            "code": code,
            "challenge": challenge,
            "expires_at": expires_at,
            "post_id": post.get("id"),
            "created_at": post.get("created_at"),
            "status": "pending",
        }
    )
    print("\nVerification required for this post.")
    if challenge:
        print(f"Challenge: {challenge}")
    if expires_at:
        print(f"Expires at: {expires_at}")
    if not code:
        print("Verification code missing in response.")
        return

    auto_answer = None
    auto_error = None
    if challenge:
        auto_answer, auto_error = _solve_math_challenge(challenge)
        if auto_answer:
            print(f"Auto-solved answer: {auto_answer}")
            if _prompt_yes_no("Submit this answer now?", default=True):
                payload = {"verification_code": code, "answer": auto_answer}
                verify_data = _request("POST", f"{base_url}/verify", api_key=api_key, json_body=payload)
                if verify_data:
                    print("Verification response:")
                    _pretty_json(verify_data)
                    _update_verification_status(code, "verified", answer=auto_answer)
                else:
                    _update_verification_status(code, "failed", answer=auto_answer)
                return
        elif auto_error:
            print(f"Auto-solve failed: {auto_error}")

    answer = _prompt("Answer (2 decimals, e.g., 75.00)", required=True)
    payload = {"verification_code": code, "answer": answer}
    verify_data = _request("POST", f"{base_url}/verify", api_key=api_key, json_body=payload)
    if verify_data:
        print("Verification response:")
        _pretty_json(verify_data)
        _update_verification_status(code, "verified", answer=answer)
    else:
        _update_verification_status(code, "failed", answer=answer)


def _cmd_register(base_url, state):
    _print_header("Create a Moltbook agent")
    name = _prompt("Agent name")
    description = _prompt("Agent description")
    payload = {"name": name, "description": description}
    data = _request("POST", f"{base_url}/agents/register", json_body=payload)
    if not data:
        return
    _pretty_json(data)
    agent = data.get("agent", {})
    api_key = agent.get("api_key")
    claim_url = agent.get("claim_url")
    if api_key:
        print("\nSave your API key now. You need it for all requests.")
        if _prompt_yes_no("Save this API key as a profile?", default=True):
            profiles = _load_profiles()
            profile_name = _prompt("Profile name", default=agent.get("name") or "default")
            profiles["profiles"][profile_name] = {"api_key": api_key}
            profiles["active"] = profile_name
            _save_profiles(profiles)
            state["api_key"] = api_key
            state["profile_name"] = profile_name
    if claim_url:
        print(f"Claim URL: {claim_url}")


def _cmd_status(base_url, state):
    _print_header("Check claim status")
    api_key = _get_api_key(state)
    data = _request("GET", f"{base_url}/agents/status", api_key=api_key)
    if data:
        _pretty_json(data)


def _cmd_post(base_url, state):
    _print_header("Create a post")
    api_key = _get_api_key(state)
    submolt = _prompt("Submolt", default="general")
    title = _prompt("Title")
    content = _prompt_multiline("Enter your post content")
    payload = {"submolt": submolt, "title": title, "content": content}
    data = _request("POST", f"{base_url}/posts", api_key=api_key, json_body=payload)
    if data:
        _pretty_json(data)
        _handle_verification(base_url, api_key, data)


def _cmd_mint(base_url, state):
    _print_header("Mint an MBC-20 inscription")
    api_key = _get_api_key(state)
    mode = _prompt("Entry mode: 1) paste full JSON/content  2) guided", default="1")
    if mode == "1":
        content = _prompt_multiline(
            "Paste mbc20.xyz + JSON, or just the JSON inscription"
        )
        if not content:
            print("No content provided.")
            return
        mint_info = _parse_mbc20_mint(content)
        if not mint_info:
            print("Could not find a valid MBC-20 mint JSON in your input.")
            return
        tick = mint_info.get("tick") or ""
        amt = mint_info.get("amt") or ""
        print(f"Detected mint: tick={tick} amt={amt}")
        if not _prompt_yes_no("Proceed with this mint?", default=True):
            return
        if "mbc20.xyz" not in content and _prompt_yes_no(
            "Add mbc20.xyz prefix?", default=True
        ):
            content = f"mbc20.xyz\n{content}"
        submolt = _prompt("Submolt", default="mbc20")
        title = f"mbc-20 mint {tick}".strip()
        body = {"submolt": submolt, "title": title, "content": content}
        data = _request("POST", f"{base_url}/posts", api_key=api_key, json_body=body)
        if data:
            _pretty_json(data)
            _handle_verification(base_url, api_key, data)
        return
    if mode != "2":
        print("Please choose 1 or 2.")
        return
    tick = _prompt("Token ticker (tick), e.g., GPT")
    amount = _prompt("Mint amount")
    submolt = _prompt("Submolt", default="mbc20")
    include_prefix = _prompt_yes_no("Include mbc20.xyz prefix?", default=True)
    payload = {"p": "mbc-20", "op": "mint", "tick": tick, "amt": str(amount)}
    payload_str = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    content = f"mbc20.xyz\n{payload_str}" if include_prefix else payload_str
    title = f"mbc-20 mint {tick}"
    body = {"submolt": submolt, "title": title, "content": content}
    data = _request("POST", f"{base_url}/posts", api_key=api_key, json_body=body)
    if data:
        _pretty_json(data)
        _handle_verification(base_url, api_key, data)


def _cmd_link(base_url, state):
    _print_header("Link a wallet (optional)")
    api_key = _get_api_key(state)
    address = _prompt("Wallet address (0x...)")
    if not re.match(r"^0x[a-fA-F0-9]{40}$", address):
        print("Warning: address format looks unusual, double-check it.")
    submolt = _prompt("Submolt", default="mbc20")
    include_prefix = _prompt_yes_no("Include mbc20.xyz prefix?", default=True)
    payload = {"p": "mbc-20", "op": "link", "addr": address}
    payload_str = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    content = f"mbc20.xyz\n{payload_str}" if include_prefix else payload_str
    body = {"submolt": submolt, "title": "mbc-20 link wallet", "content": content}
    data = _request("POST", f"{base_url}/posts", api_key=api_key, json_body=body)
    if data:
        _pretty_json(data)
        _handle_verification(base_url, api_key, data)


def _cmd_verify(base_url, state):
    _print_header("Verify a post")
    api_key = _get_api_key(state)
    pending = _pending_verification_entries()
    if pending:
        print("Saved challenges:")
        for idx, entry in enumerate(pending, 1):
            challenge = (entry.get("challenge") or "").strip()
            if len(challenge) > 60:
                challenge = challenge[:57] + "..."
            post_id = entry.get("post_id") or "unknown"
            expires_at = entry.get("expires_at") or "unknown"
            print(f"{idx}) {post_id} | expires {expires_at} | {challenge}")
        choice = _prompt("Select challenge number (or press Enter to enter manually)", required=False)
        if choice:
            if not choice.isdigit() or not (1 <= int(choice) <= len(pending)):
                print("Invalid selection.")
                return
            entry = pending[int(choice) - 1]
            code = entry.get("code")
            challenge = entry.get("challenge")
            if challenge:
                auto_answer, auto_error = _solve_math_challenge(challenge)
                if auto_answer:
                    print(f"Auto-solved answer: {auto_answer}")
                    if _prompt_yes_no("Submit this answer now?", default=True):
                        payload = {"verification_code": code, "answer": auto_answer}
                        data = _request("POST", f"{base_url}/verify", api_key=api_key, json_body=payload)
                        if data:
                            _pretty_json(data)
                            _update_verification_status(code, "verified", answer=auto_answer)
                        else:
                            _update_verification_status(code, "failed", answer=auto_answer)
                        return
                elif auto_error:
                    print(f"Auto-solve failed: {auto_error}")
            answer = _prompt("Answer (2 decimals, e.g., 75.00)")
            payload = {"verification_code": code, "answer": answer}
            data = _request("POST", f"{base_url}/verify", api_key=api_key, json_body=payload)
            if data:
                _pretty_json(data)
                _update_verification_status(code, "verified", answer=answer)
            else:
                _update_verification_status(code, "failed", answer=answer)
            return

    print("No saved challenges found.")
    if not _prompt_yes_no("Enter a verification code manually?", default=False):
        return
    code = _prompt("Verification code")
    if not re.fullmatch(r"[a-fA-F0-9]{64}", code):
        print("Warning: this does not look like a post verification code.")
        print("Post codes are 64 hex chars; claim codes like claw-XXXX won't work.")
        if not _prompt_yes_no("Continue anyway?", default=False):
            return
    answer = _prompt("Answer (2 decimals, e.g., 75.00)")
    payload = {"verification_code": code, "answer": answer}
    data = _request("POST", f"{base_url}/verify", api_key=api_key, json_body=payload)
    if data:
        _pretty_json(data)
        _update_verification_status(code, "verified", answer=answer)
    else:
        _update_verification_status(code, "failed", answer=answer)


def _extract_tokens(data):
    if not isinstance(data, dict):
        return None
    for key in ("tokens", "data", "stats"):
        value = data.get(key)
        if isinstance(value, dict):
            tokens = value.get("tokens")
            if isinstance(tokens, list):
                return tokens
        if isinstance(value, list) and key == "tokens":
            return value
    return None


def _format_number(value):
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if num.is_integer():
        return f"{int(num):,}"
    return f"{num:,.2f}"


def _cmd_tokens():
    _print_header("MBC-20 tokens (mbc20.xyz)")
    last_error = None
    for url in MBC20_TOKEN_URLS:
        data, error = _request_public(url, silent=True)
        if not data:
            last_error = error
            continue
        tokens = _extract_tokens(data)
        if tokens:
            print(f"Source: {url}")
            print(f"Found {len(tokens)} tokens. Showing up to 50.")
            print("tick | minted/max | holders | operations")
            print("-" * 50)
            for token in tokens[:50]:
                tick = token.get("tick") or token.get("ticker") or "?"
                minted = token.get("minted") or token.get("minted_amount") or token.get("supply") or ""
                max_supply = token.get("max") or token.get("max_supply") or token.get("supply_max") or ""
                holders = token.get("holders") or token.get("holder_count") or token.get("holders_count") or ""
                operations = token.get("operations") or token.get("operation_count") or token.get("ops") or ""
                minted_str = _format_number(minted)
                max_str = _format_number(max_supply)
                if minted_str and max_str:
                    minted_str = f"{minted_str}/{max_str}"
                print(f"{tick} | {minted_str} | {_format_number(holders)} | {_format_number(operations)}")
            return
        last_error = "No token list found in response"
    print("Unable to fetch token list right now.")
    if last_error:
        print(f"Last error: {last_error}")
    print("Tip: try again later or open https://mbc20.xyz/tokens in a browser.")


def _extract_posts(data):
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    for key in ("posts", "data", "results"):
        value = data.get(key)
        if isinstance(value, list):
            return value
    post = data.get("post")
    if isinstance(post, dict):
        return [post]
    return []


def _parse_mbc20_mint(content):
    if not content:
        return None
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    candidates = []
    for line in lines:
        if line.startswith("{") and line.endswith("}"):
            candidates.append(line)
    if not candidates:
        candidates = re.findall(r"\{[^{}]*\}", content)
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        if str(data.get("p", "")).lower() != "mbc-20":
            continue
        if str(data.get("op", "")).lower() != "mint":
            continue
        tick = data.get("tick") or data.get("nametoken") or data.get("token") or ""
        amt = data.get("amt") or data.get("amount") or ""
        return {"tick": str(tick), "amt": str(amt)}
    return None


def _cmd_my_mints(base_url, state):
    _print_header("Your MBC-20 mints (recent)")
    api_key = _get_api_key(state)
    me = _request("GET", f"{base_url}/agents/me", api_key=api_key)
    if not me:
        return
    agent = me.get("agent", me)
    agent_name = agent.get("name")
    if not agent_name:
        print("Unable to determine agent name.")
        return

    posts_data = _request(
        "GET",
        f"{base_url}/posts?submolt=mbc20&sort=new&limit=100",
        api_key=api_key,
    )
    if not posts_data:
        return
    posts = _extract_posts(posts_data)
    if not posts:
        print("No posts found.")
        return

    results = []
    for post in posts:
        author = post.get("author", {}) or {}
        if author.get("name") != agent_name:
            continue
        mint_info = _parse_mbc20_mint(post.get("content", ""))
        if not mint_info:
            continue
        url = post.get("url") or ""
        if url.startswith("/"):
            url = f"https://www.moltbook.com{url}"
        results.append(
            {
                "tick": mint_info.get("tick"),
                "amt": mint_info.get("amt"),
                "created_at": post.get("created_at"),
                "url": url,
                "verification_status": post.get("verification_status"),
                "post_id": post.get("id"),
            }
        )

    if not results:
        print("No mint posts found in the last 100 MBC-20 posts.")
        print("Tip: if you minted earlier, try again later or use the mbc20.xyz agent search.")
        return

    print(f"Found {len(results)} mint post(s) for {agent_name}.")
    print("Available mint IDs:")
    for item in results:
        post_id = item.get("post_id") or "unknown"
        tick = item.get("tick") or "?"
        created = item.get("created_at") or ""
        print(f"- {post_id} | {tick} | {created}")
    selection = _prompt("Show which ID? (type 'all' for everything)", default="all", required=False)
    selection = (selection or "all").strip()
    if selection.lower() != "all":
        filtered = [item for item in results if item.get("post_id") == selection]
        if not filtered:
            print("No mint found with that ID.")
            return
        results = filtered

    print("tick | amount | created_at | url | status")
    print("-" * 80)
    for item in results:
        tick = item.get("tick") or "?"
        amt = item.get("amt") or "?"
        created = item.get("created_at") or ""
        url = item.get("url") or ""
        status = item.get("verification_status") or "published"
        print(f"{tick} | {amt} | {created} | {url} | {status}")


def _cmd_accounts(state):
    _print_header("Manage accounts")
    profiles = _load_profiles()
    while True:
        active = profiles.get("active")
        names = sorted(profiles.get("profiles", {}).keys())
        print(f"Active profile: {active or 'none'}")
        print("1) Add account")
        print("2) Switch account")
        print("3) Remove account")
        print("4) Back")
        choice = _prompt("Select", required=True)
        if choice == "1":
            name = _prompt("Profile name")
            api_key = _prompt("Moltbook API key", secret=True)
            profiles.setdefault("profiles", {})[name] = {"api_key": api_key}
            profiles["active"] = name
            _save_profiles(profiles)
            state["api_key"] = api_key
            state["profile_name"] = name
        elif choice == "2":
            if not names:
                print("No profiles saved.")
                continue
            selected = _choose_profile_name(profiles.get("profiles", {}), prompt_text="Select profile")
            if selected:
                profiles["active"] = selected
                _save_profiles(profiles)
                api_key = profiles["profiles"][selected].get("api_key")
                if api_key:
                    state["api_key"] = api_key
                    state["profile_name"] = selected
        elif choice == "3":
            if not names:
                print("No profiles saved.")
                continue
            selected = _choose_profile_name(profiles.get("profiles", {}), prompt_text="Select profile to remove")
            if selected and _prompt_yes_no(f"Remove '{selected}'?", default=False):
                profiles["profiles"].pop(selected, None)
                if profiles.get("active") == selected:
                    profiles["active"] = sorted(profiles["profiles"].keys())[0] if profiles["profiles"] else None
                _save_profiles(profiles)
                if state.get("profile_name") == selected:
                    state.pop("profile_name", None)
                    state.pop("api_key", None)
        elif choice == "4":
            break
        else:
            print("Please choose a valid option.")


def _menu():
    print("\nWhat do you want to do?")
    print("1) Create a Moltbook agent")
    print("2) Check claim status")
    print("3) Create a post")
    print("4) Mint an MBC-20 inscription")
    print("5) Link a wallet (optional)")
    print("6) Verify a post challenge")
    print("7) Show my minted items")
    print("8) Show minted tokens (mbc20.xyz)")
    print("9) Manage accounts")
    print("10) Auto-mint every 2 hours (all accounts)")
    print("11) Next allowed request time")
    print("12) Exit")
    return _prompt("Select", required=True)


def _script_path(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def _run_script(args):
    try:
        result = subprocess.run(args, check=False)
    except OSError as exc:
        print(f"Failed to run script: {exc}")
        return 1
    return result.returncode


def _cmd_auto_mint():
    _print_header("Auto-mint every 2 hours (all accounts)")
    content = _prompt_multiline(
        "Paste mbc20.xyz + JSON, or just the JSON inscription"
    )
    if not content:
        print("No content provided.")
        return
    mint_info = _parse_mbc20_mint(content)
    if not mint_info:
        print("Could not find a valid MBC-20 mint JSON in your input.")
        return
    tick = mint_info.get("tick") or ""
    amount = mint_info.get("amt") or ""
    if not tick or not amount:
        print("Missing tick or amount in the pasted JSON.")
        return
    print(f"Auto-mint configured: tick={tick} amt={amount} interval=120m profiles=all")
    submolt = "mbc20"
    prefix = "mbc20.xyz"
    interval = "120"
    screen_name = f"auto-mint-{tick.lower()}"
    args = [
        sys.executable,
        _script_path("auto_mint_scheduler.py"),
        "--tick",
        tick,
        "--amount",
        amount,
        "--submolt",
        submolt,
        "--prefix",
        prefix,
        "--interval-minutes",
        interval,
        "--require-claimed",
    ]
    inner = " ".join(shlex.quote(part) for part in args)
    loop_cmd = f"while true; do {inner}; sleep {int(interval) * 60}; done"
    screen_cmd = ["screen", "-dmS", screen_name, "bash", "-lc", loop_cmd]
    try:
        result = subprocess.run(screen_cmd, check=False)
    except OSError as exc:
        print(f"Failed to start screen: {exc}")
        return
    if result.returncode != 0:
        print("Failed to start screen session. Is screen installed?")
        return
    print(f"Started screen session: {screen_name}")
    print(f"View: screen -r {screen_name}")
    print(f"Stop: screen -S {screen_name} -X quit")


def _cmd_next_request():
    _print_header("Next allowed request time")
    interval = "120"
    args = [
        sys.executable,
        _script_path("next_request.py"),
        "--interval-minutes",
        interval,
    ]
    profiles_data = _load_profiles()
    profiles = profiles_data.get("profiles", {})
    active = profiles_data.get("active")
    if active and active in profiles:
        args.extend(["--profile", active])
        _run_script(args)
        return
    if profiles:
        first = sorted(profiles.keys())[0]
        args.extend(["--profile", first])
        print(f"Using profile '{first}'. Set a different active profile in Manage accounts.")
        _run_script(args)
        return
    api_key = _prompt("Moltbook API key", secret=True)
    args.extend(["--api-key", api_key])
    _run_script(args)


def main():
    _configure_stdio()
    base_url = _get_base_url()
    state = {}
    while True:
        try:
            choice = _menu()
            if choice == "1":
                _cmd_register(base_url, state)
            elif choice == "2":
                _cmd_status(base_url, state)
            elif choice == "3":
                _cmd_post(base_url, state)
            elif choice == "4":
                _cmd_mint(base_url, state)
            elif choice == "5":
                _cmd_link(base_url, state)
            elif choice == "6":
                _cmd_verify(base_url, state)
            elif choice == "7":
                _cmd_my_mints(base_url, state)
            elif choice == "8":
                _cmd_tokens()
            elif choice == "9":
                _cmd_accounts(state)
            elif choice == "10":
                _cmd_auto_mint()
            elif choice == "11":
                _cmd_next_request()
            elif choice == "12":
                break
            else:
                print("Please choose a valid option.")
        except (ExitWizard, EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break


if __name__ == "__main__":
    try:
        main()
    except ExitWizard:
        sys.exit(1)
