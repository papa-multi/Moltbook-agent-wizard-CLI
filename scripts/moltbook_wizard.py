#!/usr/bin/env python3
import json
import os
import re
import sys
from getpass import getpass

import requests

DEFAULT_BASE_URL = "https://www.moltbook.com/api/v1"
PROFILE_DIR = os.path.join(os.path.expanduser("~"), ".config", "moltbook-wizard")
PROFILES_PATH = os.path.join(PROFILE_DIR, "profiles.json")


class ExitWizard(Exception):
    pass


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


def _pretty_json(data):
    print(json.dumps(data, indent=2, sort_keys=True))


def _handle_verification(base_url, api_key, data):
    post = data.get("post") or data.get("data") or {}
    verification_required = data.get("verification_required") or post.get("verification_status") == "pending"
    if not verification_required:
        return

    verification = data.get("verification") or {}
    code = verification.get("code") or verification.get("verification_code")
    challenge = verification.get("challenge")
    expires_at = verification.get("expires_at")
    print("\nVerification required for this post.")
    if challenge:
        print(f"Challenge: {challenge}")
    if expires_at:
        print(f"Expires at: {expires_at}")
    if not code:
        print("Verification code missing in response.")
        return

    answer = _prompt("Answer (2 decimals, e.g., 75.00)", required=True)
    payload = {"verification_code": code, "answer": answer}
    verify_data = _request("POST", f"{base_url}/verify", api_key=api_key, json_body=payload)
    if verify_data:
        print("Verification response:")
        _pretty_json(verify_data)


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
    code = _prompt("Verification code")
    answer = _prompt("Answer (2 decimals, e.g., 75.00)")
    payload = {"verification_code": code, "answer": answer}
    data = _request("POST", f"{base_url}/verify", api_key=api_key, json_body=payload)
    if data:
        _pretty_json(data)


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
    print("7) Manage accounts")
    print("8) Exit")
    return _prompt("Select", required=True)


def main():
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
                _cmd_accounts(state)
            elif choice == "8":
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
