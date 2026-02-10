#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime, timedelta, timezone

import requests

DEFAULT_BASE_URL = "https://www.moltbook.com/api/v1"
PROFILE_DIR = os.path.join(os.path.expanduser("~"), ".config", "moltbook-wizard")
PROFILES_PATH = os.path.join(PROFILE_DIR, "profiles.json")


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


def _load_profiles(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return data.get("profiles", {})


def _request(method, url, api_key=None):
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        resp = requests.request(method, url, headers=headers, timeout=20)
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


def main():
    parser = argparse.ArgumentParser(description="Show next allowed post time")
    parser.add_argument("--api-key", help="Moltbook API key")
    parser.add_argument("--profile", help="profile name from wizard")
    parser.add_argument("--profiles", default=PROFILES_PATH, help="profiles.json path")
    parser.add_argument("--interval-minutes", type=int, default=120)
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("MOLTBOOK_API_KEY")
    if not api_key and args.profile:
        profiles = _load_profiles(args.profiles)
        api_key = profiles.get(args.profile, {}).get("api_key")
    if not api_key:
        print("Missing API key. Use --api-key, MOLTBOOK_API_KEY, or --profile.")
        return 1

    base_url = os.getenv("MOLTBOOK_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    if base_url.startswith("https://moltbook.com"):
        print("Error: use https://www.moltbook.com to avoid auth header stripping")
        return 1

    me, error = _request("GET", f"{base_url}/agents/me", api_key=api_key)
    if not me:
        print(f"Failed to fetch agent: {error}")
        return 1
    agent = me.get("agent", me)
    name = agent.get("name")
    if not name:
        print("Could not determine agent name.")
        return 1

    profile_data, error = _request("GET", f"{base_url}/agents/profile?name={name}", api_key=api_key)
    if not profile_data:
        print(f"Failed to fetch profile: {error}")
        return 1

    last_post = _latest_post_time(profile_data)
    now = _utc_now()
    if not last_post:
        print(f"No posts found for {name}. You can post now.")
        return 0

    next_allowed = last_post + timedelta(minutes=args.interval_minutes)
    remaining = next_allowed - now
    remaining_seconds = int(remaining.total_seconds())
    if remaining_seconds < 0:
        remaining_seconds = 0

    hours, rem = divmod(remaining_seconds, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f"Agent: {name}")
    print(f"Last post: {last_post.isoformat()}")
    print(f"Next allowed: {next_allowed.isoformat()}")
    print(f"Time remaining: {hours}h {minutes}m {seconds}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
