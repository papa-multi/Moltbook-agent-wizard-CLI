#!/usr/bin/env python3
import argparse
import json
import os
import sys

import requests

DEFAULT_BASE_URL = "https://www.moltbook.com/api/v1"


def _get_api_key(args):
    return args.api_key or os.getenv("MOLTBOOK_API_KEY")


def _get_base_url(args):
    base_url = args.base_url or os.getenv("MOLTBOOK_BASE_URL") or DEFAULT_BASE_URL
    if base_url.startswith("https://moltbook.com"):
        print("Error: use https://www.moltbook.com to avoid auth header stripping", file=sys.stderr)
        sys.exit(1)
    return base_url.rstrip("/")


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
        sys.exit(1)
    if resp.status_code not in (200, 201):
        print(f"HTTP {resp.status_code}: {resp.text}")
        sys.exit(1)
    try:
        return resp.json()
    except ValueError:
        print("Response was not JSON")
        sys.exit(1)


def _print_json(data):
    print(json.dumps(data, indent=2, sort_keys=True))


def cmd_register(args):
    base_url = _get_base_url(args)
    payload = {"name": args.name, "description": args.description}
    data = _request("POST", f"{base_url}/agents/register", json_body=payload)
    _print_json(data)


def cmd_status(args):
    base_url = _get_base_url(args)
    api_key = _get_api_key(args)
    if not api_key:
        print("Missing API key. Set MOLTBOOK_API_KEY or pass --api-key.")
        sys.exit(1)
    data = _request("GET", f"{base_url}/agents/status", api_key=api_key)
    _print_json(data)


def _load_content(args):
    if args.content and args.content_file:
        print("Use --content or --content-file, not both.")
        sys.exit(1)
    if args.content_file:
        with open(args.content_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    if args.content:
        return args.content
    print("Missing content. Use --content or --content-file.")
    sys.exit(1)


def cmd_post(args):
    base_url = _get_base_url(args)
    api_key = _get_api_key(args)
    if not api_key:
        print("Missing API key. Set MOLTBOOK_API_KEY or pass --api-key.")
        sys.exit(1)
    content = _load_content(args)
    payload = {"submolt": args.submolt, "title": args.title, "content": content}
    if args.dry_run:
        _print_json(payload)
        return
    data = _request("POST", f"{base_url}/posts", api_key=api_key, json_body=payload)
    _print_json(data)


def _build_mbc20_content(prefix, payload):
    payload_str = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    if prefix:
        return f"{prefix}\n{payload_str}"
    return payload_str


def cmd_mint(args):
    base_url = _get_base_url(args)
    api_key = _get_api_key(args)
    if not api_key:
        print("Missing API key. Set MOLTBOOK_API_KEY or pass --api-key.")
        sys.exit(1)
    payload = {"p": "mbc-20", "op": "mint", "tick": args.tick, "amt": str(args.amount)}
    content = _build_mbc20_content(args.prefix, payload)
    title = args.title or f"mbc-20 mint {args.tick}"
    body = {"submolt": args.submolt, "title": title, "content": content}
    if args.dry_run:
        _print_json(body)
        return
    data = _request("POST", f"{base_url}/posts", api_key=api_key, json_body=body)
    _print_json(data)


def cmd_link(args):
    base_url = _get_base_url(args)
    api_key = _get_api_key(args)
    if not api_key:
        print("Missing API key. Set MOLTBOOK_API_KEY or pass --api-key.")
        sys.exit(1)
    payload = {"p": "mbc-20", "op": "link", "addr": args.address}
    content = _build_mbc20_content(args.prefix, payload)
    title = args.title or "mbc-20 link wallet"
    body = {"submolt": args.submolt, "title": title, "content": content}
    if args.dry_run:
        _print_json(body)
        return
    data = _request("POST", f"{base_url}/posts", api_key=api_key, json_body=body)
    _print_json(data)


def cmd_verify(args):
    base_url = _get_base_url(args)
    api_key = _get_api_key(args)
    if not api_key:
        print("Missing API key. Set MOLTBOOK_API_KEY or pass --api-key.")
        sys.exit(1)
    payload = {"verification_code": args.code, "answer": args.answer}
    data = _request("POST", f"{base_url}/verify", api_key=api_key, json_body=payload)
    _print_json(data)


def main():
    parser = argparse.ArgumentParser(description="Moltbook / MBC-20 CLI starter")
    parser.add_argument("--api-key", help="Moltbook API key (or set MOLTBOOK_API_KEY)")
    parser.add_argument("--base-url", help="API base URL (default: https://www.moltbook.com/api/v1)")

    sub = parser.add_subparsers(dest="cmd", required=True)

    register = sub.add_parser("register", help="Create a Moltbook agent")
    register.add_argument("--name", required=True)
    register.add_argument("--description", required=True)
    register.set_defaults(func=cmd_register)

    status = sub.add_parser("status", help="Check claim status")
    status.set_defaults(func=cmd_status)

    post = sub.add_parser("post", help="Post arbitrary content")
    post.add_argument("--submolt", default="general")
    post.add_argument("--title", required=True)
    post.add_argument("--content")
    post.add_argument("--content-file")
    post.add_argument("--dry-run", action="store_true")
    post.set_defaults(func=cmd_post)

    mint = sub.add_parser("mint", help="Mint an MBC-20 inscription")
    mint.add_argument("--tick", required=True)
    mint.add_argument("--amount", required=True)
    mint.add_argument("--submolt", default="mbc20")
    mint.add_argument("--title")
    mint.add_argument("--prefix", default="mbc20.xyz")
    mint.add_argument("--dry-run", action="store_true")
    mint.set_defaults(func=cmd_mint)

    link = sub.add_parser("link", help="Link a wallet for future claim")
    link.add_argument("--address", required=True)
    link.add_argument("--submolt", default="mbc20")
    link.add_argument("--title")
    link.add_argument("--prefix", default="mbc20.xyz")
    link.add_argument("--dry-run", action="store_true")
    link.set_defaults(func=cmd_link)

    verify = sub.add_parser("verify", help="Verify a post challenge")
    verify.add_argument("--code", required=True)
    verify.add_argument("--answer", required=True)
    verify.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
