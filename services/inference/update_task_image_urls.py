"""Bulk-update Label Studio task data, replacing an old image URL prefix
(e.g. a stale localhost:PORT) with a new one (e.g. a cloudflared tunnel URL),
across all tasks in a project.

Authenticates by logging in with your Label Studio username/password (read
from LABEL_STUDIO_USER / LABEL_STUDIO_PASSWORD env vars) to obtain a fresh
JWT access token, so no token or password ever has to be typed into a chat
or committed to a file. Alternatively, set LABEL_STUDIO_TOKEN directly if you
already have a valid access token.

Usage (PowerShell):
    $env:LABEL_STUDIO_USER = "you@example.com"
    $env:LABEL_STUDIO_PASSWORD = "..."
    python update_task_image_urls.py `
        --url http://localhost:8080 `
        --project 1 `
        --old http://localhost:8899 `
        --new https://dining-let-cassette-memphis.trycloudflare.com

Add --dry-run to preview changes without writing anything.
"""

import argparse
import os
import sys

import requests


def get_access_token(base_url: str, session: requests.Session) -> str:
    username = os.environ.get("LABEL_STUDIO_USER")
    password = os.environ.get("LABEL_STUDIO_PASSWORD")
    if not username or not password:
        print(
            "Set LABEL_STUDIO_USER and LABEL_STUDIO_PASSWORD in your environment "
            "(or LABEL_STUDIO_TOKEN if you already have an access token).",
            file=sys.stderr,
        )
        sys.exit(1)

    # 1. Try standard Django token API endpoints
    resp = session.post(
        f"{base_url}/api/token/",
        json={"email": username, "password": password},
    )
    if resp.ok and "access" in resp.json():
        return resp.json()["access"]

    # 2. Try HTML session login to /user/login/
    login_url = f"{base_url}/user/login/"
    login_page = session.get(login_url)
    csrf_token = session.cookies.get("csrftoken", "")
    
    login_data = {
        "email": username,
        "password": password,
        "csrfmiddlewaretoken": csrf_token,
    }
    headers = {"Referer": login_url}
    post_resp = session.post(login_url, data=login_data, headers=headers)
    if post_resp.ok:
        # Check if logged in by querying current user API or legacy token
        token_resp = session.get(f"{base_url}/api/current-user/whoami")
        if token_resp.ok:
            user_data = token_resp.json()
            if "auth_token" in user_data:
                return user_data["auth_token"]
            return "SESSION_AUTH"

    print(f"Login failed. API Token response: {resp.status_code} {resp.text[:200]}", file=sys.stderr)
    post_resp.raise_for_status()
    resp.raise_for_status()
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Label Studio base URL")
    parser.add_argument("--project", required=True, type=int, help="Project ID")
    parser.add_argument("--old", required=True, help="Old URL prefix to replace")
    parser.add_argument("--new", required=True, help="New URL prefix")
    parser.add_argument("--field", default="image", help="Task data field holding the URL")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--legacy-token",
        action="store_true",
        help="Use 'Authorization: Token ...' instead of 'Bearer ...' (older Label Studio versions)",
    )
    args = parser.parse_args()

    session = requests.Session()
    token = os.environ.get("LABEL_STUDIO_TOKEN")
    if token:
        # Auto-detect JWT vs legacy DRF token header scheme
        if token.startswith("eyJ") and not args.legacy_token:
            auth_scheme = "Bearer"
        elif args.legacy_token or len(token) == 40:
            auth_scheme = "Token"
        else:
            auth_scheme = "Bearer"
        session.headers["Authorization"] = f"{auth_scheme} {token}"
    else:
        token = get_access_token(args.url, session)
        if token and token != "SESSION_AUTH":
            auth_scheme = "Bearer" if token.startswith("eyJ") else "Token"
            session.headers["Authorization"] = f"{auth_scheme} {token}"
        else:
            auth_scheme = "Session Cookie"

    tasks_url = f"{args.url}/api/tasks"
    resp = session.get(tasks_url, params={"project": args.project, "page_size": 10000})
    print(f"Request URL: {resp.request.url}", file=sys.stderr)
    print(f"Request headers sent: {dict(resp.request.headers)}", file=sys.stderr)
    if resp.history:
        print(f"Redirected via: {[r.status_code for r in resp.history]}", file=sys.stderr)
    if not resp.ok:
        print(f"Auth scheme used: {auth_scheme}", file=sys.stderr)
        print(f"Response status: {resp.status_code}", file=sys.stderr)
        print(f"Response body: {resp.text[:500]}", file=sys.stderr)
    resp.raise_for_status()
    tasks = resp.json().get("tasks", resp.json()) if isinstance(resp.json(), dict) else resp.json()

    changed = 0
    for task in tasks:
        data = task.get("data", {})
        value = data.get(args.field)
        if not isinstance(value, str) or not value.startswith(args.old):
            continue
        new_value = args.new + value[len(args.old):]
        changed += 1
        print(f"task {task['id']}: {value} -> {new_value}")
        if not args.dry_run:
            patch_resp = session.patch(
                f"{args.url}/api/tasks/{task['id']}",
                json={"data": {**data, args.field: new_value}},
            )
            patch_resp.raise_for_status()

    print(f"\n{'Would update' if args.dry_run else 'Updated'} {changed} task(s).")


if __name__ == "__main__":
    main()
