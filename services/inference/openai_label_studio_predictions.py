"""Create review-only cavity predictions in Label Studio with the OpenAI Responses API.

This tool never creates or updates Label Studio annotations.  It creates model
predictions alongside the existing annotations in Projects 1 and 2, so every
suggested lesion remains reviewable by a dental professional before it is used
as training data.

Required services/inference/.env values:
    OPENAI_API_KEY=...
    LABEL_STUDIO_URL=http://127.0.0.1:8080
    LABEL_STUDIO_REFRESH_TOKEN=...

Preview the exact task selection without calling OpenAI or writing to Label Studio:
    python openai_label_studio_predictions.py --projects 1 7 --dry-run

Create predictions after reviewing the preview:
    python openai_label_studio_predictions.py --projects 1 7 --commit
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

DEFAULT_PROJECT_IDS = (1, 7)  # Label Studio's “New Project #1” and “New Project #2”.
DEFAULT_MODEL = "gpt-6-astra"
MODEL_VERSION_PREFIX = "openai-cavity-draft"
IMAGE_DATA_KEY = "image"

# These are the only cavity labels shared by both existing projects.  Labels
# such as root canal, filling, sinus and infection are deliberately out of
# scope: this workflow does not claim to annotate them.
CAVITY_LABELS = (
    "interproximal_mild",
    "interproximal_moderate",
    "interproximal_severe",
)

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "label": {"type": "string", "enum": list(CAVITY_LABELS)},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "points": {
                        "type": "array",
                        "minItems": 3,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "x": {"type": "number", "minimum": 0, "maximum": 100},
                                "y": {"type": "number", "minimum": 0, "maximum": 100},
                            },
                            "required": ["x", "y"],
                        },
                    },
                },
                "required": ["label", "confidence", "points"],
            },
        }
    },
    "required": ["findings"],
}

SYSTEM_PROMPT = """You are preparing draft research labels for dental panoramic X-rays.
Identify only visible interproximal caries lesions. Return no finding when the
image is insufficient, ambiguous, or shows a non-caries finding. For each
finding, choose one supplied severity label and provide a tight polygon around
the radiolucent lesion. Polygon coordinates are percentages of the original
image (x left-to-right, y top-to-bottom). Do not label a whole tooth, infer a
diagnosis from missing teeth, or emit root canals, fillings, infection, sinus,
or nerve findings. These are AI suggestions for clinician review, not clinical
diagnoses or final training labels."""


def load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE entries without printing or persisting secrets."""
    if not path.exists():
        raise RuntimeError(f"Missing environment file: {path}")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required; add it to services/inference/.env")
    return value


def get_access_token(client: httpx.Client, base_url: str, refresh_token: str) -> str:
    response = client.post(f"{base_url}/api/token/refresh/", json={"refresh": refresh_token})
    response.raise_for_status()
    token = response.json().get("access")
    if not token:
        raise RuntimeError("Label Studio refresh response did not contain an access token.")
    return token


def list_tasks(client: httpx.Client, base_url: str, headers: dict[str, str], project_id: int) -> list[dict]:
    tasks: list[dict] = []
    page = 1
    page_size = 1000
    while True:
        response = client.get(
            f"{base_url}/api/projects/{project_id}/tasks",
            headers=headers,
            params={"page": page, "page_size": page_size, "fields": "all"},
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list):
            tasks.extend(payload)
            if len(payload) < page_size:
                break
            page += 1
            continue
        tasks.extend(payload.get("tasks", payload.get("results", [])))
        if not payload.get("next"):
            break
        page += 1
    return tasks


def image_url(base_url: str, task: dict) -> str:
    try:
        url = task["data"][IMAGE_DATA_KEY]
    except KeyError as exc:
        raise RuntimeError(f"Task {task.get('id')} has no data.{IMAGE_DATA_KEY}") from exc
    return url if url.startswith(("http://", "https://")) else f"{base_url}{url}"


def response_text(payload: dict) -> str:
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                return content["text"]
    raise RuntimeError("OpenAI response contained no structured output text.")


def ask_openai(client: httpx.Client, api_key: str, image_bytes: bytes, content_type: str, model: str) -> dict:
    data_url = f"data:{content_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Review this X-ray and return the required JSON."},
                    {"type": "input_image", "image_url": data_url, "detail": "high"},
                ],
            },
        ],
        "text": {"format": {"type": "json_schema", "name": "cavity_draft", "strict": True, "schema": RESPONSE_SCHEMA}},
    }
    response = client.post(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=120.0,
    )
    response.raise_for_status()
    return json.loads(response_text(response.json()))


def to_label_studio_result(findings: list[dict]) -> list[dict]:
    result = []
    for finding in findings:
        label = finding.get("label")
        points = finding.get("points", [])
        if label not in CAVITY_LABELS or len(points) < 3:
            continue
        result.append(
            {
                "from_name": "label",
                "to_name": "image",
                "type": "polygonlabels",
                "score": float(finding.get("confidence", 0)),
                "value": {
                    "points": [[round(float(point["x"]), 3), round(float(point["y"]), 3)] for point in points],
                    "polygonlabels": [label],
                },
            }
        )
    return result


def already_predicted(task: dict, model_version: str) -> bool:
    return any(prediction.get("model_version") == model_version for prediction in task.get("predictions", []))


def create_prediction(
    client: httpx.Client, base_url: str, headers: dict[str, str], task_id: int, result: list[dict], model_version: str
) -> None:
    response = client.post(
        f"{base_url}/api/predictions/",
        headers=headers,
        json={"task": task_id, "result": result, "model_version": model_version},
    )
    response.raise_for_status()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projects", type=int, nargs="+", default=list(DEFAULT_PROJECT_IDS))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, help="Process only this many unpredicted tasks per project.")
    parser.add_argument("--commit", action="store_true", help="Call OpenAI and create Label Studio predictions.")
    parser.add_argument("--dry-run", action="store_true", help="List eligible tasks only (the default).")
    args = parser.parse_args()
    if args.commit and args.dry_run:
        parser.error("Use either --commit or --dry-run, not both.")

    load_dotenv(Path(__file__).with_name(".env"))
    base_url = required_env("LABEL_STUDIO_URL").rstrip("/")
    refresh_token = required_env("LABEL_STUDIO_REFRESH_TOKEN")
    api_key = required_env("OPENAI_API_KEY")
    model_version = f"{MODEL_VERSION_PREFIX}:{args.model}"

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        access_token = get_access_token(client, base_url, refresh_token)
        ls_headers = {"Authorization": f"Bearer {access_token}"}
        eligible: list[tuple[int, dict]] = []
        for project_id in args.projects:
            tasks = list_tasks(client, base_url, ls_headers, project_id)
            pending = [task for task in tasks if not already_predicted(task, model_version)]
            if args.limit is not None:
                pending = pending[: args.limit]
            eligible.extend((project_id, task) for task in pending)
            print(f"Project {project_id}: {len(tasks)} tasks; {len(pending)} eligible for {model_version}.")

        if not args.commit:
            print(f"Dry run: {len(eligible)} tasks selected. No OpenAI calls or Label Studio writes were made.")
            return

        for position, (project_id, task) in enumerate(eligible, start=1):
            task_id = task["id"]
            image_response = client.get(image_url(base_url, task), headers=ls_headers)
            image_response.raise_for_status()
            findings = ask_openai(
                client,
                api_key,
                image_response.content,
                image_response.headers.get("content-type", "image/png").split(";", 1)[0],
                args.model,
            )["findings"]
            result = to_label_studio_result(findings)
            create_prediction(client, base_url, ls_headers, task_id, result, model_version)
            print(f"[{position}/{len(eligible)}] project={project_id} task={task_id}: {len(result)} cavity drafts")
            time.sleep(0.25)


if __name__ == "__main__":
    try:
        main()
    except (httpx.HTTPError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
