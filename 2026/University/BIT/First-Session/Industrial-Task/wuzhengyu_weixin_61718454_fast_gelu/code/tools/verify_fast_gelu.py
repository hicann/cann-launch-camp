#!/usr/bin/env python3
"""Reusable checks for the FastGelu CANNJudge solution.

Default mode runs local static tests and prints normalized SHA-256 hashes for
files that are editable on the CANNJudge submit page.

Use --webbridge to also borrow the active browser tab and compare the page
editor contents against local files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
EDITABLE_FILES = [
    "op_host/fast_gelu.cpp",
    "op_kernel/fast_gelu_tiling.h",
    "op_kernel/fast_gelu.cpp",
    "op_kernel/tiling_key_fast_gelu.h",
]
DEFAULT_SUBMIT_URL = "https://cannjudge.cn/bit/public/public/submit"
DEFAULT_SESSION = "fastgelu-verify"
WEBBRIDGE_ENDPOINT = "http://127.0.0.1:10086/command"


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_local_files() -> dict[str, dict[str, str | int]]:
    files: dict[str, dict[str, str | int]] = {}
    for rel in EDITABLE_FILES:
        path = CODE_DIR / rel
        raw = path.read_text(encoding="utf-8")
        normalized = normalize_newlines(raw)
        files[rel] = {
            "path": str(path),
            "text": normalized,
            "length": len(normalized),
            "sha256": sha256_text(normalized),
        }
    return files


def run_static_tests() -> bool:
    print("[static] running unittest discovery...")
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"],
        cwd=CODE_DIR,
        text=True,
        capture_output=True,
    )
    output = (proc.stdout + proc.stderr).strip()
    if output:
        print(output)
    ok = proc.returncode == 0
    print(f"[static] {'OK' if ok else 'FAILED'}")
    return ok


def webbridge_command(action: str, args: dict, session: str) -> dict:
    body = json.dumps(
        {"action": action, "args": args, "session": session},
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        WEBBRIDGE_ENDPOINT,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        payload = resp.read().decode("utf-8")
    result = json.loads(payload)
    if isinstance(result, dict) and "data" in result:
        return result["data"]
    return result


def borrow_submit_tab(url: str, session: str) -> None:
    data = webbridge_command("find_tab", {"url": url, "active": True}, session)
    if not data.get("success"):
        raise RuntimeError(f"Could not borrow submit tab: {data}")
    print(f"[webbridge] borrowed tab {data.get('tabId')} {data.get('url')}")


def editor_available(session: str) -> bool:
    js = """
    (() => JSON.stringify({
      url: location.href,
      hasEditor: Boolean(document.querySelector('textarea#editor-main')),
      fileCount: document.querySelectorAll('button.open-editor-tree-row[title]').length
    }))()
    """
    data = webbridge_command("evaluate", {"code": js}, session)
    page_result = json.loads(data["value"])
    print(
        "[webbridge] page "
        f"url={page_result.get('url')} "
        f"hasEditor={page_result.get('hasEditor')} "
        f"fileCount={page_result.get('fileCount')}"
    )
    return bool(page_result.get("hasEditor")) and int(page_result.get("fileCount", 0)) > 0


def open_submit_editor(url: str, session: str) -> None:
    data = webbridge_command(
        "navigate",
        {"url": url, "newTab": True, "group_title": "FastGelu verify"},
        session,
    )
    if not data.get("success"):
        raise RuntimeError(f"Could not open submit page: {data}")
    print(f"[webbridge] opened submit page {data.get('url')}")
    time.sleep(1.0)


def read_page_file(rel: str, session: str) -> str:
    js = f"""
    (async () => {{
      const rel = {json.dumps(rel)};
      const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
      const buttons = [...document.querySelectorAll('button[title]')];
      const btn =
        buttons.find(b => b.getAttribute('title') === rel && b.className.includes('open-editor-tree-row')) ||
        buttons.find(b => b.getAttribute('title') === rel && b.className.includes('open-editor-file-tab-trigger')) ||
        buttons.find(b => b.getAttribute('title') === rel);
      if (!btn) return JSON.stringify({{ok:false, error:'file button not found', rel}});
      btn.click();
      await sleep(250);
      const ta = document.querySelector('textarea#editor-main');
      if (!ta) return JSON.stringify({{ok:false, error:'textarea not found', rel}});
      return JSON.stringify({{
        ok: true,
        rel,
        value: ta.value,
        activeTitle: document.querySelector('.open-editor-file-tab-trigger.active')?.getAttribute('title') || null
      }});
    }})()
    """
    data = webbridge_command("evaluate", {"code": js}, session)
    page_result = json.loads(data["value"])
    if not page_result.get("ok"):
        raise RuntimeError(f"Could not read {rel} from page: {page_result}")
    return normalize_newlines(page_result["value"])


def verify_webbridge(local_files: dict[str, dict[str, str | int]], url: str, session: str) -> bool:
    print("[webbridge] verifying editor contents...")
    borrow_submit_tab(url, session)
    if not editor_available(session):
        print("[webbridge] active page is not the editor; opening submit page...")
        open_submit_editor(url, session)
    if not editor_available(session):
        raise RuntimeError("Submit editor is still unavailable after opening the submit page.")
    ok = True
    for rel in EDITABLE_FILES:
        time.sleep(0.05)
        page_text = read_page_file(rel, session)
        page_len = len(page_text)
        page_sha = sha256_text(page_text)
        local_len = local_files[rel]["length"]
        local_sha = local_files[rel]["sha256"]
        match = page_text == local_files[rel]["text"]
        ok = ok and match
        status = "OK" if match else "MISMATCH"
        print(f"[webbridge] {status} {rel} len={page_len} sha256={page_sha}")
        if not match:
            print(f"  local len={local_len} sha256={local_sha}")
    return ok


def print_local_hashes(local_files: dict[str, dict[str, str | int]]) -> None:
    print("[local] normalized editable file hashes:")
    for rel in EDITABLE_FILES:
        info = local_files[rel]
        print(f"[local] {rel} len={info['length']} sha256={info['sha256']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify FastGelu local and CANNJudge editor state.")
    parser.add_argument("--webbridge", action="store_true", help="Compare the active submit page editor with local files.")
    parser.add_argument("--skip-static", action="store_true", help="Skip unittest-based static checks.")
    parser.add_argument("--url", default=DEFAULT_SUBMIT_URL, help="Submit page URL to borrow when --webbridge is set.")
    parser.add_argument("--session", default=DEFAULT_SESSION, help="Kimi WebBridge session name.")
    args = parser.parse_args()

    local_files = load_local_files()
    print_local_hashes(local_files)

    ok = True
    if not args.skip_static:
        ok = run_static_tests() and ok
    if args.webbridge:
        ok = verify_webbridge(local_files, args.url, args.session) and ok

    print(f"[result] {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
