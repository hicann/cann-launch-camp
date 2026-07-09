#!/usr/bin/env python3
"""Overnight FastGelu optimizer for CANNJudge.

The pipeline edits only the four CANNJudge-editable source files, runs local
static checks, syncs the page editor through Kimi WebBridge, submits, polls the
result, and records every attempt as JSONL.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = PROJECT_ROOT / "code"
RUN_DIR = PROJECT_ROOT / "allnight_runs"
SUBMIT_URL = "https://cannjudge.cn/bit/public/public/submit"
WEBBRIDGE_ENDPOINT = "http://127.0.0.1:10086/command"
EDITABLE_FILES = [
    "op_host/fast_gelu.cpp",
    "op_kernel/fast_gelu_tiling.h",
    "op_kernel/fast_gelu.cpp",
    "op_kernel/tiling_key_fast_gelu.h",
]
CURRENT_BEST_SUM_US = 28.76
KNOWN_PER_POINT_BEST_US = [3.52, 3.38, 6.40, 6.46, 7.92]


@dataclasses.dataclass(frozen=True)
class Candidate:
    name: str
    kind: str
    tile: int
    threshold_mult: int
    buffer_num: int
    half_tile: int | None = None
    hypothesis: str = ""


def normalize(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_sources() -> dict[str, str]:
    return {rel: normalize((CODE_DIR / rel).read_text(encoding="utf-8")) for rel in EDITABLE_FILES}


def write_sources(files: dict[str, str]) -> None:
    for rel, text in files.items():
        (CODE_DIR / rel).write_text(text, encoding="utf-8", newline="\n")


def replace_one(pattern: str, repl: str, text: str, label: str) -> str:
    new_text, count = re.subn(pattern, repl, text, count=1)
    if count != 1:
        raise RuntimeError(f"could not replace {label}")
    return new_text


def apply_candidate(base_files: dict[str, str], cand: Candidate) -> dict[str, str]:
    files = dict(base_files)
    host = files["op_host/fast_gelu.cpp"]
    kernel = files["op_kernel/fast_gelu.cpp"]
    if "Sigmoid(" not in kernel:
        raise RuntimeError("baseline kernel must already use the sigmoid formula")

    if cand.kind == "dtype_tile":
        host = replace_one(
            r"constexpr uint32_t TILE_ELEM_NUM = \d+;",
            f"constexpr uint32_t TILE_ELEM_NUM = {max(cand.tile, cand.half_tile or cand.tile)};",
            host,
            "host dtype-aware TILE_ELEM_NUM",
        )
        if "constexpr uint32_t FLOAT_TILE_ELEM_NUM" in host:
            host = replace_one(
                r"constexpr uint32_t FLOAT_TILE_ELEM_NUM = \d+;",
                f"constexpr uint32_t FLOAT_TILE_ELEM_NUM = {cand.tile};",
                host,
                "host FLOAT_TILE_ELEM_NUM",
            )
            host = replace_one(
                r"constexpr uint32_t HALF_TILE_ELEM_NUM = \d+;",
                f"constexpr uint32_t HALF_TILE_ELEM_NUM = {cand.half_tile};",
                host,
                "host HALF_TILE_ELEM_NUM",
            )
        else:
            host = replace_one(
                r"(constexpr uint32_t TILE_ELEM_NUM = \d+;\n)",
                (
                    r"\1"
                    f"constexpr uint32_t FLOAT_TILE_ELEM_NUM = {cand.tile};\n"
                    f"constexpr uint32_t HALF_TILE_ELEM_NUM = {cand.half_tile};\n"
                ),
                host,
                "host dtype-specific tile constants",
            )
        if "GetTileElemNum" not in host:
            host = replace_one(
                r"(    static uint32_t GetDataTypeSize\(ge::DataType dtype\) \{\n"
                r"        return dtype == ge::DT_FLOAT16 \? 2U : 4U;\n"
                r"    \}\n)",
                (
                    r"\1\n"
                    "    static uint32_t GetTileElemNum(ge::DataType dtype) {\n"
                    "        return dtype == ge::DT_FLOAT16 ? HALF_TILE_ELEM_NUM : FLOAT_TILE_ELEM_NUM;\n"
                    "    }\n"
                ),
                host,
                "host GetTileElemNum",
            )
        if "uint32_t tile_elem_num = GetTileElemNum(dtype_x);" not in host:
            host = replace_one(
                r"(        uint32_t type_length = GetDataTypeSize\(dtype_x\);\n)",
                r"\1        uint32_t tile_elem_num = GetTileElemNum(dtype_x);\n",
                host,
                "host tile_elem_num local",
            )
        host = host.replace(
            "uint32_t tile_block_num = (TILE_ELEM_NUM * type_length) / BLOCK_SIZE;",
            "uint32_t tile_block_num = (tile_elem_num * type_length) / BLOCK_SIZE;",
        )
        host = host.replace(
            "uint32_t small_tail_data_num = small_core_data_num - TILE_ELEM_NUM * small_tile_num;",
            "uint32_t small_tail_data_num = small_core_data_num - tile_elem_num * small_tile_num;",
        )
        host = host.replace("small_tail_data_num = TILE_ELEM_NUM;", "small_tail_data_num = tile_elem_num;")
        host = host.replace(
            "uint32_t big_tail_data_num = big_core_data_num - TILE_ELEM_NUM * big_tile_num;",
            "uint32_t big_tail_data_num = big_core_data_num - tile_elem_num * big_tile_num;",
        )
        host = host.replace("big_tail_data_num = TILE_ELEM_NUM;", "big_tail_data_num = tile_elem_num;")
        host = host.replace("tiling->tileDataNum = TILE_ELEM_NUM;", "tiling->tileDataNum = tile_elem_num;")
    else:
        host = replace_one(
            r"constexpr uint32_t TILE_ELEM_NUM = \d+;",
            f"constexpr uint32_t TILE_ELEM_NUM = {cand.tile};",
            host,
            "host TILE_ELEM_NUM",
        )
        if "constexpr uint32_t FLOAT_TILE_ELEM_NUM" in host:
            host = replace_one(
                r"constexpr uint32_t FLOAT_TILE_ELEM_NUM = \d+;",
                f"constexpr uint32_t FLOAT_TILE_ELEM_NUM = {cand.tile};",
                host,
                "host const FLOAT_TILE_ELEM_NUM",
            )
            host = replace_one(
                r"constexpr uint32_t HALF_TILE_ELEM_NUM = \d+;",
                f"constexpr uint32_t HALF_TILE_ELEM_NUM = {cand.tile};",
                host,
                "host const HALF_TILE_ELEM_NUM",
            )
    host = replace_one(
        r"constexpr uint32_t LARGE_CORE_THRESHOLD = (?:TILE_ELEM_NUM|CORE_SPLIT_ELEM_NUM) \* \d+;",
        f"constexpr uint32_t LARGE_CORE_THRESHOLD = CORE_SPLIT_ELEM_NUM * {cand.threshold_mult};",
        host,
        "host LARGE_CORE_THRESHOLD",
    )
    kernel = replace_one(
        r"constexpr int32_t BUFFER_NUM = \d+;",
        f"constexpr int32_t BUFFER_NUM = {cand.buffer_num};",
        kernel,
        "kernel BUFFER_NUM",
    )
    kernel = replace_one(
        r"constexpr uint32_t TILE_ELEM_NUM = \d+;",
        f"constexpr uint32_t TILE_ELEM_NUM = {max(cand.tile, cand.half_tile or cand.tile)};",
        kernel,
        "kernel TILE_ELEM_NUM",
    )
    files["op_host/fast_gelu.cpp"] = host
    files["op_kernel/fast_gelu.cpp"] = kernel
    return files


def candidate_plan() -> list[Candidate]:
    candidates: list[Candidate] = []
    high_value = [
        Candidate(
            "v6_dtype_f4096_h8192_thr4_buf2",
            "dtype_tile",
            4096,
            4,
            2,
            half_tile=8192,
            hypothesis="Keep float32 16KB tiles and raise float16 tiles to 16KB copy size.",
        ),
        Candidate(
            "v6_dtype_f4096_h8192_thr4_buf1",
            "dtype_tile",
            4096,
            4,
            1,
            half_tile=8192,
            hypothesis="Serial loop may not benefit from BUFFER_NUM=2; reduce queue/UB pressure.",
        ),
        Candidate(
            "v6_dtype_f4096_h6144_thr4_buf2",
            "dtype_tile",
            4096,
            4,
            2,
            half_tile=6144,
            hypothesis="Intermediate half tile between 8KB and 16KB may help test 4 tail balance.",
        ),
        Candidate(
            "v6_const_t3072_thr4_buf2",
            "const_tile",
            3072,
            4,
            2,
            hypothesis="Reduce V5 tile size to recover test 4 while staying larger than V3.",
        ),
        Candidate(
            "v6_const_t2048_thr4_buf1",
            "const_tile",
            2048,
            4,
            1,
            hypothesis="Use V3-sized tiles with generalized distribution and lower buffer depth.",
        ),
    ]
    seen: set[tuple[str, int, int, int, int | None]] = set()
    for cand in high_value:
        key = (cand.kind, cand.tile, cand.threshold_mult, cand.buffer_num, cand.half_tile)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(cand)

    # Lower-priority coarse sweeps. Keep these behind the hypothesis-driven
    # candidates so overnight runs do not spam CANNJudge with low-signal tries.
    for tile in [4096, 2048, 3072, 2560, 3584, 1024]:
        for mult in [3, 4, 6]:
            key = ("const_tile", tile, mult, 2, None)
            if key not in seen:
                seen.add(key)
                candidates.append(
                    Candidate(
                        f"v6_const_t{tile}_thr{mult}_buf2",
                        "const_tile",
                        tile,
                        mult,
                        2,
                        hypothesis="Low-priority coarse sweep after dtype-aware candidates.",
                    )
                )
    return candidates


def run_static() -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "tools/verify_fast_gelu.py"],
        cwd=CODE_DIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc.returncode == 0, proc.stdout


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
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = resp.read().decode("utf-8")
    result = json.loads(payload)
    return result.get("data", result)


def evaluate(code: str, session: str) -> dict:
    data = webbridge_command("evaluate", {"code": code}, session)
    value = data.get("value")
    if isinstance(value, str):
        return json.loads(value)
    return data


def page_state(session: str) -> dict:
    return evaluate(
        """
        (() => JSON.stringify({
          url: location.href,
          hasEditor: Boolean(document.querySelector('textarea#editor-main')),
          fileCount: document.querySelectorAll('button.open-editor-tree-row[title]').length,
          text: document.body ? document.body.innerText : ''
        }))()
        """,
        session,
    )


def ensure_submit_editor(session: str) -> None:
    webbridge_command("find_tab", {"url": SUBMIT_URL, "active": True}, session)
    state = page_state(session)
    if state.get("hasEditor") and state.get("fileCount", 0) > 0:
        return
    webbridge_command(
        "navigate",
        {"url": SUBMIT_URL, "newTab": False, "group_title": "FastGelu allnight"},
        session,
    )
    for _ in range(20):
        time.sleep(0.5)
        state = page_state(session)
        if state.get("hasEditor") and state.get("fileCount", 0) > 0:
            return
    raise RuntimeError("submit editor did not become available")


def write_editor_file(rel: str, text: str, session: str) -> dict:
    js = """
    (async () => {
      const rel = REL_JSON;
      const text = TEXT_JSON;
      const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
      const setNativeValue = (element, value) => {
        const descriptor = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(element), 'value');
        descriptor.set.call(element, value);
        element.dispatchEvent(new Event('input', {bubbles: true}));
        element.dispatchEvent(new Event('change', {bubbles: true}));
      };
      const buttons = [...document.querySelectorAll('button[title]')];
      const btn =
        buttons.find(b => b.getAttribute('title') === rel && b.className.includes('open-editor-tree-row')) ||
        buttons.find(b => b.getAttribute('title') === rel && b.className.includes('open-editor-file-tab-trigger')) ||
        buttons.find(b => b.getAttribute('title') === rel);
      if (!btn) {
        return JSON.stringify({rel, ok: false, error: 'file button not found'});
      }
      btn.click();
      await sleep(200);
      const ta = document.querySelector('textarea#editor-main');
      if (!ta) {
        return JSON.stringify({rel, ok: false, error: 'textarea not found'});
      }
      setNativeValue(ta, text);
      await sleep(100);
      return JSON.stringify({rel, ok: ta.value === text, length: ta.value.length});
    })()
    """.replace("REL_JSON", json.dumps(rel)).replace("TEXT_JSON", json.dumps(text))
    data = webbridge_command("evaluate", {"code": js}, session)
    return json.loads(data["value"])


def sync_editor(files: dict[str, str], session: str) -> list[dict]:
    ensure_submit_editor(session)
    result = []
    for rel in EDITABLE_FILES:
        result.append(write_editor_file(rel, files[rel], session))
    return result


def read_editor_file(rel: str, session: str) -> str:
    js = f"""
    (async () => {{
      const rel = {json.dumps(rel)};
      const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
      const buttons = [...document.querySelectorAll('button[title]')];
      const btn =
        buttons.find(b => b.getAttribute('title') === rel && b.className.includes('open-editor-tree-row')) ||
        buttons.find(b => b.getAttribute('title') === rel && b.className.includes('open-editor-file-tab-trigger')) ||
        buttons.find(b => b.getAttribute('title') === rel);
      if (!btn) return JSON.stringify({{ok:false, error:'file button not found'}});
      btn.click();
      await sleep(200);
      const ta = document.querySelector('textarea#editor-main');
      if (!ta) return JSON.stringify({{ok:false, error:'textarea not found'}});
      return JSON.stringify({{ok:true, value:ta.value}});
    }})()
    """
    data = webbridge_command("evaluate", {"code": js}, session)
    result = json.loads(data["value"])
    if not result.get("ok"):
        raise RuntimeError(f"could not read editor file {rel}: {result}")
    return normalize(result["value"])


def verify_editor(files: dict[str, str], session: str) -> bool:
    for rel in EDITABLE_FILES:
        if read_editor_file(rel, session) != files[rel]:
            return False
    return True


def click_submit(session: str) -> dict:
    js = """
    (() => {
      const buttons = [...document.querySelectorAll('button')];
      const btn = buttons.find(b => (b.innerText || b.textContent || '').trim() === '提交代码');
      if (!btn) return JSON.stringify({ok:false, error:'submit button not found'});
      btn.click();
      return JSON.stringify({ok:true});
    })()
    """
    return evaluate(js, session)


def parse_result(text: str, url: str) -> dict:
    times = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*μs", text)
    run_times = times[0::2] if len(times) >= 10 else times
    best_times = times[1::2] if len(times) >= 10 else []
    passed = re.search(r"通过\s*[:：]\s*(\d+)\s*/\s*(\d+)", text)
    submission_id = re.search(r"提交 ID\s*[:：]\s*(\d+)", text)
    status = re.search(r"提交状态\s*[:：]\s*([^\n]+)", text)
    numeric = [float(x) for x in run_times[:5]]
    return {
        "url": url,
        "submission_id": submission_id.group(1) if submission_id else None,
        "status": status.group(1).strip() if status else None,
        "pass_count": passed.group(1) if passed else None,
        "total_count": passed.group(2) if passed else None,
        "times_us": numeric,
        "best_times_us": [float(x) for x in best_times[:5]],
        "sum_us": round(sum(numeric), 3) if numeric else None,
        "passed": "Pass" in text and len(numeric) == 5,
        "failed": any(token in text for token in ["Fail", "Wrong Answer", "Compile Error", "Runtime Error"]),
    }


def refresh_page(session: str) -> None:
    webbridge_command("evaluate", {"code": "(() => { location.reload(); return JSON.stringify({ok:true}); })()"}, session)


def submit_and_poll(session: str, timeout_sec: int, poll_sec: float, refresh_after_polls: int) -> dict:
    clicked = click_submit(session)
    if not clicked.get("ok"):
        raise RuntimeError(f"submit click failed: {clicked}")
    start = time.monotonic()
    last_status = None
    poll_count = 0
    while time.monotonic() - start < timeout_sec:
        state = page_state(session)
        result = parse_result(state.get("text", ""), state.get("url", ""))
        poll_count += 1
        if result.get("status") != last_status:
            print(json.dumps({"poll_status": result.get("status"), "url": result.get("url")}), flush=True)
            last_status = result.get("status")
        if "/submission/" in result.get("url", "") and (result["passed"] or result["failed"]) and result["times_us"]:
            result["elapsed_sec"] = round(time.monotonic() - start, 1)
            return result
        if (
            refresh_after_polls > 0
            and poll_count % refresh_after_polls == 0
            and "/submission/" in result.get("url", "")
            and result.get("status") == "Running"
        ):
            print(json.dumps({"refresh": True, "poll_count": poll_count, "url": result.get("url")}), flush=True)
            refresh_page(session)
            time.sleep(2)
        time.sleep(poll_sec)
    state = page_state(session)
    result = parse_result(state.get("text", ""), state.get("url", ""))
    result["elapsed_sec"] = round(time.monotonic() - start, 1)
    result["timeout"] = True
    return result


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_history() -> dict:
    attempted: set[str] = set()
    best_sum = CURRENT_BEST_SUM_US
    best_row: dict | None = None
    per_point_best: list[float | None] = list(KNOWN_PER_POINT_BEST_US)
    if not RUN_DIR.exists():
        return {
            "attempted": attempted,
            "best_sum": best_sum,
            "best_row": best_row,
            "per_point_best": per_point_best,
        }
    for path in RUN_DIR.glob("*.jsonl"):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            cand = row.get("candidate") or {}
            name = cand.get("name")
            result = row.get("result") or {}
            if name and result.get("submission_id"):
                attempted.add(name)
            if result.get("passed") and result.get("sum_us") is not None:
                if result["sum_us"] < best_sum:
                    best_sum = result["sum_us"]
                    best_row = row
                for idx, value in enumerate(result.get("times_us", [])[:5]):
                    old = per_point_best[idx]
                    per_point_best[idx] = value if old is None else min(old, value)
    return {
        "attempted": attempted,
        "best_sum": best_sum,
        "best_row": best_row,
        "per_point_best": per_point_best,
    }


def write_review_diff(run_name: str, cand: Candidate) -> Path:
    review_dir = RUN_DIR / run_name / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    diff_path = review_dir / f"{cand.name}.diff"
    proc = subprocess.run(
        ["git", "diff", "--", *[str(CODE_DIR / rel) for rel in EDITABLE_FILES]],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    diff_path.write_text(proc.stdout, encoding="utf-8")
    return diff_path


def git_commit_candidate(cand: Candidate, result: dict) -> None:
    subprocess.run(["git", "add", *[str(CODE_DIR / rel) for rel in EDITABLE_FILES]], cwd=PROJECT_ROOT, check=True)
    times = ", ".join(f"{x:.2f}us" for x in result.get("times_us", []))
    message = f"Allnight candidate {cand.name}"
    body = (
        f"CANNJudge submission {result.get('submission_id')}: Pass 5/5, "
        f"times {times}; sum {result.get('sum_us')}us; elapsed {result.get('elapsed_sec')} seconds."
    )
    subprocess.run(["git", "commit", "-m", message, "-m", body], cwd=PROJECT_ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run overnight FastGelu CANNJudge optimization.")
    parser.add_argument("--submit", action="store_true", help="Actually sync to CANNJudge and submit candidates.")
    parser.add_argument("--commit-improvements", action="store_true", help="Commit candidates that beat --best-sum.")
    parser.add_argument("--max-candidates", type=int, default=0, help="0 means all candidates.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--ignore-history", action="store_true", help="Do not skip candidates already found in logs.")
    parser.add_argument("--explain-plan", action="store_true", help="Print ranked candidates/history and exit.")
    parser.add_argument(
        "--review-gate",
        action="store_true",
        help="Generate candidate code, run static checks, write a diff, then stop before online submission.",
    )
    parser.add_argument("--best-sum", type=float, default=CURRENT_BEST_SUM_US)
    parser.add_argument("--min-improvement", type=float, default=0.03)
    parser.add_argument("--timeout-sec", type=int, default=720)
    parser.add_argument("--poll-sec", type=float, default=8.0)
    parser.add_argument(
        "--refresh-after-polls",
        type=int,
        default=8,
        help="Refresh a Running submission page after this many polls; 0 disables refresh.",
    )
    parser.add_argument(
        "--cooldown-sec",
        type=int,
        default=900,
        help="Delay between online submissions; default 900s to avoid overloading CANNJudge.",
    )
    parser.add_argument("--no-cooldown", action="store_true", help="Disable inter-candidate cooldown.")
    parser.add_argument(
        "--stop-after-improvement",
        action="store_true",
        help="Stop the run after the first candidate beating --best-sum.",
    )
    parser.add_argument("--session", default="fastgelu-allnight")
    parser.add_argument("--run-name", default=time.strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    history = load_history()
    best_sum = min(args.best_sum, history["best_sum"])
    plan = candidate_plan()
    if not args.ignore_history:
        attempted = history["attempted"]
        plan = [cand for cand in plan if cand.name not in attempted]
    plan = plan[args.start_index :]
    if args.max_candidates > 0:
        plan = plan[: args.max_candidates]
    log_path = RUN_DIR / f"{args.run_name}.jsonl"
    base_files = read_sources()
    best_files = dict(base_files)
    best_name = "current"
    if history["best_row"]:
        best_name = (history["best_row"].get("candidate") or {}).get("name", "history")

    plan_summary = {
        "candidates": [dataclasses.asdict(c) for c in plan],
        "submit": args.submit,
        "review_gate": args.review_gate,
        "history": {
            "attempted_count": len(history["attempted"]),
            "best_sum": history["best_sum"],
            "per_point_best": history["per_point_best"],
        },
    }
    print(json.dumps(plan_summary, ensure_ascii=True))
    if args.explain_plan:
        return 0
    try:
        for idx, cand in enumerate(plan, start=args.start_index):
            if args.submit and idx != args.start_index and not args.no_cooldown and args.cooldown_sec > 0:
                print(
                    json.dumps(
                        {
                            "cooldown_sec": args.cooldown_sec,
                            "next_candidate": cand.name,
                            "reason": "avoid frequent CANNJudge submissions",
                        },
                        ensure_ascii=True,
                    ),
                    flush=True,
                )
                time.sleep(args.cooldown_sec)
            files = apply_candidate(base_files, cand)
            write_sources(files)
            ok, static_output = run_static()
            row = {
                "index": idx,
                "candidate": dataclasses.asdict(cand),
                "static_ok": ok,
                "local_hashes": {rel: sha256_text(text) for rel, text in files.items()},
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            if not ok:
                row["static_output_tail"] = static_output[-2000:]
                append_jsonl(log_path, row)
                print(json.dumps(row, ensure_ascii=True), flush=True)
                continue
            if args.review_gate:
                diff_path = write_review_diff(args.run_name, cand)
                row["review_gate"] = True
                row["diff_path"] = str(diff_path)
                row["review_decision"] = "stopped_before_submit"
                append_jsonl(log_path, row)
                print(json.dumps(row, ensure_ascii=True), flush=True)
                break
            if args.submit:
                sync_result = sync_editor(files, args.session)
                row["sync_ok"] = all(item.get("ok") for item in sync_result)
                row["web_hash_ok"] = verify_editor(files, args.session)
                if row["sync_ok"] and row["web_hash_ok"]:
                    result = submit_and_poll(
                        args.session,
                        args.timeout_sec,
                        args.poll_sec,
                        args.refresh_after_polls,
                    )
                    row["result"] = result
                    if result.get("passed") and result.get("sum_us") is not None:
                        improved = result["sum_us"] + args.min_improvement < best_sum
                        row["improved"] = improved
                        if improved:
                            best_sum = result["sum_us"]
                            best_name = cand.name
                            best_files = files
                            if args.commit_improvements:
                                git_commit_candidate(cand, result)
                            if args.stop_after_improvement:
                                append_jsonl(log_path, row)
                                print(json.dumps(row, ensure_ascii=True), flush=True)
                                break
                    else:
                        row["improved"] = False
                append_jsonl(log_path, row)
            else:
                append_jsonl(log_path, row)
            print(json.dumps(row, ensure_ascii=True), flush=True)
    finally:
        write_sources(best_files)

    print(json.dumps({"best_name": best_name, "best_sum": best_sum, "log": str(log_path)}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
