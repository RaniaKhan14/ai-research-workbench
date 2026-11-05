 """Rate-limited, resumable DeepEval of cached Papeer answers using Gemini.

- Does NOT invoke the Papeer graph or answer-generation model.
- Uses cached testcases_0_15.json.
- Paces every judge.generate() call to stay under the Gemini free-tier RPM cap.
- Retries 429/5xx errors with backoff and saves after EACH metric, not just after a whole case.
- Resumes completed cases and partially completed cases in the existing results JSON.

Run from the project root with uv, setting GEMINI_JUDGE_MODEL and
GEMINI_EVAL_RESULTS_FILE to match the existing results file.
"""
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "3600")
os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "7200")

from dotenv import load_dotenv
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase

load_dotenv()

ANSWERING_MODEL = "openai/gpt-oss-20b"
JUDGE_MODEL = os.getenv("GEMINI_JUDGE_MODEL", "gemini-3.5-flash-lite")
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
THRESHOLD = float(os.getenv("METRIC_THRESHOLD", "0.7"))
MIN_SECONDS_BETWEEN_CALLS = float(os.getenv("GEMINI_MIN_SECONDS_BETWEEN_CALLS", "4.5"))
MAX_API_RETRIES = int(os.getenv("GEMINI_MAX_API_RETRIES", "4"))
BATCH_START = int(os.getenv("EVAL_BATCH_START", "0"))
BATCH_END = int(os.getenv("EVAL_BATCH_END", "15"))
CASES_PATH = Path(os.getenv("EVAL_CASES_FILE", "testcases_0_15.json"))
RESULTS_PATH = Path(os.getenv(
    "GEMINI_EVAL_RESULTS_FILE",
    "eval_results_gemini_3_5_flash_lite_0_15.json",
))

EXPECTED_METRICS = [
    "Contextual Precision",
    "Contextual Recall",
    "Contextual Relevancy",
    "Answer Relevancy",
    "Faithfulness",
]


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc


def write_json(path, value):
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(path)


def is_rate_or_transient_error(exc):
    msg = f"{type(exc).__name__}: {exc}".upper()
    return any(x in msg for x in (
        "429", "RESOURCE_EXHAUSTED", "TOO MANY REQUESTS", "503", "UNAVAILABLE",
        "500 INTERNAL", "502", "BAD GATEWAY", "504", "GATEWAY TIMEOUT",
    ))


def retry_wait_seconds(exc, attempt):
    """Use explicit API retry delay if present; otherwise wait conservatively."""
    msg = str(exc)
    # Examples in Gemini errors: 'retry in 34.75s', 'retryDelay": "34s"'.
    matches = re.findall(r"retry(?:Delay)?[^\d]{0,20}(\d+(?:\.\d+)?)\s*s", msg, re.I)
    hinted = float(matches[-1]) if matches else 0.0
    msg_upper = msg.upper()
    if "429" in msg_upper or "RESOURCE_EXHAUSTED" in msg_upper:
        return max(65.0, hinted + 5.0)
    return max(20.0 * (attempt + 1), hinted + 5.0)


def attach_rate_limiter(judge):
    """Pace ALL calls routed through DeepEval's model.generate()."""
    original_generate = judge.generate
    state = {"last_call_start": 0.0, "calls": 0}

    def paced_generate(*args, **kwargs):
        for attempt in range(MAX_API_RETRIES + 1):
            elapsed = time.monotonic() - state["last_call_start"]
            wait = MIN_SECONDS_BETWEEN_CALLS - elapsed
            if wait > 0:
                time.sleep(wait)
            state["last_call_start"] = time.monotonic()
            state["calls"] += 1
            if state["calls"] % 5 == 0:
                print(f"   [rate limiter] judge requests attempted: {state['calls']}")
            try:
                return original_generate(*args, **kwargs)
            except Exception as exc:
                if not is_rate_or_transient_error(exc) or attempt >= MAX_API_RETRIES:
                    raise
                wait_seconds = retry_wait_seconds(exc, attempt)
                print(
                    f"   [Gemini rate/capacity limit] {type(exc).__name__}; "
                    f"waiting {wait_seconds:.0f}s before retry {attempt + 1}/{MAX_API_RETRIES}."
                )
                time.sleep(wait_seconds)
        raise RuntimeError("Gemini request retry loop ended unexpectedly.")

    # GeminiModel's schema helper delegates to self.generate, so this pacing covers
    # both ordinary prompts and DeepEval's structured/schema judge prompts.
    judge.generate = paced_generate
    return state


def result_item_for(case, context, previous):
    item = next((x for x in previous if x.get("input") == case["input"]), None)
    if item is None:
        item = {
            "input": case["input"],
            "actual_output": case["actual_output"],
            "expected_output": case["expected_output"],
            "success": None,
            "evaluation_complete": False,
            "answering_model": ANSWERING_MODEL,
            "judge_model": JUDGE_MODEL,
            "has_retrieval_context": bool(context),
            "retrieval_context_chunks": len(context),
            "retrieval_context_unique_chunks": len(set(str(c) for c in context)),
            "retrieval_context_chars": sum(len(str(c)) for c in context),
            "route": case.get("route"),
            "retrieval_attempts": case.get("retrieval_attempts"),
            "rewrite_count": case.get("rewrite_count"),
            "is_relevant": case.get("is_relevant"),
            "metrics": [],
        }
        previous.append(item)
    # Legacy completed rows from the previous script won't have this flag.
    existing_names = {m.get("name") for m in item.get("metrics", [])}
    item["evaluation_complete"] = all(name in existing_names for name in EXPECTED_METRICS)
    return item


def main():
    if not API_KEY:
        raise SystemExit("Missing GOOGLE_API_KEY or GEMINI_API_KEY in .env. Do not share the key.")
    if not CASES_PATH.exists():
        raise SystemExit(f"Could not find {CASES_PATH}; run this from the Papeer project root.")
    all_cases = read_json(CASES_PATH)
    if not isinstance(all_cases, list) or not all_cases:
        raise SystemExit(f"{CASES_PATH} must contain a non-empty JSON list.")
    if not (0 <= BATCH_START < BATCH_END <= len(all_cases)):
        raise SystemExit(f"Invalid batch {BATCH_START}:{BATCH_END} for {len(all_cases)} cases.")

    previous = read_json(RESULTS_PATH) if RESULTS_PATH.exists() else []
    if not isinstance(previous, list):
        raise SystemExit(f"{RESULTS_PATH} must contain a JSON list.")
    for item in previous:
        if item.get("answering_model") != ANSWERING_MODEL or item.get("judge_model") != JUDGE_MODEL:
            raise SystemExit(
                f"{RESULTS_PATH} contains a different model setting. Do not overwrite it; "
                "choose the exact existing judge model in GEMINI_JUDGE_MODEL or use a new filename."
            )

    judge = GeminiModel(model=JUDGE_MODEL, api_key=API_KEY, temperature=0)
    request_state = attach_rate_limiter(judge)
    metric_builders = [
        lambda: ContextualPrecisionMetric(threshold=THRESHOLD, model=judge, async_mode=False),
        lambda: ContextualRecallMetric(threshold=THRESHOLD, model=judge, async_mode=False),
        lambda: ContextualRelevancyMetric(threshold=THRESHOLD, model=judge, async_mode=False),
        lambda: AnswerRelevancyMetric(threshold=THRESHOLD, model=judge, async_mode=False),
        lambda: FaithfulnessMetric(threshold=THRESHOLD, model=judge, async_mode=False),
    ]

    print(f"Cached cases: {CASES_PATH} ({len(all_cases)} total)")
    print(f"Output (resume enabled): {RESULTS_PATH}")
    print(f"Judge: {JUDGE_MODEL}; pacing one generate call every {MIN_SECONDS_BETWEEN_CALLS:.1f}s")
    print("Answer-generation model / Papeer graph will NOT be called.")

    for index in range(BATCH_START, BATCH_END):
        case = all_cases[index]
        context = case.get("retrieval_context") or []
        item = result_item_for(case, context, previous)
        if item["evaluation_complete"]:
            print(f"[case {index + 1}/{len(all_cases)}] all five metrics saved; skipping")
            continue

        test_case = LLMTestCase(
            input=case["input"],
            actual_output=case["actual_output"],
            expected_output=case["expected_output"],
            retrieval_context=context,
        )
        print(f"\n[case {index + 1}/{len(all_cases)}] {case['input'][:85]}")
        for build_metric in metric_builders:
            metric = build_metric()
            existing_names = {m.get("name") for m in item.get("metrics", [])}
            if metric.__class__.__name__ in existing_names or metric.__class__.__name__.replace("Metric", "") in existing_names:
                continue
            # Match DeepEval's public displayed names to avoid duplicate metrics on resume.
            display_name = {
                "ContextualPrecisionMetric": "Contextual Precision",
                "ContextualRecallMetric": "Contextual Recall",
                "ContextualRelevancyMetric": "Contextual Relevancy",
                "AnswerRelevancyMetric": "Answer Relevancy",
                "FaithfulnessMetric": "Faithfulness",
            }[metric.__class__.__name__]
            if display_name in existing_names:
                continue
            print(f"   scoring {display_name}...")
            try:
                metric.measure(test_case)
                row = {
                    "name": display_name,
                    "score": metric.score,
                    "passed": bool(metric.success),
                    "reason": metric.reason,
                }
                item.setdefault("metrics", []).append(row)
                # Keep metric rows in canonical order across resumed partial cases.
                order = {name: i for i, name in enumerate(EXPECTED_METRICS)}
                item["metrics"].sort(key=lambda m: order.get(m.get("name"), 99))
                names_now = {m.get("name") for m in item["metrics"]}
                item["evaluation_complete"] = all(name in names_now for name in EXPECTED_METRICS)
                if item["evaluation_complete"]:
                    item["success"] = all(bool(m.get("passed")) for m in item["metrics"])
                write_json(RESULTS_PATH, previous)
                print(f"      saved score={metric.score}; passed={metric.success}")
            except KeyboardInterrupt:
                write_json(RESULTS_PATH, previous)
                print(f"Interrupted; all completed metrics remain saved to {RESULTS_PATH}.")
                raise
            except Exception as exc:
                write_json(RESULTS_PATH, previous)
                print(f"\nStopped at case {index + 1}, metric {display_name}: {type(exc).__name__}: {exc}")
                print(f"Any earlier metric scores are saved in {RESULTS_PATH}.")
                print("Correct the quota/capacity issue and rerun the same command; saved metrics will be skipped.")
                return
        write_json(RESULTS_PATH, previous)
        if item["evaluation_complete"]:
            print(f"[case {index + 1}] all five metric scores saved (success={item['success']}).")

    completed = sum(
        all(name in {m.get('name') for m in row.get('metrics', [])} for name in EXPECTED_METRICS)
        for row in previous
    )
    print(f"\nDone with selected range. Fully scored cases: {completed}/{len(all_cases)}.")
    print(f"Results: {RESULTS_PATH}; judge requests attempted this run: {request_state['calls']}.")


if __name__ == "__main__":
    main()
