"""
LLM-as-Judge layer for OCRBench v2.
Run AFTER eval.py — takes its output as input.
Adds judge_score + judge_reason to each sample.
"""

import os
import json
import time
import argparse
from tqdm import tqdm
from openai import OpenAI

# ── Config ───────────────────────────────────────────────────────────────────
OPENAI_API_KEY = "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"
JUDGE_MODEL    = "gpt-4o"



# ── Prompt ───────────────────────────────────────────────────────────────────
def build_judge_prompt(task_type: str, question: str, ground_truth, prediction: str) -> str:
    gt_str = ground_truth if isinstance(ground_truth, str) else json.dumps(ground_truth, ensure_ascii=False)

    return f"""You are an expert evaluator for OCR and document understanding tasks.

Task type: {task_type}
Question: {question}

Ground Truth:
{gt_str}

Model Prediction:
{prediction}

Evaluate whether the prediction is semantically correct and captures the key information from the ground truth.
Focus on content accuracy, NOT formatting differences (e.g. LaTeX vs plain text, extra explanation words, minor spacing).

Respond ONLY with a JSON object in this exact format (no markdown, no extra text):
{{"score": <float 0.0 to 1.0>, "reason": "<one sentence>"}}

Scoring guide:
- 1.0 : Perfect or semantically equivalent
- 0.7-0.9 : Mostly correct, minor omissions or formatting differences
- 0.4-0.6 : Partially correct, key info present but incomplete
- 0.1-0.3 : Mostly wrong but some relevant content
- 0.0 : Completely wrong or empty
"""

# ── Judge API call ────────────────────────────────────────────────────────────
def call_judge(client: OpenAI, prompt: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.0,
            )
            text = response.choices[0].message.content.strip()

            # Strip markdown fences if model adds them
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            result = json.loads(text)
            return {
                "judge_score": float(result.get("score", 0.0)),
                "judge_reason": result.get("reason", "")
            }

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {"judge_score": -1.0, "judge_reason": f"Judge failed: {e}"}

# ── Main inference loop ───────────────────────────────────────────────────────
def run_judge(input_path: str, output_path: str, task_filter: str = None,
              max_samples: int = None, resume: bool = True):

    with open(input_path, "r") as f:
        data = json.load(f)

    results = list(data)

    # Resume: reload already-judged items
    judged_indices = set()
    if resume and os.path.exists(output_path):
        with open(output_path, "r") as f:
            existing = json.load(f)
        for i, item in enumerate(existing):
            if "judge_score" in item:
                results[i] = item
                judged_indices.add(i)
        print(f"Resuming: {len(judged_indices)} items already judged.")

    client = OpenAI(api_key=OPENAI_API_KEY)
    count = 0

    for i, item in enumerate(tqdm(results)):
        task = item.get("type", "")

        # Skip if task filter is set and doesn't match
        if task_filter and task != task_filter:
            continue

        # Skip already judged
        if i in judged_indices:
            continue

        # Skip empty predictions
        pred = item.get("predict", "")
        if not isinstance(pred, str) or not pred.strip():
            item["judge_score"] = 0.0
            item["judge_reason"] = "Empty prediction"
            continue

        answers = item.get("answers", [""])
        gt = answers[0] if answers else ""

        prompt = build_judge_prompt(
            task_type=task,
            question=item.get("question", ""),
            ground_truth=gt,
            prediction=pred
        )

        judge_result = call_judge(client, prompt)
        item["judge_score"] = judge_result["judge_score"]
        item["judge_reason"] = judge_result["judge_reason"]
        count += 1

        # Save every 50 items
        if count % 50 == 0:
            with open(output_path, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"  Saved checkpoint at {count} judged items")

        if max_samples and count >= max_samples:
            print(f"Reached max_samples={max_samples}, stopping.")
            break

    # Final save
    with open(output_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Judged {count} new items. Output: {output_path}")
    return results

# ── Comparison report ─────────────────────────────────────────────────────────
def print_comparison_report(results: list):
    from collections import defaultdict

    task_stats = defaultdict(lambda: {"strict": [], "judge": [], "delta": []})

    for item in results:
        task   = item.get("type", "unknown")
        strict = item.get("score")
        judge  = item.get("judge_score")

        if strict is None or judge is None or judge < 0:
            continue

        task_stats[task]["strict"].append(strict)
        task_stats[task]["judge"].append(judge)
        task_stats[task]["delta"].append(judge - strict)

    print("\n" + "="*78)
    print(f"{'Task':<42} {'Strict':>8} {'Judge':>8} {'Delta':>8} {'N':>5}")
    print("="*78)

    for task in sorted(task_stats.keys()):
        s = task_stats[task]
        n = len(s["strict"])
        if n == 0:
            continue
        strict_avg = sum(s["strict"]) / n
        judge_avg  = sum(s["judge"]) / n
        delta_avg  = sum(s["delta"]) / n
        flag = " ⬆" if delta_avg > 0.1 else (" ⬇" if delta_avg < -0.05 else "")
        print(f"{task:<42} {strict_avg:>8.3f} {judge_avg:>8.3f} {delta_avg:>+8.3f}{flag}  {n:>4}")

    print("="*78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",  required=True,  help="Output of eval.py (has 'score' field)")
    parser.add_argument("--output_path", required=True,  help="Where to save judge-augmented JSON")
    parser.add_argument("--task_filter", default=None,   help="Only judge one task type e.g. 'formula recognition cn'")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap API calls for testing")
    parser.add_argument("--no_resume",   action="store_true",    help="Start fresh, ignore existing output")
    args = parser.parse_args()

    results = run_judge(
        input_path=args.input_path,
        output_path=args.output_path,
        task_filter=args.task_filter,
        max_samples=args.max_samples,
        resume=not args.no_resume,
    )

    print_comparison_report(results)
