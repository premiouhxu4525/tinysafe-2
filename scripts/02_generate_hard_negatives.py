#!/usr/bin/env python3
"""
Step 2: Generate hard negatives (safe-but-edgy) with Sonnet.
Targets OR-Bench-style prompts the model might over-flag.
"""

import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic
from tqdm import tqdm

from src.utils import CATEGORIES, load_config, normalize_sample, save_jsonl

DATA_DIR = Path("data/raw")

EXAMPLES_PER_REQUEST = 15
MAX_WORKERS = 8

HARD_NEGATIVE_PROMPTS = {
    "violence": """Generate {n} examples of SAFE text that discusses violence-adjacent topics but is clearly NOT harmful. These should be texts that a safety classifier might incorrectly flag as violent.

Topics: war history, martial arts instruction, hunting regulations, medical trauma (ER/surgery), self-defense training, aggressive sports commentary, video game strategy, true crime analysis, boxing match commentary, military history documentaries.

Return ONLY a JSON array of {n} objects:
[{{"text": "...", "context": "history|martial_arts|hunting|medical|self_defense|sports|gaming|true_crime"}}]

Make realistic, varied-length examples (1-4 sentences). Every example must be unique and clearly safe.""",

    "sexual": """Generate {n} examples of SAFE text that discusses sex-adjacent topics but is clearly NOT harmful.

Topics: reproductive health education, anatomy/biology, romance fiction (non-explicit), sex ed curriculum, medical discussions (STIs, pregnancy), breastfeeding support, gender studies, dating advice, puberty education, OB-GYN visits.

Return ONLY a JSON array of {n} objects:
[{{"text": "...", "context": "health|anatomy|romance|education|medical|academic|dating"}}]

Make realistic, varied-length examples. Every example must be unique and clearly safe.""",

    "self_harm": """Generate {n} examples of SAFE text that discusses self-harm-adjacent topics but is clearly NOT harmful.

Topics: therapy/counseling discussions, harm reduction info, recovery narratives, mental health awareness, crisis hotline sharing, suicide prevention research, supporting friends through dark times, medication management, grief processing, therapist recommendations.

Return ONLY a JSON array of {n} objects:
[{{"text": "...", "context": "therapy|harm_reduction|recovery|awareness|crisis_support|research|support|medication"}}]

Make realistic, varied-length examples. Every example must be unique and clearly safe.""",

    "dangerous_info": """Generate {n} examples of SAFE text that discusses dangerous-info-adjacent topics but is clearly NOT harmful.

Topics: chemistry education (reactions, lab safety), defensive cybersecurity tutorials, locksmithing/lock sport, pharmacology education, physics demos, ethical hacking/CTF writeups, gun safety courses, fireworks regulations, kitchen knife safety, power tool instructions.

Return ONLY a JSON array of {n} objects:
[{{"text": "...", "context": "chemistry|cybersecurity|locksmithing|pharmacology|physics|ethical_hacking|gun_safety|regulations"}}]

Make realistic, varied-length examples. Every example must be unique and clearly safe.""",

    "harassment": """Generate {n} examples of SAFE text that might be confused with harassment but is clearly NOT harmful.

Topics: roast comedy between friends, competitive gaming trash talk, blunt professional feedback, heated political debate, constructive criticism, sarcastic humor, tough performance reviews, coach motivational speeches, sibling banter, stand-up comedy bits.

Return ONLY a JSON array of {n} objects:
[{{"text": "...", "context": "comedy|trash_talk|feedback|debate|criticism|sarcasm|review|coaching"}}]

Make realistic, varied-length examples. Every example must be unique and clearly safe.""",

    "hate": """Generate {n} examples of SAFE text that discusses hate-adjacent topics but is clearly NOT harmful.

Topics: cultural criticism/analysis, demographic statistics/research, historical discussion of discrimination, academic analysis of extremism, anti-racism education, comparative religion studies, immigration policy debate, sociological analysis of prejudice, civil rights history, diversity training materials.

Return ONLY a JSON array of {n} objects:
[{{"text": "...", "context": "cultural_criticism|statistics|history|academic|anti_racism|religion|policy|sociology"}}]

Make realistic, varied-length examples. Every example must be unique and clearly safe.""",

    "illegal_activity": """Generate {n} examples of SAFE text that discusses illegal-activity-adjacent topics but is clearly NOT harmful.

Topics: legal analysis of criminal cases, true crime discussion, drug legalization policy debate, criminal justice reform, forensic science education, law school exam questions, crime journalism, whistleblower protection, regulatory compliance discussions, court case summaries.

Return ONLY a JSON array of {n} objects:
[{{"text": "...", "context": "legal_analysis|true_crime|policy|reform|forensics|law_school|journalism|whistleblower"}}]

Make realistic, varied-length examples. Every example must be unique and clearly safe.""",
}


class CostTracker:
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
        self.failed_requests = 0
        self.start_time = time.time()
        self._lock = threading.Lock()

    def add(self, usage):
        with self._lock:
            self.total_input_tokens += usage.input_tokens
            self.total_output_tokens += usage.output_tokens
            self.total_requests += 1

    def add_failure(self):
        with self._lock:
            self.failed_requests += 1

    def report(self, prefix=""):
        elapsed = time.time() - self.start_time
        print(f"{prefix}  Requests: {self.total_requests} ({self.failed_requests} failed) | "
              f"Time: {elapsed/60:.1f}m")


def generate_batch(client, prompt, model, tracker, n=EXAMPLES_PER_REQUEST):
    prompt = prompt.format(n=n)
    for attempt in range(3):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4000,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.9,
            )
            tracker.add(response.usage)
            content = response.content[0].text.strip()
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            return json.loads(content)
        except json.JSONDecodeError:
            tracker.add_failure()
            if attempt < 2:
                time.sleep(1)
                continue
            return []
        except Exception as e:
            tracker.add_failure()
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            print(f"  Failed (API): {e}")
            return []


def main():
    config = load_config()
    hn_config = config["hard_negatives"]
    total_target = hn_config["total"]
    per_category = total_target // len(CATEGORIES)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    model = hn_config["model"]
    tracker = CostTracker()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Model: {model}")
    print(f"Target: {total_target} hard negatives (~{per_category} per category)")

    all_samples = []

    for category in CATEGORIES:
        prompt_template = HARD_NEGATIVE_PROMPTS[category]
        batches_needed = max(1, per_category // EXAMPLES_PER_REQUEST)

        print(f"\n{'='*60}")
        print(f"Generating ~{per_category} hard negatives for {category}")
        print(f"{'='*60}")

        cat_samples = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = [
                pool.submit(generate_batch, client, prompt_template, model, tracker, EXAMPLES_PER_REQUEST)
                for _ in range(batches_needed)
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"  {category}"):
                results = future.result()
                for item in results:
                    if "text" in item:
                        cat_samples.append(normalize_sample(
                            item["text"], "safe", source=f"hard_neg_{category}",
                        ))

        all_samples.extend(cat_samples)
        print(f"  Generated: {len(cat_samples)}")
        tracker.report(f"  After {category}:")

    save_jsonl(all_samples, DATA_DIR / "hard_negatives.jsonl")

    print(f"\n{'='*60}")
    print(f"DONE: {len(all_samples)} hard negatives")
    print(f"{'='*60}")
    for cat in CATEGORIES:
        count = sum(1 for s in all_samples if f"hard_neg_{cat}" in s.get("source", ""))
        print(f"  {cat}: {count}")
    tracker.report("Final:")


if __name__ == "__main__":
    main()
