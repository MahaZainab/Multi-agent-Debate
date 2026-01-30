#!/usr/bin/env python3
"""

Key features:
- Strict JSON output schema and non-empty prediction.
- One repair retry if output invalid.
- MCQ canonicalization/validation.
- Stopping: stop if predictions didn't change vs previous round OR max rounds.
- Majority vote aggregation.

Input formats supported:
1) JSON array file: [ {Problem, options, correct, ...}, ... ]
2) JSONL file: one object per line with same keys

Env vars:
  DATA_PATH      default: math_qa.json
  N_EXAMPLES     default: 10
  MAX_ROUNDS     default: 5
  OUT_JSONL      default: mmad_math_results.jsonl
  OUT_CSV        default: mmad_math_results.csv
  MODEL_1 / MODEL_2 / MODEL_3 override model IDs
"""

import csv
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class QAExample:
    id: str
    problem: str
    options: str
    correct: str
    rationale: str = ""
    category: str = ""


@dataclass
class AgentResponse:
    agent_name: str
    model_name: str
    round_id: int
    prediction: str
    reasoning: str
    raw_output: str
    parse_ok: bool
    repaired: bool = False


@dataclass
class TheoryOfMind:
    target_agent: str
    trust: int
    strengths: List[str]
    weaknesses: List[str]
    risk: str


@dataclass
class MMADResult:
    example_id: str
    gold: str
    baseline_predictions: List[str]
    round1_responses: List[AgentResponse]
    last_round_responses: List[AgentResponse]
    final_prediction: str
    num_rounds: int
    converged: bool


def _load_json_array(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _load_jsonl(path: str):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def load_dataset(path: str, limit: Optional[int] = None) -> List[QAExample]:
    if path.lower().endswith(".jsonl"):
        raw = _load_jsonl(path)
    else:
        raw = _load_json_array(path)

    exs: List[QAExample] = []
    for i, obj in enumerate(raw):
        exs.append(
            QAExample(
                id=str(obj.get("id", f"ex_{i}")),
                problem=str(obj.get("Problem", "")),
                options=str(obj.get("options", "")),
                correct=str(obj.get("correct", "")).strip().lower(),
                rationale=str(obj.get("Rationale", "")),
                category=str(obj.get("category", "")),
            )
        )
        if limit and len(exs) >= limit:
            break
    return exs


_MC_CHOICES = {"a", "b", "c", "d", "e"}

def normalize_letter(s: str) -> str:
    t = (s or "").strip().lower()
    m = re.search(r"\b([a-e])\b", t)
    if m:
        return m.group(1)
    return t[:1] if t[:1] in _MC_CHOICES else ""


class LocalTransformersAgent:
    def __init__(self, model_id: str, agent_id: int, max_new_tokens: int = 256):
        self.model_id = model_id
        self.agent_name = f"Agent-{agent_id}({model_id.split('/')[-1]})"
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        self.model.eval()

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        t = (text or "").strip()
        if t.startswith("```"):
            t = re.sub(r"^```[a-zA-Z]*\s*\n", "", t)
            t = re.sub(r"\n```$", "", t.strip())
        return t.strip()

    @staticmethod
    def _extract_first_json_object(text: str) -> Optional[dict]:
        t = LocalTransformersAgent._strip_markdown_fences(text)
        n = len(t)
        for i, ch in enumerate(t):
            if ch != "{":
                continue
            for j in range(n - 1, i, -1):
                if t[j] != "}":
                    continue
                cand = t[i:j + 1]
                try:
                    obj = json.loads(cand)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    pass
        return None

    def _generate_text(self, prompt: str, temperature: float = 0.2, top_p: float = 0.95) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    @staticmethod
    def _output_contract() -> str:
        return (
            "Return ONLY valid JSON (no markdown, no backticks, no extra text).\n"
            "JSON keys must be exactly: prediction, reasoning.\n"
            "prediction MUST be exactly one letter: a, b, c, d, or e.\n"
        )

    @classmethod
    def _round1_prompt(cls, ex: QAExample, show_rationale: bool) -> str:
        ctx = f"OPTIONS:\n{ex.options}\n"
        if show_rationale and ex.rationale:
            ctx += f"\nRATIONALE (treat as context):\n{ex.rationale}\n"
        return (
            "You are solving a multiple-choice math word problem.\n"
            "Choose the correct option letter.\n\n"
            f"{cls._output_contract()}\n"
            "PROBLEM:\n"
            f"{ex.problem}\n\n"
            f"{ctx}\n"
            "Return JSON now:\n"
        )

    @classmethod
    def _repair_prompt(cls, ex: QAExample, show_rationale: bool, prior_raw: str) -> str:
        ctx = f"OPTIONS:\n{ex.options}\n"
        if show_rationale and ex.rationale:
            ctx += f"\nRATIONALE (treat as context):\n{ex.rationale}\n"
        return (
            "Your previous response was invalid. Fix it.\n\n"
            f"{cls._output_contract()}\n"
            "PROBLEM:\n"
            f"{ex.problem}\n\n"
            f"{ctx}\n"
            "Your previous response was:\n"
            f"{prior_raw[:1000]}\n\n"
            "Return corrected JSON now:\n"
        )

    @classmethod
    def _tom_prompt(cls, ex: QAExample, other: AgentResponse) -> str:
        return (
            "You are modeling another agent's reliability for this MCQ math problem.\n\n"
            "Return ONLY JSON with keys: trust, strengths, weaknesses, risk.\n"
            "trust must be an integer 1-5. strengths/weaknesses are arrays (<=2 items).\n\n"
            "PROBLEM:\n"
            f"{ex.problem}\n\n"
            "OPTIONS:\n"
            f"{ex.options}\n\n"
            f"OTHER AGENT PREDICTION: {other.prediction}\n"
            f"OTHER AGENT REASONING: {other.reasoning[:600]}\n\n"
            "Return JSON now:\n"
        )

    @classmethod
    def _debate_prompt(cls, ex: QAExample, show_rationale: bool, prev_self: AgentResponse,
                      others: List[AgentResponse], toms: Dict[str, TheoryOfMind], round_id: int) -> str:
        ctx = f"OPTIONS:\n{ex.options}\n"
        if show_rationale and ex.rationale:
            ctx += f"\nRATIONALE (treat as context):\n{ex.rationale}\n"

        others_info = "\n\n".join(
            [f"Agent: {r.agent_name}\nPrediction: {r.prediction}\nReasoning: {r.reasoning[:350]}" for r in others]
        ) if others else "No other agents."

        tom_info = "\n\n".join(
            [f"ToM for {t.target_agent}: trust={t.trust}; strengths={t.strengths}; weaknesses={t.weaknesses}; risk={t.risk}"
             for t in toms.values()]
        ) if toms else "No ToM available."

        return (
            "You are in a multi-agent debate on a multiple-choice math problem.\n"
            "Re-compute the answer carefully and choose exactly one option letter.\n\n"
            f"{cls._output_contract()}\n"
            "DEBATE RULES:\n"
            "- If you disagree with others, identify the exact calculation step causing disagreement.\n"
            "- Prefer answers that map cleanly to one of the given options.\n\n"
            f"ROUND: {round_id}\n\n"
            "PROBLEM:\n"
            f"{ex.problem}\n\n"
            f"{ctx}\n"
            "YOUR PREVIOUS:\n"
            f"prediction={prev_self.prediction}; reasoning={prev_self.reasoning[:350]}\n\n"
            "OTHER AGENTS:\n"
            f"{others_info}\n\n"
            "YOUR THEORY OF MIND:\n"
            f"{tom_info}\n\n"
            "Return JSON now:\n"
        )

    def _parse_prediction_json(self, raw: str) -> Tuple[str, str, bool]:
        obj = self._extract_first_json_object(raw)
        if not isinstance(obj, dict):
            return "", self._strip_markdown_fences(raw)[:800], False

        if "prediction" not in obj:
            return "", str(obj)[:800], False

        pred = obj.get("prediction", "")
        if not isinstance(pred, str):
            pred = str(pred)

        pred_norm = normalize_letter(pred)
        if pred_norm not in _MC_CHOICES:
            return "", str(obj.get("reasoning", ""))[:800], False

        reasoning = obj.get("reasoning", "")
        if not isinstance(reasoning, str):
            reasoning = str(reasoning)
        reasoning = reasoning.strip() or self._strip_markdown_fences(raw).strip()

        return pred_norm, reasoning, True

    def _parse_tom(self, raw: str, target_agent: str) -> TheoryOfMind:
        obj = self._extract_first_json_object(raw)
        if not isinstance(obj, dict):
            return TheoryOfMind(target_agent, 3, [], [], "Unable to assess")
        trust = obj.get("trust", 3)
        try:
            trust = int(trust)
        except Exception:
            trust = 3
        trust = max(1, min(5, trust))
        strengths = obj.get("strengths", [])
        weaknesses = obj.get("weaknesses", [])
        if not isinstance(strengths, list):
            strengths = []
        if not isinstance(weaknesses, list):
            weaknesses = []
        risk = obj.get("risk", "")
        return TheoryOfMind(target_agent, trust, [str(x) for x in strengths][:2], [str(x) for x in weaknesses][:2], str(risk)[:300])

    def _generate_and_parse(self, ex: QAExample, show_rationale: bool, prompt: str, allow_repair: bool = True):
        raw = self._generate_text(prompt, temperature=0.2)
        pred, reasoning, ok = self._parse_prediction_json(raw)
        if ok:
            return pred, reasoning, raw, True, False
        if not allow_repair:
            return pred, reasoning, raw, False, False

        rep_prompt = self._repair_prompt(ex, show_rationale, raw)
        rep_raw = self._generate_text(rep_prompt, temperature=0.2)
        rep_pred, rep_reasoning, rep_ok = self._parse_prediction_json(rep_raw)
        return rep_pred, rep_reasoning, rep_raw, rep_ok, True

    def initial_response(self, ex: QAExample, show_rationale: bool) -> AgentResponse:
        prompt = self._round1_prompt(ex, show_rationale)
        pred, reasoning, raw, ok, repaired = self._generate_and_parse(ex, show_rationale, prompt, allow_repair=True)
        return AgentResponse(self.agent_name, self.model_id, 1, pred, reasoning, raw, ok, repaired)

    def construct_tom(self, ex: QAExample, target: AgentResponse) -> TheoryOfMind:
        prompt = self._tom_prompt(ex, target)
        raw = self._generate_text(prompt, temperature=0.2)
        return self._parse_tom(raw, target.agent_name)

    def debate_response(self, ex: QAExample, show_rationale: bool, prev_self: AgentResponse,
                        others: List[AgentResponse], toms: Dict[str, TheoryOfMind], round_id: int) -> AgentResponse:
        prompt = self._debate_prompt(ex, show_rationale, prev_self, others, toms, round_id)
        pred, reasoning, raw, ok, repaired = self._generate_and_parse(ex, show_rationale, prompt, allow_repair=True)
        return AgentResponse(self.agent_name, self.model_id, round_id, pred, reasoning, raw, ok, repaired)


class HeuristicDecider:
    def should_continue(self, prev: Optional[List[AgentResponse]], curr: List[AgentResponse],
                        current_round: int, max_rounds: int) -> bool:
        if current_round >= max_rounds:
            return False
        if prev is None:
            return True
        prev_map = {r.agent_name: r.prediction for r in prev}
        curr_map = {r.agent_name: r.prediction for r in curr}
        return prev_map != curr_map


class MMADFramework:
    def __init__(self, model_ids: List[str], max_rounds: int = 5, show_rationale: bool = False):
        self.agents = [LocalTransformersAgent(mid, i + 1) for i, mid in enumerate(model_ids)]
        self.decider = HeuristicDecider()
        self.max_rounds = max_rounds
        self.show_rationale = show_rationale

    def solve(self, ex: QAExample, verbose: bool = True) -> MMADResult:
        if verbose:
            print(f"\n{'='*80}")
            print(f"MMAD(Math-MCQ): Solving {ex.id}  category={ex.category}")
            print(f"{'='*80}")
            print(f"Problem: {ex.problem}")
            print(f"Options: {ex.options}")
            print(f"Gold: {ex.correct}\n")

        if verbose:
            print("ROUND 1: Independent")
            print("-" * 80)

        round1: List[AgentResponse] = []
        for ag in self.agents:
            if verbose:
                print(f"  {ag.agent_name} thinking...")
            r = ag.initial_response(ex, self.show_rationale)
            round1.append(r)
            if verbose:
                print(f"    Prediction: {r.prediction}  ParseOK={r.parse_ok}  Repaired={r.repaired}")

        baseline_preds = [r.prediction for r in round1]
        current = round1
        prev_round: Optional[List[AgentResponse]] = None
        round_id = 2

        while round_id <= self.max_rounds and self.decider.should_continue(prev_round, current, round_id - 1, self.max_rounds):
            if verbose:
                print(f"\nROUND {round_id}: Debate + ToM")
                print("-" * 80)

            new_round: List[AgentResponse] = []
            for ag in self.agents:
                prev_self = next(r for r in current if r.agent_name == ag.agent_name)
                others = [r for r in current if r.agent_name != ag.agent_name]

                toms: Dict[str, TheoryOfMind] = {}
                for o in others:
                    toms[o.agent_name] = ag.construct_tom(ex, o)

                nr = ag.debate_response(ex, self.show_rationale, prev_self, others, toms, round_id)
                new_round.append(nr)
                if verbose:
                    print(f"  {ag.agent_name} -> {nr.prediction}  (ParseOK={nr.parse_ok}, Repaired={nr.repaired})")

            prev_round = current
            current = new_round
            round_id += 1

        final_prediction = self._aggregate(current)
        converged = len(set(r.prediction for r in current if r.prediction)) == 1
        num_rounds = round_id - 1

        if verbose:
            print(f"\n{'='*80}")
            print("FINAL")
            print(f"{'='*80}")
            print(f"Final Prediction: {final_prediction}")
            print(f"Gold:             {ex.correct}")
            print(f"Correct:          {final_prediction == ex.correct}")
            print(f"Rounds:           {num_rounds}")
            print(f"Converged:        {converged}\n")

        return MMADResult(ex.id, ex.correct, baseline_preds, round1, current, final_prediction, num_rounds, converged)

    @staticmethod
    def _aggregate(responses: List[AgentResponse]) -> str:
        usable = [r.prediction for r in responses if r.prediction in _MC_CHOICES]
        if not usable:
            return ""
        return Counter(usable).most_common(1)[0][0]


def compute_accuracy(results: List[MMADResult]) -> Tuple[float, float, float]:
    if not results:
        return 0.0, 0.0, 0.0
    total = len(results)

    baseline_correct = 0
    for r in results:
        votes = Counter([p for p in r.baseline_predictions if p in _MC_CHOICES])
        baseline_pred = votes.most_common(1)[0][0] if votes else ""
        baseline_correct += int(baseline_pred == r.gold)

    mmad_correct = sum(1 for r in results if r.final_prediction == r.gold)

    oracle = 0
    for r in results:
        oracle += int(any(p == r.gold for p in r.baseline_predictions))

    return baseline_correct / total, mmad_correct / total, oracle / total


def save_results_csv(path: str, results: List[MMADResult]):
    fields = ["id", "gold", "baseline_pred", "mmad_pred", "baseline_correct", "mmad_correct", "num_rounds", "converged"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            votes = Counter([p for p in r.baseline_predictions if p in _MC_CHOICES])
            baseline_pred = votes.most_common(1)[0][0] if votes else ""
            w.writerow({
                "id": r.example_id,
                "gold": r.gold,
                "baseline_pred": baseline_pred,
                "mmad_pred": r.final_prediction,
                "baseline_correct": baseline_pred == r.gold,
                "mmad_correct": r.final_prediction == r.gold,
                "num_rounds": r.num_rounds,
                "converged": r.converged,
            })


def save_results_jsonl(path: str, results: List[MMADResult]):
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            obj = {
                "id": r.example_id,
                "gold": r.gold,
                "baseline_predictions": r.baseline_predictions,
                "final_prediction": r.final_prediction,
                "num_rounds": r.num_rounds,
                "converged": r.converged,
                "round1": [
                    {"agent": x.agent_name, "prediction": x.prediction, "parse_ok": x.parse_ok, "repaired": x.repaired,
                     "reasoning": (x.reasoning or "")[:800]}
                    for x in r.round1_responses
                ],
                "last_round": [
                    {"agent": x.agent_name, "prediction": x.prediction, "parse_ok": x.parse_ok, "repaired": x.repaired,
                     "reasoning": (x.reasoning or "")[:800]}
                    for x in r.last_round_responses
                ],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    data_path = os.environ.get("DATA_PATH", "math_qa.json")
    n_examples = int(os.environ.get("N_EXAMPLES", "10"))
    max_rounds = int(os.environ.get("MAX_ROUNDS", "5"))
    show_rationale = os.environ.get("SHOW_RATIONALE", "0").strip() == "1"

    model_ids = [
        os.environ.get("MODEL_1", "Qwen/Qwen2.5-Coder-1.5B-Instruct"),
        os.environ.get("MODEL_2", "microsoft/Phi-3-mini-4k-instruct"),
        os.environ.get("MODEL_3", "deepseek-ai/deepseek-coder-1.3b-instruct"),
    ]

    print("=" * 80)
    print("MMAD - Math MCQ QA (local transformers)")
    print("=" * 80)
    print(f"DATA_PATH={data_path}")
    print(f"N_EXAMPLES={n_examples}  MAX_ROUNDS={max_rounds}  SHOW_RATIONALE={int(show_rationale)}")
    print("Models:")
    for m in model_ids:
        print("  -", m)
    print()

    dataset = load_dataset(data_path, limit=n_examples)
    print(f"Loaded {len(dataset)} examples\n")

    mmad = MMADFramework(model_ids=model_ids, max_rounds=max_rounds, show_rationale=show_rationale)

    results: List[MMADResult] = []
    for i, ex in enumerate(dataset, start=1):
        print(f"\n{'#'*80}\nProcessing {i}/{len(dataset)}\n{'#'*80}")
        results.append(mmad.solve(ex, verbose=True))

    bacc, macc, oacc = compute_accuracy(results)
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Examples: {len(results)}")
    print(f"Baseline majority@R1: {bacc:.3f}")
    print(f"MMAD (debate):        {macc:.3f}")
    print(f"Oracle@R1:            {oacc:.3f}")
    print(f"Improvement:          {(macc - bacc):.3f}")
    print("=" * 80)

    out_csv = os.environ.get("OUT_CSV", "mmad_math_results.csv")
    out_jsonl = os.environ.get("OUT_JSONL", "mmad_math_results.jsonl")
    save_results_csv(out_csv, results)
    save_results_jsonl(out_jsonl, results)
    print(f"\nSaved CSV:   {out_csv}")
    print(f"Saved JSONL: {out_jsonl}")


if __name__ == "__main__":
    main()
