#!/usr/bin/env python3
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import ollama


# =============================
# 1) Dataset
# =============================

@dataclass
class CodeExample:
    id: Any
    code: str
    input: str
    output: str  # ground truth (optional for later eval)

def load_test_jsonl(path: str) -> List[CodeExample]:
    rows: List[CodeExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(
                CodeExample(
                    id=obj.get("id"),
                    code=obj["code"],
                    input=obj["input"],
                    output=obj.get("output", ""),
                )
            )
    return rows


# =============================
# 2) Agent output contract
# =============================

@dataclass
class PrepOutput:
    agent_name: str
    answer: str
    confidence: float   # 0..1
    reasoning: str
    raw_output: str


class ExecutionAgent:
    """
    Debate-Preparation agent for code execution tasks.
    Mirrors the Ollama usage and JSON parsing style in debate.py.  (same pattern)
    """

    def __init__(self, model_name: str, agent_id: int, role_hint: str, max_tokens: int = 512):
        self.model_name = model_name
        self.agent_name = f"Agent-{agent_id}({model_name})"
        self.role_hint = role_hint
        self.max_tokens = max_tokens

    # ----- helpers (same spirit as your debate.py) -----

    @staticmethod
    def _safe_float(x, default: float = 0.5) -> float:
        try:
            v = float(x)
            if math.isnan(v) or math.isinf(v):
                return default
            return max(0.0, min(1.0, v))
        except Exception:
            return default

    @staticmethod
    def _extract_json_block(text: str) -> str:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return text
        return text[start : end + 1]

    def _call_ollama(self, prompt: str) -> str:
        res = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            options={"num_predict": self.max_tokens},
            stream=False,
        )
        return res["response"]

    def _parse_output(self, raw_output: str) -> Tuple[str, float, str]:
        """
        Expect strict JSON:
          {
            "answer": "...",
            "confidence": 0..1,
            "reasoning": "..."
          }
        Fallbacks if model misbehaves.
        """
        json_block = self._extract_json_block(raw_output)
        answer = ""
        confidence = 0.5
        reasoning = json_block.strip()

        try:
            obj = json.loads(json_block)
            if isinstance(obj, dict):
                answer = str(obj.get("answer", answer))
                confidence = self._safe_float(obj.get("confidence", confidence))
                reasoning = str(obj.get("reasoning", reasoning))
        except Exception:
            pass

        return answer, confidence, reasoning

    # ----- prompt builder for Debate Preparation -----

    def _prep_prompt(self, ex: CodeExample) -> str:
        # IMPORTANT: we’re not executing code here, we’re asking the model to trace.
        # We require: answer + confidence + reasoning in JSON.
        return "\n".join(
            [
                "You are one agent in a multi-agent reasoning system.",
                f"Role: {self.role_hint}",
                "",
                "Task: Trace/execute the following Python code on the given input.",
                "Return the FINAL output value as your answer.",
                "",
                "Output format requirement:",
                "Return STRICT JSON with keys:",
                '  "answer": the final output (as a string; if it is a list/tuple/dict you may JSON-format it)',
                '  "confidence": number 0 to 1',
                '  "reasoning": a clear step-by-step trace explaining how you got the output',
                "",
                "### Code",
                ex.code,
                "",
                "### Input",
                ex.input,
                "",
                "JSON:",
            ]
        )

    # ----- public API -----

    def prepare(self, ex: CodeExample) -> PrepOutput:
        prompt = self._prep_prompt(ex)
        raw = self._call_ollama(prompt)
        ans, conf, reasoning = self._parse_output(raw)
        return PrepOutput(
            agent_name=self.agent_name,
            answer=ans,
            confidence=conf,
            reasoning=reasoning,
            raw_output=raw,
        )


# =============================
# 3) Debate Preparation runner
# =============================

def debate_preparation(ex: CodeExample, agents: List[ExecutionAgent]) -> List[PrepOutput]:
    # Independent, no peer visibility in this phase (as required by §2.1.2)
    return [ag.prepare(ex) for ag in agents]


# =============================
# 4) Example main
# =============================

def main():
    data = load_test_jsonl("test.jsonl")
    ex = data[0]

    agents = [
        ExecutionAgent("deepseek-r1:1.5b", 1, role_hint="Direct tracer: execute step-by-step, track variables."),
        ExecutionAgent("gemma3:1b",       2, role_hint="Checker: compute output then sanity-check for mistakes."),
        ExecutionAgent("llama3.2:1b",     3, role_hint="Skeptic: look for Python gotchas, types, mutation, edge cases."),
    ]

    prep = debate_preparation(ex, agents)

    print("=" * 80)
    print("Example id:", ex.id)
    print("Gold output (if present):", ex.output)
    print("=" * 80)
    for out in prep:
        print(f"\n{out.agent_name}")
        print("answer     :", out.answer)
        print("confidence :", f"{out.confidence:.2f}")
        print("reasoning  :", out.reasoning[:500])


if __name__ == "__main__":
    main()
