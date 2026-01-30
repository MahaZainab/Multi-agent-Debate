# Importing libraries and modules
import argparse
import json
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# For MMAD we need to store predictions, ToM models, and debate results. For that, we define data classes.
# Configrations and data classes for MMAD framework
@dataclass
class AgentPrediction:
    """Single agent's prediction with metadata"""
    agent_id: int
    agent_name: str
    prediction: str
    confidence: float  # 0.0 to 1.0
    reasoning_trace: str
    round_num: int
    timestamp: float
    tokens_used: int = 0


@dataclass
class TheoryOfMindModel:
    """Agent's model of another agent's thinking"""
    target_agent_id: int
    strengths: List[str]
    weaknesses: List[str]
    reasoning_patterns: List[str]
    bias_tendencies: List[str]
    reliability_score: float  # 0.0 to 1.0


@dataclass
class MutualToM:
    """Mutual Theory of Mind state for all agents"""
    round_num: int
    tom_models: Dict[int, Dict[int, TheoryOfMindModel]]  # agent_id -> {other_agent_id -> ToM}
    convergence_score: float
    disagreement_points: List[str]
    consensus_points: List[str]


@dataclass
class DebateConfig:
    """Configuration for MMAD debate"""
    max_rounds: int = 5
    convergence_threshold: float = 0.90
    min_confidence_delta: float = 0.05
    enable_mtom: bool = True
    enable_decider: bool = True

    # Agent models (HF model IDs)
    models: List[str] = field(default_factory=lambda: [
        "Qwen/Qwen2.5-7B-Instruct",                 # agent 0
        "mistralai/Mistral-7B-Instruct-v0.3",       # agent 1
        "deepseek-ai/deepseek-coder-6.7b-instruct"  # agent 2
    ])


    # Generation controls
    temperature: float = 0.3 # lower for more focused responses
    top_p: float = 0.95 # standard for reasoning tasks
    max_new_tokens: int = 512  # max tokens per generation # As we are passing reasoning for ToM, we need more tokens
    do_sample: bool = False   # deterministic by default for eval

    # Backend controls
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"  # "bfloat16" or "float16" or "float32"


@dataclass
class DebateResult:
    """Complete debate result with all metadata"""
    question: str
    final_answer: str
    ground_truth: Any
    correct: bool
    debate_history: List[AgentPrediction]
    mtom_history: List[MutualToM]
    num_rounds: int
    total_tokens: int
    total_time: float
    convergence_achieved: bool
    config: DebateConfig
    item_id: Optional[str] = None
    task: Optional[str] = None



# HUGGING FACE CLIENT


class HFClient:
    """Local HF transformers client with caching per model_id."""

    def __init__(self, device_map: str = "auto", torch_dtype: str = "bfloat16"):
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self._cache: Dict[str, Tuple[AutoTokenizer, AutoModelForCausalLM]] = {}

    def _dtype(self):
        if self.torch_dtype == "float16":
            return torch.float16
        if self.torch_dtype == "float32":
            return torch.float32
        # default bfloat16
        return torch.bfloat16

    def load(self, model_id: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        if model_id in self._cache:
            return self._cache[model_id]

        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=self.device_map,
            torch_dtype=self._dtype() if torch.cuda.is_available() else torch.float32
        )
        model.eval()
        self._cache[model_id] = (tokenizer, model)
        return tokenizer, model

    def generate(self, model: str, prompt: str, temperature: float = 0.7,
                 max_tokens: int = 1024, top_p: float = 0.95,
                 do_sample: bool = True) -> Dict[str, Any]:
        """Generate response from HF model (local)."""
        tok, mdl = self.load(model)

        inputs = tok(prompt, return_tensors="pt").to(mdl.device)
        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            pad_token_id=tok.eos_token_id,
            do_sample=do_sample,
        )
        if do_sample:
            gen_kwargs.update(dict(temperature=temperature, top_p=top_p))

        with torch.no_grad():
            out = mdl.generate(**inputs, **gen_kwargs)

        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        text = tok.decode(gen_ids, skip_special_tokens=True).strip()

        # tokens_used is approximate; use generated token count
        tokens_used = int(gen_ids.shape[0]) if hasattr(gen_ids, "shape") else 0
        return {"response": text, "tokens": tokens_used, "success": True}



# PROMPT TEMPLATES WITH ToM CAPABILITIES


class MMADPrompts:
    """Prompt templates for MMAD framework"""

    @staticmethod
    def initial_prediction(question: str, code: str, agent_name: str) -> str:
        return f"""You are {agent_name}, an expert code reasoning agent.

Code:
```python
{code}
```

Question: {question}

Your task:
1. Analyze the code carefully
2. Provide your prediction/answer
3. Explain your reasoning step-by-step
4. Assign a confidence score (0.0-1.0) to your answer

Format your response EXACTLY as:
PREDICTION: [your answer]
CONFIDENCE: [0.0-1.0]
REASONING:
[step 1]
[step 2]
...

Your response:"""

    @staticmethod
    def cross_agent_analysis(question: str, code: str, my_pred: AgentPrediction,
                            other_preds: List[AgentPrediction], agent_name: str) -> str:
        others_text = "\n\n".join([
            f"Agent {p.agent_name}:\n"
            f"  Prediction: {p.prediction}\n"
            f"  Confidence: {p.confidence:.2f}\n"
            f"  Reasoning: {p.reasoning_trace}"
            for p in other_preds
        ])

        return f"""You are {agent_name}. You are in a collaborative debate with other expert agents.

Code:
```python
{code}
```

Question: {question}

YOUR PREVIOUS RESPONSE:
Prediction: {my_pred.prediction}
Confidence: {my_pred.confidence:.2f}
Reasoning: {my_pred.reasoning_trace}

OTHER AGENTS' RESPONSES:
{others_text}

CROSS-AGENT ANALYSIS TASK:
1. Compare your reasoning with others
2. Identify where others agree or disagree with you
3. Spot potential errors in others' reasoning
4. Find insights you might have missed
5. Evaluate alternative reasoning paths

Provide your analysis:"""

    @staticmethod
    def construct_mtom(question: str, code: str, my_pred: AgentPrediction,
                      other_preds: List[AgentPrediction], analysis: str,
                      agent_name: str) -> str:
        others_text = "\n\n".join([
            f"Agent {p.agent_name}:\n"
            f"  Prediction: {p.prediction}\n"
            f"  Confidence: {p.confidence:.2f}\n"
            f"  Reasoning: {p.reasoning_trace}"
            for p in other_preds
        ])

        return f"""You are {agent_name}. Construct a Theory of Mind model for each other agent.

Code:
```python
{code}
```

Question: {question}

OTHER AGENTS' RESPONSES:
{others_text}

YOUR ANALYSIS:
{analysis}

THEORY OF MIND CONSTRUCTION:
For EACH other agent, identify:
1. STRENGTHS
2. WEAKNESSES
3. PATTERNS
4. BIASES
5. RELIABILITY (0.0-1.0)

Format for EACH agent:
AGENT: [agent_name]
STRENGTHS: [comma-separated]
WEAKNESSES: [comma-separated]
PATTERNS: [comma-separated]
BIASES: [comma-separated]
RELIABILITY: [0.0-1.0]
---

Your Theory of Mind models:"""

    @staticmethod
    def self_update_with_mtom(question: str, code: str, my_pred: AgentPrediction,
                             tom_models: Dict[int, TheoryOfMindModel],
                             other_preds: List[AgentPrediction],
                             agent_name: str, round_num: int) -> str:
        tom_summary = "\n\n".join([
            f"Your ToM Model of Agent {model.target_agent_id}:\n"
            f"  Strengths: {', '.join(model.strengths)}\n"
            f"  Weaknesses: {', '.join(model.weaknesses)}\n"
            f"  Reliability: {model.reliability_score:.2f}"
            for model in tom_models.values()
        ]) or "[No ToM models available]"

        others_text = "\n\n".join([
            f"Agent {p.agent_name}: {p.prediction} (confidence: {p.confidence:.2f})"
            for p in other_preds
        ]) or "[No other agents]"

        return f"""You are {agent_name}, Round {round_num}.

Code:
```python
{code}
```

Question: {question}

YOUR CURRENT ANSWER:
Prediction: {my_pred.prediction}
Confidence: {my_pred.confidence:.2f}
Reasoning: {my_pred.reasoning_trace}

OTHER AGENTS' CURRENT ANSWERS:
{others_text}

YOUR THEORY OF MIND MODELS:
{tom_summary}

SELF-UPDATE WITH MUTUAL ToM:
Based on your understanding of other agents' minds:
1. Should you revise your answer? Why or why not?
2. Which agent(s) have reliable reasoning you should consider?
3. What errors should you correct in your own thinking?
4. What superior reasoning strategies should you adopt?
5. Where should you strengthen your argument?

Provide your UPDATED response:

Format:
PREDICTION: [updated answer]
CONFIDENCE: [updated 0.0-1.0]
REASONING:
[updated step-by-step reasoning incorporating insights from ToM]
CHANGES_MADE: [explain what you changed and why]

Your updated response:"""

    @staticmethod
    def decider_evaluation(question: str, code: str, all_predictions: List[AgentPrediction],
                          round_num: int, max_rounds: int) -> str:
        current_answers = {}
        for pred in all_predictions:
            if pred.round_num == round_num:
                current_answers[pred.agent_name] = {
                    'prediction': pred.prediction,
                    'confidence': pred.confidence
                }

        summary = "\n".join([
            f"{name}: {data['prediction']} (confidence: {data['confidence']:.2f})"
            for name, data in current_answers.items()
        ]) or "[No predictions]"

        return f"""You are the Decider Agent evaluating debate progress.

Code:
```python
{code}
```

Question: {question}

Round: {round_num}/{max_rounds}

CURRENT AGENT PREDICTIONS:
{summary}

DECISION CRITERIA:
1. Are predictions converging (similar answers)?
2. Are confidence scores stabilizing?
3. Is further debate likely to improve the answer?
4. Have we reached maximum rounds?

Provide your decision:

Format:
CONTINUE_DEBATE: [YES or NO]
REASON: [explain your decision]
CONVERGENCE_SCORE: [0.0-1.0, where 1.0 is full convergence]

Your decision:"""

    @staticmethod
    def final_synthesis(question: str, code: str, all_predictions: List[AgentPrediction],
                       mtom_history: List[MutualToM]) -> str:
        final_round = max(p.round_num for p in all_predictions)
        final_preds = [p for p in all_predictions if p.round_num == final_round]

        preds_text = "\n".join([
            f"{p.agent_name}: {p.prediction} (confidence: {p.confidence:.2f})"
            for p in final_preds
        ])

        return f"""Synthesize the final answer from multi-agent debate with Theory of Mind.

Code:
```python
{code}
```

Question: {question}

FINAL AGENT PREDICTIONS:
{preds_text}

DEBATE ROUNDS: {final_round + 1}

SYNTHESIS TASK:
1. Identify the consensus answer
2. Consider confidence levels
3. Evaluate reasoning quality
4. Produce the most accurate final answer

Format:
FINAL_ANSWER: [synthesized answer]
CONFIDENCE: [0.0-1.0]
JUSTIFICATION: [explain why this is the best answer]

Your synthesis:"""



# MMAD AGENT WITH ToM CAPABILITIES
class MMADAgent:
    """Individual agent in MMAD framework with ToM capabilities"""

    def __init__(self, agent_id: int, model_name: str, client: HFClient):
        self.agent_id = agent_id
        self.model_name = model_name
        self.agent_name = f"Agent_{agent_id}_{model_name.split('/')[-1].split(':')[0]}"
        self.client = client
        self.tom_models: Dict[int, TheoryOfMindModel] = {}

    def generate_initial_prediction(self, question: str, code: str,
                                   round_num: int = 0,
                                   temperature: float = 0.3,
                                   max_tokens: int = 512,
                                   top_p: float = 0.95,
                                   do_sample: bool = False) -> AgentPrediction:
        prompt = MMADPrompts.initial_prediction(question, code, self.agent_name)
        result = self.client.generate(
            self.model_name, prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            do_sample=do_sample
        )
        prediction, confidence, reasoning = self._parse_prediction_response(result["response"])
        return AgentPrediction(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            prediction=prediction,
            confidence=confidence,
            reasoning_trace=reasoning,
            round_num=round_num,
            timestamp=time.time(),
            tokens_used=result["tokens"]
        )

    def construct_tom_models(self, question: str, code: str, my_pred: AgentPrediction,
                            other_preds: List[AgentPrediction],
                            temperature: float,
                            max_tokens: int,
                            top_p: float,
                            do_sample: bool) -> Dict[int, TheoryOfMindModel]:
        analysis_prompt = MMADPrompts.cross_agent_analysis(question, code, my_pred, other_preds, self.agent_name)
        analysis_result = self.client.generate(
            self.model_name, analysis_prompt, temperature=temperature, max_tokens=max_tokens,
            top_p=top_p, do_sample=do_sample
        )
        analysis = analysis_result["response"]

        tom_prompt = MMADPrompts.construct_mtom(question, code, my_pred, other_preds, analysis, self.agent_name)
        tom_result = self.client.generate(
            self.model_name, tom_prompt, temperature=temperature, max_tokens=max_tokens,
            top_p=top_p, do_sample=do_sample
        )

        tom_models = self._parse_tom_response(tom_result["response"], other_preds)
        self.tom_models = tom_models
        return tom_models

    def update_with_mtom(self, question: str, code: str, my_pred: AgentPrediction,
                        other_preds: List[AgentPrediction],
                        round_num: int,
                        temperature: float,
                        max_tokens: int,
                        top_p: float,
                        do_sample: bool) -> AgentPrediction:
        prompt = MMADPrompts.self_update_with_mtom(
            question, code, my_pred, self.tom_models, other_preds, self.agent_name, round_num
        )
        result = self.client.generate(
            self.model_name, prompt,
            temperature=temperature, max_tokens=max_tokens, top_p=top_p, do_sample=do_sample
        )
        prediction, confidence, reasoning = self._parse_prediction_response(result["response"])
        return AgentPrediction(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            prediction=prediction,
            confidence=confidence,
            reasoning_trace=reasoning,
            round_num=round_num,
            timestamp=time.time(),
            tokens_used=result["tokens"]
        )

    def _parse_prediction_response(self, response: str) -> Tuple[str, float, str]:
        lines = response.strip().split('\n')
        prediction = ""
        confidence = 0.5
        reasoning = ""
        current_section = None

        for line in lines:
            line_upper = line.upper().strip()
            if line_upper.startswith("PREDICTION:"):
                prediction = line.split(":", 1)[1].strip()
                current_section = None
            elif line_upper.startswith("CONFIDENCE:"):
                try:
                    conf_str = line.split(":", 1)[1].strip()
                    confidence = float(conf_str)
                    confidence = max(0.0, min(1.0, confidence))
                except Exception:
                    confidence = 0.5
                current_section = None
            elif line_upper.startswith("REASONING:"):
                current_section = "reasoning"
                reasoning = ""
            elif current_section == "reasoning" and line.strip():
                reasoning += line + "\n"

        if not prediction:
            # fallback: last non-empty line not containing headers
            for line in reversed(lines):
                if line.strip() and not any(x in line.upper() for x in ["PREDICTION", "CONFIDENCE", "REASONING"]):
                    prediction = line.strip()
                    break

        return prediction.strip(), confidence, reasoning.strip()

    def _parse_tom_response(self, response: str, other_preds: List[AgentPrediction]) -> Dict[int, TheoryOfMindModel]:
        tom_models = {}
        sections = response.split("---")
        for section in sections:
            if not section.strip():
                continue
            lines = section.strip().split('\n')
            target_agent = None
            strengths, weaknesses, patterns, biases = [], [], [], []
            reliability = 0.5

            for line in lines:
                line_upper = line.upper().strip()
                if line_upper.startswith("AGENT:"):
                    target_name = line.split(":", 1)[1].strip()
                    for pred in other_preds:
                        if pred.agent_name in target_name or target_name in pred.agent_name:
                            target_agent = pred.agent_id
                            break
                elif line_upper.startswith("STRENGTHS:"):
                    strengths = [s.strip() for s in line.split(":", 1)[1].split(",") if s.strip()]
                elif line_upper.startswith("WEAKNESSES:"):
                    weaknesses = [w.strip() for w in line.split(":", 1)[1].split(",") if w.strip()]
                elif line_upper.startswith("PATTERNS:"):
                    patterns = [p.strip() for p in line.split(":", 1)[1].split(",") if p.strip()]
                elif line_upper.startswith("BIASES:"):
                    biases = [b.strip() for b in line.split(":", 1)[1].split(",") if b.strip()]
                elif line_upper.startswith("RELIABILITY:"):
                    try:
                        reliability = float(line.split(":", 1)[1].strip())
                        reliability = max(0.0, min(1.0, reliability))
                    except Exception:
                        reliability = 0.5

            if target_agent is not None:
                tom_models[target_agent] = TheoryOfMindModel(
                    target_agent_id=target_agent,
                    strengths=strengths or ["general reasoning"],
                    weaknesses=weaknesses or ["none identified"],
                    reasoning_patterns=patterns or ["standard approach"],
                    bias_tendencies=biases or ["none detected"],
                    reliability_score=reliability
                )
        return tom_models


# ===============================================
# TASK ADAPTERS: MathQA + CruxEval
# ===============================================

class Task(str, Enum):
    MATHQA = "mathqa"
    CRUXEVAL = "cruxeval"


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def build_prompt_for_task(task: Task, item: dict) -> Tuple[str, str]:
    """
    Returns (question, code_text_for_prompt).
    We reuse your 'code' prompt slot:
      - For MathQA, we place (problem + options) in the code block and ask for option letter.
      - For CruxEval, we use code + input and ask for output.
    """
    if task == Task.MATHQA:
        problem = item.get("Problem", "")
        options = item.get("options", "")
        question = "Choose the correct option letter (a/b/c/d/e)."
        code = f"PROBLEM:\n{problem}\n\nOPTIONS:\n{options}\n\nReturn only the option letter."
        return question, code

    code = item.get("code", "")
    inp = item.get("input", "")
    question = "What is the returned value / printed output for the given input?"
    code_text = f"{code}\n\n# INPUT:\n{inp}\n\n# Return the exact output as a Python literal."
    return question, code_text


def extract_task_answer(task: Task, text: str) -> str:
    """
    Maps the framework's free-form final_answer into:
      - MathQA: option letter a-e
      - CruxEval: raw string
    """
    if task == Task.MATHQA:
        t = (text or "").lower()
        m = re.search(r"\b([a-e])\b", t)
        return m.group(1) if m else ""
    return (text or "").strip()


def check_task_correct(task: Task, pred: str, item: dict) -> Optional[bool]:
    if task == Task.MATHQA:
        gold = (item.get("correct", "") or "").strip().lower()
        if not gold:
            return None
        return extract_task_answer(task, pred) == gold

    gold = item.get("output", None)
    if gold is None:
        return None
    return _norm_ws(pred) == _norm_ws(str(gold))


# MMAD FRAMEWORK 

class MMADFramework:
    """Main Multi-Agent Mutual Awareness Debate Framework"""

    def __init__(self, config: DebateConfig, client: HFClient):
        self.config = config
        self.client = client
        self.agents = [MMADAgent(i, model, client) for i, model in enumerate(config.models)]

    def run_debate(self, question: str, code: str, ground_truth: Any = None,
                   item_id: Optional[str] = None, task: Optional[str] = None) -> DebateResult:
        start_time = time.time()
        debate_history: List[AgentPrediction] = []
        mtom_history: List[MutualToM] = []
        convergence_achieved = False
        round_num = 0

        # Stage 1: Initial predictions
        for agent in self.agents:
            pred = agent.generate_initial_prediction(
                question, code, round_num=0,
                temperature=self.config.temperature,
                max_tokens=self.config.max_new_tokens,
                top_p=self.config.top_p,
                do_sample=self.config.do_sample
            )
            debate_history.append(pred)

        # Stage 2: Debate rounds
        for round_num in range(1, self.config.max_rounds + 1):
            latest_preds = self._get_latest_predictions(debate_history)

            if self.config.enable_mtom:
                mtom = self._construct_mutual_tom(question, code, latest_preds, round_num)
                mtom_history.append(mtom)

            # update each agent
            for agent in self.agents:
                my_pred = next(p for p in latest_preds if p.agent_id == agent.agent_id)
                other_preds = [p for p in latest_preds if p.agent_id != agent.agent_id]

                if self.config.enable_mtom:
                    if not agent.tom_models:
                        agent.construct_tom_models(
                            question, code, my_pred, other_preds,
                            temperature=self.config.temperature,
                            max_tokens=self.config.max_new_tokens,
                            top_p=self.config.top_p,
                            do_sample=self.config.do_sample
                        )
                    updated = agent.update_with_mtom(
                        question, code, my_pred, other_preds, round_num,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_new_tokens,
                        top_p=self.config.top_p,
                        do_sample=self.config.do_sample
                    )
                else:
                    updated = agent.generate_initial_prediction(
                        question, code, round_num=round_num,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_new_tokens,
                        top_p=self.config.top_p,
                        do_sample=self.config.do_sample
                    )

                debate_history.append(updated)

            # decider
            if self.config.enable_decider:
                should_continue, conv_score = self._decider_check(question, code, debate_history, round_num)
                if (not should_continue) or (conv_score >= self.config.convergence_threshold):
                    convergence_achieved = True
                    break

        # Stage 3: Final synthesis (FIXED / COMPLETED)
        final_answer = self._synthesize_final_answer(question, code, debate_history, mtom_history)

        # evaluation
        correct = self._check_correctness(final_answer, ground_truth) if ground_truth is not None else False
        total_tokens = sum(p.tokens_used for p in debate_history)
        total_time = time.time() - start_time

        return DebateResult(
            question=question,
            final_answer=final_answer,
            ground_truth=ground_truth,
            correct=bool(correct),
            debate_history=debate_history,
            mtom_history=mtom_history,
            num_rounds=round_num,
            total_tokens=total_tokens,
            total_time=total_time,
            convergence_achieved=convergence_achieved,
            config=self.config,
            item_id=item_id,
            task=task
        )

    def _get_latest_predictions(self, history: List[AgentPrediction]) -> List[AgentPrediction]:
        latest = {}
        for pred in history:
            if pred.agent_id not in latest or pred.round_num > latest[pred.agent_id].round_num:
                latest[pred.agent_id] = pred
        return list(latest.values())

    def _construct_mutual_tom(self, question: str, code: str,
                             predictions: List[AgentPrediction],
                             round_num: int) -> MutualToM:
        tom_models = {}
        for agent in self.agents:
            my_pred = next(p for p in predictions if p.agent_id == agent.agent_id)
            other_preds = [p for p in predictions if p.agent_id != agent.agent_id]
            models = agent.construct_tom_models(
                question, code, my_pred, other_preds,
                temperature=self.config.temperature,
                max_tokens=self.config.max_new_tokens,
                top_p=self.config.top_p,
                do_sample=self.config.do_sample
            )
            tom_models[agent.agent_id] = models

        convergence_score = self._calculate_convergence(predictions)
        consensus_points, disagreement_points = self._identify_consensus(predictions)

        return MutualToM(
            round_num=round_num,
            tom_models=tom_models,
            convergence_score=convergence_score,
            disagreement_points=disagreement_points,
            consensus_points=consensus_points
        )

    def _calculate_convergence(self, predictions: List[AgentPrediction]) -> float:
        if len(predictions) < 2:
            return 1.0
        pred_texts = [p.prediction.lower().strip() for p in predictions]
        confidences = [p.confidence for p in predictions]
        unique_preds = len(set(pred_texts))
        text_convergence = 1.0 - (unique_preds - 1) / len(pred_texts)
        conf_std = float(np.std(confidences))
        conf_convergence = max(0.0, 1.0 - conf_std)
        return float((text_convergence + conf_convergence) / 2.0)

    def _identify_consensus(self, predictions: List[AgentPrediction]) -> Tuple[List[str], List[str]]:
        pred_texts = [p.prediction for p in predictions]
        counts = Counter([p.lower().strip() for p in pred_texts])
        consensus, disagreements = [], []
        if len(counts) == 1:
            consensus.append("All agents agree")
        else:
            most_common = counts.most_common(1)[0]
            if most_common[1] > len(predictions) / 2:
                consensus.append(f"Majority agrees on: {most_common[0]}")
            for pred, count in counts.items():
                if count == 1:
                    disagreements.append(f"Unique position: {pred}")
        return consensus, disagreements

    def _decider_check(self, question: str, code: str,
                       history: List[AgentPrediction], round_num: int) -> Tuple[bool, float]:
        prompt = MMADPrompts.decider_evaluation(question, code, history, round_num, self.config.max_rounds)
        # Use first agent model as decider (same as your original)
        result = self.client.generate(
            self.config.models[0], prompt,
            temperature=max(0.05, self.config.temperature * 0.6),
            max_tokens=256,
            top_p=self.config.top_p,
            do_sample=self.config.do_sample
        )
        response = result["response"]

        # Parse decision
        cont = True
        m = re.search(r"CONTINUE_DEBATE:\s*(YES|NO)", response, re.IGNORECASE)
        if m:
            cont = m.group(1).upper() == "YES"

        conv_score = 0.5
        m2 = re.search(r"CONVERGENCE_SCORE:\s*([0-1](?:\.\d+)?)", response, re.IGNORECASE)
        if m2:
            try:
                conv_score = float(m2.group(1))
            except Exception:
                conv_score = 0.5

        return (cont and round_num < self.config.max_rounds), float(max(0.0, min(1.0, conv_score)))

    def _synthesize_final_answer(self, question: str, code: str,
                                 history: List[AgentPrediction],
                                 mtom_history: List[MutualToM]) -> str:
        """Synthesize final answer from debate (COMPLETES YOUR TRUNCATED SECTION)."""
        prompt = MMADPrompts.final_synthesis(question, code, history, mtom_history)
        result = self.client.generate(
            self.config.models[0], prompt,
            temperature=max(0.05, self.config.temperature * 0.6),
            max_tokens=256,
            top_p=self.config.top_p,
            do_sample=self.config.do_sample
        )
        response = result["response"]

        # Extract FINAL_ANSWER line if present
        for line in response.splitlines():
            if line.strip().upper().startswith("FINAL_ANSWER:"):
                ans = line.split(":", 1)[1].strip()
                if ans:
                    return ans

        # Otherwise, fallback: use best last-round by confidence
        final_round = max(p.round_num for p in history)
        final_preds = [p for p in history if p.round_num == final_round]
        if final_preds:
            best = sorted(final_preds, key=lambda p: p.confidence, reverse=True)[0]
            return best.prediction.strip()

        # last resort: majority of all
        counts = Counter([p.prediction.strip().lower() for p in history if p.prediction.strip()])
        return counts.most_common(1)[0][0] if counts else ""

    def _check_correctness(self, final_answer: str, ground_truth: Any) -> bool:
        fa = str(final_answer).strip().lower()
        gt = str(ground_truth).strip().lower()
        return fa == gt


# ===============================================
# EXPERIMENT RUNNER FOR MATHQA + CRUXEVAL
# ===============================================

def load_json_any(path: str) -> List[dict]:
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        items = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=[t.value for t in Task])
    ap.add_argument("--data", required=True, help="MathQA: JSON/JSONL; CruxEval: JSONL recommended")
    ap.add_argument("--outdir", default="results_hf_mmad")
    ap.add_argument("--max-rounds", type=int, default=5)

   # model overrides (keep your framework structure: 3 agents)
    ap.add_argument("--models", nargs="+", default=None, help="3 HF model IDs (agent0 agent1 agent2)")

    # generation overrides
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--do-sample", action="store_true")
    ap.add_argument("--torch-dtype", default="bfloat16", choices=["bfloat16","float16","float32"])
    args = ap.parse_args()

    task = Task(args.task)
    items = load_json_any(args.data)

    cfg = DebateConfig(
        max_rounds=args.max_rounds,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        torch_dtype=args.torch_dtype,
    )
    if args.models:
        if len(args.models) != 3:
            raise SystemExit("--models must provide exactly 3 HF model IDs")
        cfg.models = args.models

    client = HFClient(device_map=cfg.device_map, torch_dtype=cfg.torch_dtype)
    framework = MMADFramework(cfg, client)

    outdir = Path(args.outdir)
    out_jsonl = outdir / f"{task.value}_results.jsonl"
    out_csv = outdir / f"{task.value}_summary.csv"

    results_rows = []
    summary_rows = []

    for idx, item in enumerate(items):
        item_id = str(item.get("id", f"{task.value}_{idx}"))
        q, code = build_prompt_for_task(task, item)

        # For task correctness, we store gold as a simple string:
        if task == Task.MATHQA:
            gt = (item.get("correct", "") or "").strip().lower()
        else:
            gt = str(item.get("output", ""))

        dr = framework.run_debate(q, code, ground_truth=gt, item_id=item_id, task=task.value)
        pred = extract_task_answer(task, dr.final_answer)
        correct = check_task_correct(task, pred, item)

        results_rows.append({
            "id": item_id,
            "task": task.value,
            "final_answer_raw": dr.final_answer,
            "prediction": pred,
            "gold": gt,
            "correct": correct,
            "num_rounds": dr.num_rounds,
            "total_tokens": dr.total_tokens,
            "total_time": dr.total_time,
            "converged": dr.convergence_achieved,
            "debate_history": [asdict(p) for p in dr.debate_history],
            "mtom_history": [
                {
                    "round_num": m.round_num,
                    "convergence_score": m.convergence_score,
                    "consensus_points": m.consensus_points,
                    "disagreement_points": m.disagreement_points,
                    "tom_models": {
                        str(aid): {str(oid): asdict(tom) for oid, tom in od.items()}
                        for aid, od in m.tom_models.items()
                    }
                } for m in dr.mtom_history
            ],
            "models": cfg.models
        })

        summary_rows.append({
            "id": item_id,
            "correct": correct,
            "prediction": pred,
            "gold": gt,
            "num_rounds": dr.num_rounds,
            "total_tokens": dr.total_tokens,
            "total_time": dr.total_time,
        })

        print(f"[{item_id}] pred={pred!r} correct={correct} rounds={dr.num_rounds} time={dr.total_time:.2f}s")

    save_jsonl(out_jsonl, results_rows)
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)

    # Aggregate
    valid = [r for r in summary_rows if r["correct"] is not None]
    acc = (sum(1 for r in valid if r["correct"]) / len(valid)) if valid else 0.0
    print(f"\nSaved: {out_jsonl}")
    print(f"Saved: {out_csv}")
    print(f"Accuracy (where gold exists): {acc:.3f} over {len(valid)} items")


if __name__ == "__main__":
    main()
