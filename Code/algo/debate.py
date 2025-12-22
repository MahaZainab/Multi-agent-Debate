#!/usr/bin/env python3
"""
Multi-Agent Mutual Awareness Debate (MMAD) Framework for Code Reasoning
Implementation using Ollama for local LLM inference
"""

import json
import ollama
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import csv


# =============================
# 1. Data structures
# =============================

@dataclass
class CodeExample:
    """Represents a code reasoning example"""
    code: str
    input: str
    output: str
    id: str


@dataclass
class AgentResponse:
    """Agent's response with prediction, confidence, and reasoning"""
    agent_name: str
    model_name: str
    round_id: int
    prediction: str
    confidence: float
    reasoning: str
    raw_output: str


@dataclass
class TheoryOfMind:
    """Theory of Mind analysis about another agent"""
    target_agent: str
    strengths: List[str]
    weaknesses: List[str]
    reasoning_patterns: List[str]
    confidence_assessment: str


@dataclass
class MMADResult:
    """Complete result from MMAD framework"""
    example_id: str
    question: str
    gold_output: str
    baseline_predictions: List[str]  # Individual agent predictions (no debate)
    round1_responses: List[AgentResponse]
    round2_responses: List[AgentResponse]
    final_prediction: str
    num_rounds: int
    converged: bool


# =============================
# 2. Dataset loader
# =============================

def load_jsonl_dataset(path: str, limit: Optional[int] = None) -> List[CodeExample]:
    """Load code examples from JSONL file"""
    examples: List[CodeExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ex = CodeExample(
                code=obj["code"],
                input=obj["input"],
                output=obj["output"],
                id=obj["id"]
            )
            examples.append(ex)
            if limit and len(examples) >= limit:
                break
    return examples


# =============================
# 3. Agent implementation
# =============================

class OllamaAgent:
    """Agent powered by Ollama for code reasoning with ToM capabilities"""
    
    def __init__(self, model_name: str, agent_id: int, max_tokens: int = 512):
        self.model_name = model_name
        self.agent_name = f"Agent-{agent_id}({model_name})"
        self.max_tokens = max_tokens
    
    # ----- Utility helpers -----
    
    @staticmethod
    def _safe_float(x, default: float = 0.5) -> float:
        """Safely convert to float with bounds checking"""
        try:
            v = float(x)
            return max(0.0, min(1.0, v))
        except Exception:
            return default
    
    @staticmethod
    def _extract_json_block(text: str) -> str:
        """Extract JSON block from model output"""
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return text
        return text[start: end + 1]
    
    def _call_ollama(self, prompt: str, temperature: float = 0.7) -> str:
        """Call Ollama API"""
        res = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            options={
                "num_predict": self.max_tokens,
                "temperature": temperature
            },
            stream=False,
        )
        return res["response"]
    
    def _parse_response(self, raw_output: str) -> Tuple[str, float, str]:
        """Parse JSON response from model"""
        json_block = self._extract_json_block(raw_output)
        prediction = ""
        confidence = 0.5
        reasoning = json_block.strip()
        
        try:
            obj = json.loads(json_block)
            if isinstance(obj, dict):
                prediction = str(obj.get("prediction", ""))
                confidence = self._safe_float(obj.get("confidence", 0.5))
                reasoning = str(obj.get("reasoning", reasoning))
        except Exception:
            # Fallback: use raw output as prediction
            prediction = raw_output[:200]
        
        return prediction, confidence, reasoning
    
    # ----- Prompt builders -----
    
    @staticmethod
    def _round1_prompt(example: CodeExample) -> str:
        """Prompt for initial independent reasoning"""
        return f"""You are a code reasoning expert. Analyze the following Python function and predict its output.

CODE:
```python
{example.code}
```

INPUT: {example.input}

Task: Predict the output of this function when called with the given input.

Provide your answer in strict JSON format:
{{
    "prediction": "your predicted output",
    "confidence": 0.0-1.0,
    "reasoning": "step-by-step explanation of how you arrived at this answer"
}}

Think through the code execution carefully and respond ONLY with valid JSON."""
    
    @staticmethod
    def _round2_prompt(
        example: CodeExample,
        prev_self: AgentResponse,
        other_responses: List[AgentResponse],
        toms: Dict[str, TheoryOfMind]
    ) -> str:
        """Prompt for debate round with Theory of Mind"""
        
        # Format other agents' responses
        others_info = "\n\n".join([
            f"Agent: {r.agent_name}\n"
            f"  Prediction: {r.prediction}\n"
            f"  Confidence: {r.confidence:.2f}\n"
            f"  Reasoning: {r.reasoning[:300]}"
            for r in other_responses
        ])
        
        # Format ToM insights
        tom_info = "\n\n".join([
            f"Theory of Mind for {tom.target_agent}:\n"
            f"  Strengths: {', '.join(tom.strengths[:3])}\n"
            f"  Weaknesses: {', '.join(tom.weaknesses[:3])}\n"
            f"  Reasoning Patterns: {', '.join(tom.reasoning_patterns[:3])}\n"
            f"  Assessment: {tom.confidence_assessment[:200]}"
            for tom in toms.values()
        ])
        
        return f"""You are participating in a multi-agent collaborative reasoning debate.

CODE:
```python
{example.code}
```

INPUT: {example.input}

YOUR PREVIOUS RESPONSE (Round 1):
  Prediction: {prev_self.prediction}
  Confidence: {prev_self.confidence:.2f}
  Reasoning: {prev_self.reasoning[:300]}

OTHER AGENTS' RESPONSES:
{others_info}

YOUR THEORY OF MIND ANALYSIS:
{tom_info}

Based on this collaborative debate:
1. Identify any errors in your reasoning
2. Consider alternative approaches from other agents
3. Evaluate which reasoning is most sound
4. Update your prediction if necessary

Guidelines:
- Only change your answer if you find a clear error in your reasoning
- Be explicit about what convinced you to change (or not change)
- Higher confidence from others doesn't mean they're correct
- Focus on logical correctness, not consensus

Provide your UPDATED answer in strict JSON format:
{{
    "prediction": "your final predicted output",
    "confidence": 0.0-1.0,
    "reasoning": "explain your final reasoning, incorporating insights from debate"
}}

Respond ONLY with valid JSON."""
    
    @staticmethod
    def _tom_prompt(target_response: AgentResponse) -> str:
        """Prompt to construct Theory of Mind about another agent"""
        return f"""Analyze the reasoning approach of another agent solving a code problem.

Agent: {target_response.agent_name}
Prediction: {target_response.prediction}
Confidence: {target_response.confidence:.2f}
Reasoning: {target_response.reasoning}

Provide analysis in JSON format:
{{
    "strengths": ["strength1", "strength2", "strength3"],
    "weaknesses": ["weakness1", "weakness2", "weakness3"],
    "reasoning_patterns": ["pattern1", "pattern2", "pattern3"],
    "confidence_assessment": "brief assessment of their confidence level"
}}

Focus on:
- What reasoning strategies they used (e.g., line-by-line execution, pattern matching)
- Potential gaps or oversights in their analysis
- How reliable their approach seems

Respond ONLY with valid JSON."""
    
    # ----- Public API -----
    
    def initial_response(self, example: CodeExample) -> AgentResponse:
        """Generate initial independent response"""
        prompt = self._round1_prompt(example)
        raw = self._call_ollama(prompt, temperature=0.7)
        prediction, confidence, reasoning = self._parse_response(raw)
        
        return AgentResponse(
            agent_name=self.agent_name,
            model_name=self.model_name,
            round_id=1,
            prediction=prediction,
            confidence=confidence,
            reasoning=reasoning,
            raw_output=raw
        )
    
    def construct_tom(self, target_response: AgentResponse) -> TheoryOfMind:
        """Construct Theory of Mind about another agent"""
        prompt = self._tom_prompt(target_response)
        raw = self._call_ollama(prompt, temperature=0.5)
        json_block = self._extract_json_block(raw)
        
        try:
            obj = json.loads(json_block)
            return TheoryOfMind(
                target_agent=target_response.agent_name,
                strengths=obj.get("strengths", [])[:3],
                weaknesses=obj.get("weaknesses", [])[:3],
                reasoning_patterns=obj.get("reasoning_patterns", [])[:3],
                confidence_assessment=obj.get("confidence_assessment", "")
            )
        except Exception:
            return TheoryOfMind(
                target_agent=target_response.agent_name,
                strengths=["Unable to assess"],
                weaknesses=["Unable to assess"],
                reasoning_patterns=["Unable to assess"],
                confidence_assessment="Unable to assess"
            )
    
    def debate_response(
        self,
        example: CodeExample,
        prev_self: AgentResponse,
        other_responses: List[AgentResponse],
        toms: Dict[str, TheoryOfMind],
        round_num: int
    ) -> AgentResponse:
        """Generate updated response based on debate"""
        prompt = self._round2_prompt(example, prev_self, other_responses, toms)
        raw = self._call_ollama(prompt, temperature=0.6)
        prediction, confidence, reasoning = self._parse_response(raw)
        
        return AgentResponse(
            agent_name=self.agent_name,
            model_name=self.model_name,
            round_id=round_num,
            prediction=prediction,
            confidence=confidence,
            reasoning=reasoning,
            raw_output=raw
        )


# =============================
# 4. Decider agent
# =============================

class DeciderAgent:
    """Decides when to terminate debate"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def should_continue(
        self,
        responses: List[AgentResponse],
        current_round: int,
        max_rounds: int
    ) -> bool:
        """Determine if debate should continue"""
        
        if current_round >= max_rounds:
            return False
        
        # Check for strong convergence
        predictions = [r.prediction for r in responses]
        confidences = [r.confidence for r in responses]
        
        # All same prediction with high confidence
        if len(set(predictions)) == 1 and min(confidences) > 0.75:
            return False
        
        # Use LLM to make nuanced decision
        summary = "\n".join([
            f"{r.agent_name}: {r.prediction} (conf: {r.confidence:.2f})"
            for r in responses
        ])
        
        prompt = f"""Analyze if this collaborative debate should continue.

Current round: {current_round}/{max_rounds}

Agent predictions:
{summary}

Should debate continue? 

Continue if:
- Predictions differ significantly
- Agents have low confidence
- There's potential for improvement through discussion

Stop if:
- Strong consensus reached (same prediction, high confidence)
- Predictions stable across rounds
- Unlikely to improve further

Respond with ONLY 'continue' or 'stop'."""
        
        try:
            res = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={"num_predict": 10, "temperature": 0.3},
                stream=False,
            )
            decision = res["response"].strip().lower()
            return "continue" in decision
        except Exception:
            # Default: continue if not at max rounds
            return current_round < max_rounds


# =============================
# 5. MMAD Framework
# =============================

class MMADFramework:
    """Multi-Agent Mutual Awareness Debate Framework"""
    
    def __init__(
        self,
        agent_configs: List[Dict[str, str]],
        decider_model: str = "qwen2.5:latest",
        max_rounds: int = 3
    ):
        """
        Initialize MMAD framework
        
        Args:
            agent_configs: List of dicts with 'name' and 'model' keys
            decider_model: Model for deciding when to stop
            max_rounds: Maximum number of debate rounds
        """
        self.agents = [
            OllamaAgent(config["model"], idx + 1, max_tokens=512)
            for idx, config in enumerate(agent_configs)
        ]
        self.decider = DeciderAgent(decider_model)
        self.max_rounds = max_rounds
    
    def solve(self, example: CodeExample, verbose: bool = True) -> MMADResult:
        """Solve a code reasoning problem using MMAD"""
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"MMAD: Solving {example.id}")
            print(f"{'='*80}")
            print(f"Code: {example.code[:100]}...")
            print(f"Input: {example.input}")
            print(f"Gold Output: {example.output}\n")
        
        # === Round 1: Independent responses ===
        if verbose:
            print("ROUND 1: Independent Reasoning")
            print("-" * 80)
        
        round1_responses = []
        for agent in self.agents:
            if verbose:
                print(f"  {agent.agent_name} thinking...")
            response = agent.initial_response(example)
            round1_responses.append(response)
            if verbose:
                print(f"    Prediction: {response.prediction}")
                print(f"    Confidence: {response.confidence:.2f}\n")
        
        # Store baseline predictions (no debate)
        baseline_preds = [r.prediction for r in round1_responses]
        
        # === Debate rounds ===
        current_responses = round1_responses
        all_rounds = [round1_responses]
        round_num = 2
        
        while round_num <= self.max_rounds + 1:
            # Check convergence
            if not self.decider.should_continue(current_responses, round_num - 1, self.max_rounds):
                if verbose:
                    print(f"\nDecider: Stopping at round {round_num - 1} (consensus reached)")
                break
            
            if verbose:
                print(f"\nROUND {round_num}: Debate with Theory of Mind")
                print("-" * 80)
            
            new_responses = []
            for agent in self.agents:
                # Get agent's previous response
                prev = next(r for r in current_responses if r.agent_name == agent.agent_name)
                others = [r for r in current_responses if r.agent_name != agent.agent_name]
                
                # Construct ToM for other agents
                if verbose:
                    print(f"\n  {agent.agent_name}:")
                    print(f"    Building Theory of Mind...")
                
                toms = {}
                for other in others:
                    tom = agent.construct_tom(other)
                    toms[other.agent_name] = tom
                
                # Generate updated response
                if verbose:
                    print(f"    Updating prediction...")
                new_resp = agent.debate_response(example, prev, others, toms, round_num)
                new_responses.append(new_resp)
                
                if verbose:
                    print(f"    New Prediction: {new_resp.prediction}")
                    print(f"    Confidence: {new_resp.confidence:.2f}")
            
            current_responses = new_responses
            all_rounds.append(new_responses)
            round_num += 1
        
        # === Aggregation ===
        final_prediction = self._aggregate(current_responses)
        converged = len(set(r.prediction for r in current_responses)) == 1
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"FINAL RESULT")
            print(f"{'='*80}")
            print(f"Final Prediction: {final_prediction}")
            print(f"Gold Output: {example.output}")
            print(f"Correct: {final_prediction == example.output}")
            print(f"Rounds: {round_num - 1}")
            print(f"Converged: {converged}\n")
        
        return MMADResult(
            example_id=example.id,
            question=f"Code: {example.code}, Input: {example.input}",
            gold_output=example.output,
            baseline_predictions=baseline_preds,
            round1_responses=round1_responses,
            round2_responses=current_responses,
            final_prediction=final_prediction,
            num_rounds=round_num - 1,
            converged=converged
        )
    
    def _aggregate(self, responses: List[AgentResponse]) -> str:
        """Aggregate responses using confidence-weighted voting"""
        vote_scores: Dict[str, float] = {}
        
        for r in responses:
            pred = r.prediction.strip()
            if pred not in vote_scores:
                vote_scores[pred] = 0.0
            vote_scores[pred] += r.confidence
        
        if vote_scores:
            return max(vote_scores.items(), key=lambda x: x[1])[0]
        return ""


# =============================
# 6. Evaluation & Saving
# =============================

def compute_accuracy(results: List[MMADResult]) -> Tuple[float, float, float]:
    """Compute accuracy metrics"""
    if not results:
        return 0.0, 0.0, 0.0
    
    total = len(results)
    
    # Baseline: majority vote from Round 1 individual predictions
    baseline_correct = 0
    for res in results:
        # Simple majority from baseline predictions
        votes = {}
        for pred in res.baseline_predictions:
            votes[pred] = votes.get(pred, 0) + 1
        baseline_pred = max(votes.items(), key=lambda x: x[1])[0]
        if baseline_pred == res.gold_output:
            baseline_correct += 1
    
    # MMAD (after debate)
    mmad_correct = sum(1 for r in results if r.final_prediction == r.gold_output)
    
    # Best single agent (oracle)
    best_single = 0
    for res in results:
        if any(pred == res.gold_output for pred in res.baseline_predictions):
            best_single += 1
    
    return baseline_correct / total, mmad_correct / total, best_single / total


def save_results_csv(path: str, results: List[MMADResult]):
    """Save results to CSV"""
    fieldnames = [
        "id",
        "gold_output",
        "baseline_pred",
        "mmad_pred",
        "baseline_correct",
        "mmad_correct",
        "num_rounds",
        "converged"
    ]
    
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for res in results:
            # Baseline majority vote
            votes = {}
            for pred in res.baseline_predictions:
                votes[pred] = votes.get(pred, 0) + 1
            baseline_pred = max(votes.items(), key=lambda x: x[1])[0]
            
            writer.writerow({
                "id": res.example_id,
                "gold_output": res.gold_output,
                "baseline_pred": baseline_pred,
                "mmad_pred": res.final_prediction,
                "baseline_correct": baseline_pred == res.gold_output,
                "mmad_correct": res.final_prediction == res.gold_output,
                "num_rounds": res.num_rounds,
                "converged": res.converged
            })


def save_results_jsonl(path: str, results: List[MMADResult]):
    """Save detailed results to JSONL"""
    with open(path, "w", encoding="utf-8") as f:
        for res in results:
            obj = {
                "id": res.example_id,
                "question": res.question,
                "gold_output": res.gold_output,
                "baseline_predictions": res.baseline_predictions,
                "final_prediction": res.final_prediction,
                "num_rounds": res.num_rounds,
                "converged": res.converged,
                "round1": [
                    {
                        "agent": r.agent_name,
                        "prediction": r.prediction,
                        "confidence": r.confidence,
                        "reasoning": r.reasoning[:500]
                    }
                    for r in res.round1_responses
                ],
                "round2": [
                    {
                        "agent": r.agent_name,
                        "prediction": r.prediction,
                        "confidence": r.confidence,
                        "reasoning": r.reasoning[:500]
                    }
                    for r in res.round2_responses
                ]
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# =============================
# 7. Main
# =============================

def main():
    DATA_PATH = "test.jsonl"
    N_EXAMPLES = 10  # Start with 50 examples
    
    # Use small models that work well with Ollama
    # Make sure these are pulled: ollama pull <model>
    AGENT_CONFIGS = [
        {"name": "Qwen", "model": "qwen2.5:1.5b"},
        {"name": "Phi", "model": "phi3:latest"},
        {"name": "Gemma", "model": "gemma2:2b"}
    ]
    
    print("=" * 80)
    print("MMAD Framework for Code Reasoning")
    print("=" * 80)
    print(f"Loading dataset from {DATA_PATH}")
    
    dataset = load_jsonl_dataset(DATA_PATH, limit=N_EXAMPLES)
    print(f"Loaded {len(dataset)} examples\n")
    
    print("Agent configurations:")
    for config in AGENT_CONFIGS:
        print(f"  - {config['name']}: {config['model']}")
    print()
    
    # Initialize framework
    mmad = MMADFramework(
        agent_configs=AGENT_CONFIGS,
        decider_model="qwen2.5:1.5b",
        max_rounds=5
    )
    
    # Run MMAD on all examples
    results = []
    for idx, example in enumerate(dataset, start=1):
        print(f"\n{'#'*80}")
        print(f"Processing example {idx}/{len(dataset)}")
        print(f"{'#'*80}")
        
        result = mmad.solve(example, verbose=True)
        results.append(result)
    
    # Compute metrics
    baseline_acc, mmad_acc, oracle_acc = compute_accuracy(results)
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Examples evaluated: {len(results)}")
    print(f"Baseline (majority vote, no debate): {baseline_acc:.3f}")
    print(f"MMAD (with debate & ToM):            {mmad_acc:.3f}")
    print(f"Oracle (best single agent):          {oracle_acc:.3f}")
    print(f"Improvement: {(mmad_acc - baseline_acc):.3f}")
    print("=" * 80)
    
    # Save results
    csv_path = "mmad_results.csv"
    save_results_csv(csv_path, results)
    print(f"\nSaved summary to {csv_path}")
    
    jsonl_path = "mmad_results.jsonl"
    save_results_jsonl(jsonl_path, results)
    print(f"Saved detailed results to {jsonl_path}")


if __name__ == "__main__":
    main()