import json
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import time

@dataclass
class AgentResponse:
    """Represents an agent's response during debate"""
    agent_name: str
    prediction: str
    confidence: float
    reasoning: str
    round: int

@dataclass
class TheoryOfMind:
    """Theory of Mind about another agent"""
    target_agent: str
    strengths: List[str]
    weaknesses: List[str]
    reasoning_patterns: List[str]
    confidence_assessment: str

class OllamaAgent:
    """Individual agent powered by Ollama"""
    
    def __init__(self, name: str, model: str, ollama_url: str = "http://localhost:11434"):
        self.name = name
        self.model = model
        self.ollama_url = ollama_url
        self.history = []
        
    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate response from Ollama"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": temperature
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            print(f"Error calling Ollama for {self.name}: {e}")
            return f"Error: {str(e)}"
    
    def initial_response(self, question: str, context: str = "") -> AgentResponse:
        """Generate initial prediction with reasoning and confidence"""
        prompt = f"""You are solving a coding problem. Provide your answer in the following JSON format:
{{
    "prediction": "your answer here",
    "confidence": 0.0-1.0,
    "reasoning": "step by step explanation"
}}

Question: {question}
{f'Context: {context}' if context else ''}

Respond ONLY with valid JSON, no additional text."""

        response_text = self.generate(prompt)
        
        try:
            # Try to extract JSON from response
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start >= 0 and end > start:
                json_text = response_text[start:end]
                response_data = json.loads(json_text)
            else:
                response_data = json.loads(response_text)
                
            return AgentResponse(
                agent_name=self.name,
                prediction=response_data.get("prediction", ""),
                confidence=float(response_data.get("confidence", 0.5)),
                reasoning=response_data.get("reasoning", ""),
                round=0
            )
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return AgentResponse(
                agent_name=self.name,
                prediction=response_text[:200],
                confidence=0.5,
                reasoning="Failed to parse structured response",
                round=0
            )
    
    def construct_tom(self, other_responses: List[AgentResponse]) -> Dict[str, TheoryOfMind]:
        """Construct Theory of Mind for other agents"""
        toms = {}
        
        for response in other_responses:
            if response.agent_name == self.name:
                continue
                
            prompt = f"""Analyze this agent's reasoning and provide insights in JSON format:
{{
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"],
    "reasoning_patterns": ["pattern1", "pattern2"],
    "confidence_assessment": "assessment"
}}

Agent: {response.agent_name}
Prediction: {response.prediction}
Confidence: {response.confidence}
Reasoning: {response.reasoning}

Respond ONLY with valid JSON."""

            tom_text = self.generate(prompt, temperature=0.5)
            
            try:
                start = tom_text.find('{')
                end = tom_text.rfind('}') + 1
                if start >= 0 and end > start:
                    json_text = tom_text[start:end]
                    tom_data = json.loads(json_text)
                else:
                    tom_data = json.loads(tom_text)
                    
                toms[response.agent_name] = TheoryOfMind(
                    target_agent=response.agent_name,
                    strengths=tom_data.get("strengths", []),
                    weaknesses=tom_data.get("weaknesses", []),
                    reasoning_patterns=tom_data.get("reasoning_patterns", []),
                    confidence_assessment=tom_data.get("confidence_assessment", "")
                )
            except json.JSONDecodeError:
                toms[response.agent_name] = TheoryOfMind(
                    target_agent=response.agent_name,
                    strengths=["Unable to assess"],
                    weaknesses=["Unable to assess"],
                    reasoning_patterns=["Unable to assess"],
                    confidence_assessment="Unable to assess"
                )
        
        return toms
    
    def debate_response(self, 
                       question: str, 
                       context: str,
                       my_previous: AgentResponse,
                       other_responses: List[AgentResponse],
                       toms: Dict[str, TheoryOfMind],
                       round_num: int) -> AgentResponse:
        """Generate updated response based on debate and ToM"""
        
        # Format other agents' information
        others_info = "\n\n".join([
            f"Agent {r.agent_name}:\n"
            f"  Prediction: {r.prediction}\n"
            f"  Confidence: {r.confidence}\n"
            f"  Reasoning: {r.reasoning}"
            for r in other_responses if r.agent_name != self.name
        ])
        
        # Format ToM insights
        tom_info = "\n\n".join([
            f"Theory of Mind for {tom.target_agent}:\n"
            f"  Strengths: {', '.join(tom.strengths)}\n"
            f"  Weaknesses: {', '.join(tom.weaknesses)}\n"
            f"  Patterns: {', '.join(tom.reasoning_patterns)}"
            for tom in toms.values()
        ])
        
        prompt = f"""You are in round {round_num} of a collaborative debate. Review other agents' responses and your Theory of Mind analysis, then update your answer.

Question: {question}
{f'Context: {context}' if context else ''}

YOUR PREVIOUS RESPONSE:
Prediction: {my_previous.prediction}
Confidence: {my_previous.confidence}
Reasoning: {my_previous.reasoning}

OTHER AGENTS' RESPONSES:
{others_info}

YOUR THEORY OF MIND ANALYSIS:
{tom_info}

Based on this information:
1. Identify gaps in your reasoning
2. Consider superior approaches from other agents
3. Correct any errors you may have made
4. Update your prediction and confidence

Respond in JSON format:
{{
    "prediction": "your updated answer",
    "confidence": 0.0-1.0,
    "reasoning": "updated reasoning incorporating insights from debate"
}}

Respond ONLY with valid JSON."""

        response_text = self.generate(prompt, temperature=0.6)
        
        try:
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start >= 0 and end > start:
                json_text = response_text[start:end]
                response_data = json.loads(json_text)
            else:
                response_data = json.loads(response_text)
                
            return AgentResponse(
                agent_name=self.name,
                prediction=response_data.get("prediction", my_previous.prediction),
                confidence=float(response_data.get("confidence", my_previous.confidence)),
                reasoning=response_data.get("reasoning", my_previous.reasoning),
                round=round_num
            )
        except json.JSONDecodeError:
            # If parsing fails, keep previous response
            return AgentResponse(
                agent_name=self.name,
                prediction=my_previous.prediction,
                confidence=my_previous.confidence,
                reasoning=f"Round {round_num}: Maintained previous position",
                round=round_num
            )


class DeciderAgent:
    """Agent that decides when to terminate debate"""
    
    def __init__(self, model: str, ollama_url: str = "http://localhost:11434"):
        self.model = model
        self.ollama_url = ollama_url
    
    def generate(self, prompt: str) -> str:
        """Generate response from Ollama"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.3
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            print(f"Error in DeciderAgent: {e}")
            return "continue"
    
    def should_continue(self, responses: List[AgentResponse], max_rounds: int) -> bool:
        """Decide if debate should continue"""
        
        current_round = responses[0].round if responses else 0
        
        if current_round >= max_rounds:
            return False
        
        # Check for convergence
        predictions = [r.prediction for r in responses]
        confidences = [r.confidence for r in responses]
        
        # Simple convergence check: all predictions are the same
        if len(set(predictions)) == 1 and min(confidences) > 0.7:
            return False
        
        # Use LLM to make decision
        summary = "\n".join([
            f"Agent {r.agent_name}: {r.prediction} (confidence: {r.confidence:.2f})"
            for r in responses
        ])
        
        prompt = f"""Analyze if this debate should continue. Current round: {current_round}/{max_rounds}

Agent responses:
{summary}

Should the debate continue? Respond with ONLY 'continue' or 'stop'.
Continue if: predictions differ significantly, low confidence, or productive disagreement
Stop if: consensus reached, high confidence, or no progress being made"""

        decision = self.generate(prompt).strip().lower()
        
        return "continue" in decision


class MMADFramework:
    """Multi-Agent Mutual Awareness Debate Framework"""
    
    def __init__(self, 
                 agent_configs: List[Dict[str, str]],
                 decider_model: str = "qwen2.5:latest",
                 ollama_url: str = "http://localhost:11434",
                 max_rounds: int = 3):
        """
        Initialize MMAD framework
        
        Args:
            agent_configs: List of dicts with 'name' and 'model' keys
            decider_model: Model for the decider agent
            ollama_url: Ollama API URL
            max_rounds: Maximum debate rounds
        """
        self.agents = [
            OllamaAgent(config["name"], config["model"], ollama_url)
            for config in agent_configs
        ]
        self.decider = DeciderAgent(decider_model, ollama_url)
        self.max_rounds = max_rounds
        self.ollama_url = ollama_url
    
    def solve(self, question: str, context: str = "", verbose: bool = True) -> Dict[str, Any]:
        """
        Solve a problem using MMAD
        
        Returns:
            Dict containing final answer, all responses, and metadata
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"MMAD Problem Solving")
            print(f"{'='*60}")
            print(f"Question: {question}\n")
        
        # Phase 1: Initial responses
        if verbose:
            print("Phase 1: Generating initial responses...")
        
        current_responses = []
        for agent in self.agents:
            if verbose:
                print(f"  {agent.name} thinking...")
            response = agent.initial_response(question, context)
            current_responses.append(response)
            if verbose:
                print(f"    Prediction: {response.prediction[:100]}...")
                print(f"    Confidence: {response.confidence:.2f}\n")
        
        all_rounds = [current_responses.copy()]
        
        # Phase 2: Debate rounds with ToM
        round_num = 1
        while round_num <= self.max_rounds:
            if verbose:
                print(f"\nRound {round_num}: Mutual ToM Construction & Debate")
                print("-" * 60)
            
            # Check if should continue
            if not self.decider.should_continue(current_responses, self.max_rounds):
                if verbose:
                    print("Decider: Consensus reached, stopping debate.")
                break
            
            new_responses = []
            
            for agent in self.agents:
                if verbose:
                    print(f"\n  {agent.name}:")
                
                # Get agent's previous response
                my_previous = next(r for r in current_responses if r.agent_name == agent.name)
                
                # Construct Theory of Mind for other agents
                if verbose:
                    print("    Constructing Theory of Mind...")
                toms = agent.construct_tom(current_responses)
                
                # Generate updated response
                if verbose:
                    print("    Updating response based on debate & ToM...")
                new_response = agent.debate_response(
                    question, context, my_previous, 
                    current_responses, toms, round_num
                )
                new_responses.append(new_response)
                
                if verbose:
                    print(f"    Updated prediction: {new_response.prediction[:100]}...")
                    print(f"    Confidence: {new_response.confidence:.2f}")
            
            current_responses = new_responses
            all_rounds.append(current_responses.copy())
            round_num += 1
        
        # Phase 3: Aggregation
        if verbose:
            print(f"\n{'='*60}")
            print("Phase 3: Generating final consensus answer...")
        
        final_answer = self._aggregate_responses(current_responses, question, context)
        
        if verbose:
            print(f"\nFinal Answer: {final_answer}\n")
            print(f"{'='*60}\n")
        
        return {
            "final_answer": final_answer,
            "all_rounds": all_rounds,
            "num_rounds": round_num - 1,
            "final_responses": current_responses
        }
    
    def _aggregate_responses(self, responses: List[AgentResponse], 
                           question: str, context: str) -> str:
        """Aggregate final responses into consensus answer"""
        
        # Use majority voting with confidence weighting
        predictions = {}
        for r in responses:
            pred = r.prediction.strip()
            if pred not in predictions:
                predictions[pred] = 0
            predictions[pred] += r.confidence
        
        if predictions:
            # Return prediction with highest weighted confidence
            return max(predictions.items(), key=lambda x: x[1])[0]
        
        return "Unable to reach consensus"


# Example usage
if __name__ == "__main__":
    # Configure heterogeneous agents
    # Make sure these models are pulled in Ollama first!
    agent_configs = [
        {"name": "Qwen", "model": "qwen2.5:latest"},
        {"name": "Phi", "model": "phi3:latest"},
        {"name": "Gemma", "model": "gemma2:latest"}
    ]
    
    # Initialize framework
    mmad = MMADFramework(
        agent_configs=agent_configs,
        decider_model="qwen2.5:latest",
        max_rounds=3
    )
    
    # Example problem from test.jsonl
    code = """def f(nums):
    output = []
    for n in nums:
        output.append((nums.count(n), n))
    output.sort(reverse=True)
    return output"""
    
    question = f"What is the output of this function?\n\nCode:\n{code}\n\nInput: [1, 1, 3, 1, 3, 1]"
    
    # Solve with MMAD
    result = mmad.solve(question, verbose=True)
    
    print("\nResult Summary:")
    print(f"Rounds completed: {result['num_rounds']}")
    print(f"Final answer: {result['final_answer']}")