"""
QA Synthesizer

Responsible for synthesizing Q&A pairs based on trajectories
"""

import openai
import json
import os
import re
from typing import Optional
from datetime import datetime
import pdb

from models import Trajectory, SynthesizedQA
from synthesis_config import SynthesisConfig


class GenericQASynthesizer:
    """
    Generic QA synthesizer that generates Q&A pairs based on configuration and examples
    """
    
    def __init__(self, config: SynthesisConfig):
        """Initialize QA synthesizer"""
        self.config = config
        
        self.client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("OPENAI_API_URL", os.environ.get("OPENAI_API_BASE", ""))
        )

    def _normalize_text(self, text: str) -> str:
        """Normalize text for simple containment checks."""
        t = str(text or "").strip().lower()
        # Collapse whitespace
        t = re.sub(r"\s+", " ", t)
        return t

    def _answer_leaks_into_question(self, question: str, answer: str) -> bool:
        """
        Basic anti-shortcut guard:
        reject if answer appears verbatim inside question (case-insensitive).
        """
        q = self._normalize_text(question)
        a = self._normalize_text(answer)
        if not q or not a:
            return False
        return a in q

    def _question_too_verbose(self, question: str, max_words: int = 85, max_chars: int = 500) -> bool:
        """Heuristic verbosity filter to reduce trivia-dump questions."""
        q = str(question or "").strip()
        if len(q) > max_chars:
            return True
        # Word count heuristic (English-centric)
        words = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", q)
        return len(words) > max_words
    
    def synthesize_qa(self, trajectory: Trajectory, qa_index: int = 0) -> Optional[SynthesizedQA]:
        """Synthesize Q&A pair based on trajectory"""
        print(f"\n🔧 Synthesizing QA pair - Trajectory: {trajectory.trajectory_id}")
        
        # Build trajectory description
        traj_description = self._format_trajectory(trajectory)
        
        max_attempts = int(getattr(self.config, "qa_synthesis_max_attempts", 3) or 3)
        max_attempts = max(1, min(max_attempts, 8))
        last_failure_reason = ""

        for attempt in range(1, max_attempts + 1):
            # Generate Q&A pair
            prompt = self._build_qa_synthesis_prompt(trajectory, traj_description)
            if last_failure_reason:
                prompt += f"""

[Regeneration Required - Previous Output Rejected]
Reason: {last_failure_reason}
You MUST regenerate a NEW question/answer that fully satisfies the guidance. Do NOT repeat the previous wording.
"""

            try:
                response = self.client.chat.completions.create(
                    # model=self.config.model_name,
                    model='deepseek/deepseek-v3.2',
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7 + 0.1 * (attempt - 1),
                    response_format={"type": "json_object"}
                )
                result = json.loads(response.choices[0].message.content)

                question = str(result.get("question", "")).strip()
                answer = str(result.get("answer", "")).strip()
                reasoning_steps = result.get("reasoning_steps", [])
                if not isinstance(reasoning_steps, list):
                    reasoning_steps = []

                # --- Hard validations to prevent trivial / low-quality samples ---
                if not question or not answer:
                    last_failure_reason = "Empty question or answer."
                    continue

                if self._answer_leaks_into_question(question, answer):
                    last_failure_reason = "Answer leakage: the answer text appears verbatim inside the question."
                    continue

                if self._question_too_verbose(question):
                    last_failure_reason = "Question too verbose (likely a trivia dump). Keep <=2 sentences and compress clues."
                    continue

                print("--------------------------------")
                print(prompt)
                print()
                print(result)
                print("--------------------------------")

                # 生成QA的唯一标识：source_id + trajectory编号 + qa编号
                qa_id = f"{trajectory.trajectory_id}_qa_{qa_index}"

                qa = SynthesizedQA(
                    question=question,
                    answer=answer,
                    trajectory_id=trajectory.trajectory_id,
                    source_id=trajectory.source_id,
                    qa_id=qa_id,
                    reasoning_steps=reasoning_steps,
                    metadata={
                        "seed_data": trajectory.seed_data,
                        "seed_description": self.config.seed_description,
                        "trajectory_depth": trajectory.total_depth,
                        "synthesis_date": datetime.now().isoformat(),
                        "environment_mode": self.config.environment_mode,
                        "qa_synthesis_attempts": attempt
                    }
                )

                print(f"  ✓ Successfully synthesized QA pair")
                print(f"    QA ID: {qa_id}")
                print(f"    Question: {qa.question}...")
                print(f"    Answer: {qa.answer}...")

                return qa

            except Exception as e:
                last_failure_reason = f"Synthesis exception: {str(e)}"
                continue

        print(f"  ✗ Synthesis failed after {max_attempts} attempts. Last reason: {last_failure_reason}")
        return None
    
    def _build_qa_synthesis_prompt(self, trajectory: Trajectory, traj_description: str) -> str:
        """Build QA synthesis prompt (dynamically generated based on configuration)"""
        
        # Generic prompt template
        prompt = f"""You are a data synthesis expert. Based on the following Agent's exploration trajectory, synthesize a high-quality Q&A pair. You should follow the Data Synthesis Guidance as much as you can.

【Starting Point Information】
Content: {trajectory.seed_data}"""
        
        if self.config.seed_description:
            prompt += f"\nDescription: {self.config.seed_description}"
        
        prompt += f"""

【Complete Exploration Trajectory】
{traj_description}

"""
        
        # Add synthesis tips
        if self.config.synthesis_tips:
            prompt += f"""Data Synthesis Guidance:\n{self.config.synthesis_tips}\n\n"""
        
        # Add QA examples
        if self.config.qa_examples:
            prompt += """Refer to the style and quality of the following examples:\n\n"""
            for i, example in enumerate(self.config.qa_examples, 1):
                prompt += f"""Example {i}:
Question: {example.get('question', '')}
Answer: {example.get('answer', '')}
"""
                if 'reasoning' in example:
                    prompt += f"Reasoning Process: {example['reasoning']}\n"
                prompt += "\n"
        
        prompt += f"""
Please synthesize a high-quality Q&A pair based on the trajectory:

## Question Requirements (Crucial for Reasoning & Brevity):
- The target answer must be a specific fact (e.g., a name, a date, a location, a count, or a yes/no status).
- **DO NOT** ask "How", "Why", or "Describe" questions that require long textual explanations. 
- The question should be understandable without seeing the trajectory and observation as a natural, factual, and self-contained question (e.g., don't include "What did the agent find...", "what is in the trajectory...", "according to the trajectory...", ...).
- **Anti-shortcut**: The question MUST NOT contain the answer text, and MUST NOT directly state the asked attribute in a definitional clause.
- **Low-entrance, deep-reasoning**: Keep the question to <=2 sentences and a small number of top-level clues; depth should come from multi-hop reasoning, not a long list of trivia.

## Answer Requirements (Crucial for Strict Length):
- **Extreme Brevity**: The answer MUST be **less than or equal to one sentence, and contain only one entity**, or ideally just a **short phrase** (e.g., "1985", "The Treaty of Versailles", "Increased by 5%").
- **No Fluff**: Do not use filler words like "According to the documents..." or "The answer is...". Provide ONLY the final answer value.
- **Groundedness**: The specific fact must be strictly derived from the provided trajectory observations without mentioning the trajectory or observation.

## Task Solving Reasoning Steps Requirements:
- Clearly show the logical deduction process (Chain-of-Thought) that links the multiple steps to the single final fact.
- Note that this is the task solving process not the steps from the trajectory. 

Return JSON EXACTLY in this schema (do not add extra fields):
{{
  "question": "question text",
  "answer": "short phrase or single sentence",
  "reasoning_steps": [
    {{
      "step": 1,
      "action": "the tool name used in this step (e.g., query_knowledge_base_dense)",
    }},
    ...
  ]
}}
"""
        
        return prompt
    
    def _format_trajectory(self, trajectory: Trajectory) -> str:
        """Format trajectory into readable text"""
        formatted = ""
        
        for i, node in enumerate(trajectory.nodes, 1):
            formatted += f"\nStep {i}:\n"
            formatted += f"  Intent: {node.intent}\n"
            
            if node.action:
                formatted += f"  Action: {node.action.get('tool_name', 'unknown')}\n"
                formatted += f"  Parameters: {json.dumps(node.action.get('parameters', {}), ensure_ascii=False)}\n"
            
            obs_preview = node.observation[:500] + "..." if len(node.observation) > 500 else node.observation
            formatted += f"  Observation: {obs_preview}\n"
        
        return formatted
