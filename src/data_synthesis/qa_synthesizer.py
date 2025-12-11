"""
QA Synthesizer

Responsible for synthesizing Q&A pairs based on trajectories
"""

import openai
import json
import os
from typing import Optional
from datetime import datetime

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
    
    def synthesize_qa(self, trajectory: Trajectory, qa_index: int = 0) -> Optional[SynthesizedQA]:
        """Synthesize Q&A pair based on trajectory"""
        print(f"\n🔧 Synthesizing QA pair - Trajectory: {trajectory.trajectory_id}")
        
        # Build trajectory description
        traj_description = self._format_trajectory(trajectory)
        
        # Generate Q&A pair
        prompt = self._build_qa_synthesis_prompt(trajectory, traj_description)
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # 生成QA的唯一标识：source_id + trajectory编号 + qa编号
            qa_id = f"{trajectory.trajectory_id}_qa_{qa_index}"
            
            qa = SynthesizedQA(
                question=result.get("question", ""),
                answer=result.get("answer", ""),
                trajectory_id=trajectory.trajectory_id,
                source_id=trajectory.source_id,
                qa_id=qa_id,
                reasoning_steps=result.get("reasoning_steps", []),
                metadata={
                    "seed_data": trajectory.seed_data,
                    "seed_description": self.config.seed_description,
                    "trajectory_depth": trajectory.total_depth,
                    "synthesis_date": datetime.now().isoformat(),
                    "environment_mode": self.config.environment_mode
                }
            )
            
            print(f"  ✓ Successfully synthesized QA pair")
            print(f"    QA ID: {qa_id}")
            print(f"    Question: {qa.question[:100]}...")
            print(f"    Answer: {qa.answer[:100]}...")
            
            return qa
            
        except Exception as e:
            print(f"  ✗ Synthesis failed: {str(e)}")
            return None
    
    def _build_qa_synthesis_prompt(self, trajectory: Trajectory, traj_description: str) -> str:
        """Build QA synthesis prompt (dynamically generated based on configuration)"""
        
        # Generic prompt template
        prompt = f"""You are a data synthesis expert. Based on the following Agent's exploration trajectory, synthesize a high-quality Q&A pair.

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
- **Multi-hop Factoid**: The question MUST require synthesizing information from multiple steps/documents to answer, BUT the target answer must be a specific fact (e.g., a name, a date, a location, a count, or a yes/no status).
- **Avoid Explanations**: **DO NOT** ask "How", "Why", or "Describe" questions that require long textual explanations. Instead of "How did X affect Y?", ask "What was the specific percentage increase in Y caused by X?".
- **De-contextualized**: The question should be understandable without seeing the trajectory (e.g., use "What represents the..." instead of "What did the agent find...").

## Answer Requirements (Crucial for Strict Length):
- **Extreme Brevity**: The answer MUST be **less than or equal to one sentence**, or ideally just a **short phrase** (e.g., "1985", "The Treaty of Versailles", "Increased by 5%").
- **No Fluff**: Do not use filler words like "According to the documents..." or "The answer is...". Provide ONLY the final answer value.
- **Groundedness**: The specific fact must be strictly derived from the provided trajectory observations.

## Reasoning Steps Requirements:
- Clearly show the logical deduction process (Chain-of-Thought) that links the multiple steps to the single final fact.
- **Observation Fidelity**: The 'observation' field in the JSON MUST contain direct excerpts.

Return JSON EXACTLY in this schema (do not add extra fields):
{{
  "question": "question text (multi-hop but targeting a specific fact)",
  "answer": "short phrase or single sentence",
  "reasoning_steps": [
    {{
      "step": 1,
      "description": "1 short sentence explaining how this step advances the answer",
      "intent": "copy or paraphrase the intent relevant to this step",
      "action": "the tool name used in this step (e.g., query_knowledge_base_dense)",
      "observation": "a short DIRECT excerpt from the step's observation that supports the answer"
    }},
    ...
  ]
}}
Rules for reasoning_steps:
- Use 2-6 steps actually needed for the answer (no generic summaries).
- Every step MUST include the tool name from the trajectory and a direct excerpt (<=200 chars) from the corresponding observation.
- Keep descriptions brief; avoid meta commentary.
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
