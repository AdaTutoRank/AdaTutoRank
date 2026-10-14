"""
Basic RAG generator on vLLM: passages + question -> short answer.

The prompt template and sampling settings come from configs/generator.yaml.
"""

import logging
from typing import Dict, List, Sequence

from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class Generator:
    """
    Args:
        model_path: Local model directory.
        prompt: Template with {contexts} and {question}.
        **llm_kwargs: Passed to ``vllm.LLM`` (dtype, max_model_len, gpu_memory_utilization, ...).
    """

    def __init__(self, model_path: str, prompt: str, **llm_kwargs):
        from vllm import LLM

        self.model_path = model_path
        self.prompt = prompt
        self.llm = LLM(model=model_path, **llm_kwargs)

    def build_prompt(self, question: str, contexts: Sequence[str]) -> str:
        return self.prompt.format(question=question, contexts="\n".join(contexts or []))

    def _generate_one(self, prompt: str, sampling_params) -> str:
        try:
            response = self.llm.generate(prompt, sampling_params, use_tqdm=False)
        except ValueError as e:
            if "longer than the maximum model length" not in str(e):
                raise
            # Keep the end of the prompt, where the question is.
            max_len = self.llm.llm_engine.model_config.max_model_len
            logger.warning(f"Prompt too long, keeping the last {max_len} tokens")
            tokenizer = self.llm.get_tokenizer()
            prompt = tokenizer.decode(tokenizer.encode(prompt)[-max_len:])
            response = self.llm.generate(prompt, sampling_params, use_tqdm=False)
        if response and response[0].outputs:
            return response[0].outputs[0].text.strip()
        return ""

    def generate(self, questions: Sequence[str], contexts: Sequence[Sequence[str]],
                 sampling: Dict) -> List[str]:
        from vllm import SamplingParams

        sampling_params = SamplingParams(**sampling)
        return [
            self._generate_one(self.build_prompt(q, ctxs), sampling_params)
            for q, ctxs in tqdm(list(zip(questions, contexts)), desc="Answering questions", unit="q")
        ]
