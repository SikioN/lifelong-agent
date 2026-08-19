"""Frozen-backbone policy: closed-set action-label scoring via causal-LM
log-likelihood. No generation, no fine-tuning on the critical path (LoRA is
optional/bonus — see docs/materials/PLAN.md). Scoring candidates instead of
generating text is what makes the Stage 4/5 grid computationally feasible on
a MacBook without CUDA (batched forward passes, verified in Task 3).
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ClosedSetPolicy:
    """Frozen causal LM wrapped for closed-set (multiple-choice) scoring."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str | None = None):
        self.device = device or _select_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        self.model.to(self.device)
        self.model.eval()

    def _candidate_token_ids(self, prompt: str, candidate: str) -> list[int]:
        """Token ids representing `candidate` as a continuation of `prompt`,
        found by diffing tokenize(prompt) against tokenize(prompt + " " +
        candidate) -- robust to BPE merge behavior at the prompt/candidate
        seam, since we only trust the *difference*, not an assumed token
        count for `candidate` alone."""
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(prompt + " " + candidate, add_special_tokens=False)["input_ids"]
        assert full_ids[: len(prompt_ids)] == prompt_ids, (
            "tokenization of prompt is not a prefix of prompt+candidate; "
            "cannot isolate the candidate's token span"
        )
        return full_ids[len(prompt_ids):]

    @torch.no_grad()
    def score_candidates(self, prompt: str, candidates: list[str]) -> np.ndarray:
        """Sum log-prob of each candidate's tokens as a continuation of
        prompt. Unbatched reference implementation -- one forward pass per
        candidate; Task 3 adds a batched version for grid-scale throughput."""
        scores = np.zeros(len(candidates), dtype=np.float64)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        for idx, candidate in enumerate(candidates):
            cand_ids = self._candidate_token_ids(prompt, candidate)
            input_ids = torch.tensor([prompt_ids + cand_ids], device=self.device)
            logits = self.model(input_ids).logits[0]  # (seq_len, vocab)
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            n_prompt = len(prompt_ids)
            total = 0.0
            for offset, token_id in enumerate(cand_ids):
                position = n_prompt + offset - 1  # logits[position] predicts token at position+1
                total += log_probs[position, token_id].item()
            scores[idx] = total
        return scores

    def predict(self, prompt: str, candidates: list[str]) -> str:
        scores = self.score_candidates(prompt, candidates)
        return candidates[int(np.argmax(scores))]
