"""Frozen-backbone policy: closed-set action-label scoring via causal-LM
log-likelihood. No generation, no fine-tuning on the critical path (LoRA is
optional/bonus — see docs/materials/PLAN.md). Scoring candidates instead of
generating text is what makes the Stage 4/5 grid computationally feasible on
a MacBook without CUDA (batched forward passes, verified in Task 3).
"""
from __future__ import annotations

import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")


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
            
        # Use bf16/fp16 and auto device mapping for CUDA to support large models on Kaggle.
        if self.device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            device_map = "cuda:0"
        else:
            dtype = torch.float32
            device_map = None
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=dtype,
            device_map=device_map
        )
        if device_map is None:
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

    @torch.no_grad()
    def score_candidates_batch(self, prompts: list[str], candidates: list[str], chunk_size: int = 16) -> np.ndarray:
        """Same scoring as score_candidates, but for many prompts against the
        SAME fixed candidate set, in one padded batched forward pass. This is
        the throughput-critical path for the Stage 4/5 grid: right-padding +
        causal attention means padded positions never influence any real
        token's prediction, so per-row prompt/candidate lengths (tracked
        exactly, pre-padding) are all that's needed to index correctly."""
        pad_id = self.tokenizer.pad_token_id
        sequences: list[list[int]] = []
        prompt_lens: list[int] = []
        cand_lens: list[int] = []
        for prompt in prompts:
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            for candidate in candidates:
                cand_ids = self._candidate_token_ids(prompt, candidate)
                sequences.append(prompt_ids + cand_ids)
                prompt_lens.append(len(prompt_ids))
                cand_lens.append(len(cand_ids))

        max_len = max(len(seq) for seq in sequences)
        batch = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
        for row, seq in enumerate(sequences):
            batch[row, : len(seq)] = torch.tensor(seq)
            attention_mask[row, : len(seq)] = 1
        flat_scores = np.zeros(len(sequences), dtype=np.float64)
        
        for i in range(0, len(sequences), chunk_size):
            b = batch[i : i + chunk_size].to(self.device)
            m = attention_mask[i : i + chunk_size].to(self.device)
            
            logits = self.model(input_ids=b, attention_mask=m).logits
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            
            for chunk_row in range(b.size(0)):
                row = i + chunk_row
                n_prompt = prompt_lens[row]
                n_cand = cand_lens[row]
                total = 0.0
                for offset in range(n_cand):
                    position = n_prompt + offset - 1
                    token_id = batch[row, n_prompt + offset].item()
                    total += log_probs[chunk_row, position, token_id].item()
                flat_scores[row] = total
                
            if self.device == "mps":
                torch.mps.empty_cache()

        return flat_scores.reshape(len(prompts), len(candidates))
