"""Frozen-backbone policy: closed-set action-label scoring via causal-LM
log-likelihood. No generation, no fine-tuning on the critical path (LoRA is
optional/bonus — see docs/materials/PLAN.md). Scoring candidates (batched
forward passes, verified in Task 3) is cheaper than generating text, but
that alone was NOT enough to make the Stage 4/5 grid feasible on a local
MacBook without CUDA — real throughput calibration showed local compute was
insufficient, which is why the grid runs on a cloud GPU instead (Kaggle,
then Google Colab; see docs/materials/PLAN.md's Архитектурная правка 3).
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Chosen from the Stage 2 backbone sweep (near-ceiling, calibrated chance,
# throughput) -- see kaggle_runner/runs/backbone-sweep/diagnostic_output/backbone_sweep.json
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


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

    def predict(self, prompt: str, candidates: list[str], calibration_prior: np.ndarray | None) -> str:
        """calibration_prior is required (not optional-with-a-default) so
        every call site states explicitly whether it wants calibrated or
        raw scoring -- see agent.policy.calibrate_scores and
        ClosedSetPolicy.measure_label_prior for how to obtain a real prior.
        Pass None for deliberately uncalibrated scoring (e.g. Stage 2's
        raw-chance diagnostic, which needs to see the uncalibrated signal)."""
        scores = self.score_candidates(prompt, candidates)
        if calibration_prior is not None:
            scores = calibrate_scores(scores, calibration_prior)
        return candidates[int(np.argmax(scores))]

    def measure_label_prior(self, prompt: str, candidates: list[str]) -> np.ndarray:
        """Convenience alias for score_candidates, named for the calibration
        use case: call this with a neutral prompt (no task-specific
        information) to find the model's baseline preference among
        candidates before any real signal is added. See calibrate_scores
        below for what to do with the result."""
        return self.score_candidates(prompt, candidates)

    @torch.no_grad()
    def score_candidates_batch(self, prompts: list[str], candidates: list[str]) -> np.ndarray:
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
        batch = batch.to(self.device)
        attention_mask = attention_mask.to(self.device)

        logits = self.model(input_ids=batch, attention_mask=attention_mask).logits
        log_probs = torch.log_softmax(logits.float(), dim=-1)

        flat_scores = np.zeros(len(sequences), dtype=np.float64)
        for row in range(len(sequences)):
            n_prompt = prompt_lens[row]
            n_cand = cand_lens[row]
            total = 0.0
            for offset in range(n_cand):
                position = n_prompt + offset - 1
                token_id = batch[row, n_prompt + offset].item()
                total += log_probs[row, position, token_id].item()
            flat_scores[row] = total

        return flat_scores.reshape(len(prompts), len(candidates))


def calibrate_scores(raw_scores: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Subtract a previously-measured label prior from raw closed-set
    scores, broadcasting over the last axis. Removes whatever preference
    among candidates exists independent of the prompt's actual content --
    see ClosedSetPolicy.measure_label_prior. Works for a single prompt's
    score vector (n_candidates,) and a batch's score matrix
    (n_prompts, n_candidates) alike, since numpy broadcasts a
    (n_candidates,) prior across either shape's last axis."""
    return raw_scores - prior
