# coding=utf-8
# Copyright 2024 The Dream team, HKUNLP Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import warnings
import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.distributions as dists
from torch.nn import functional as F
from transformers import __version__
from transformers.generation.configuration_utils import (
    GenerationConfig
)
from transformers.utils import (
    ModelOutput,
    is_torchdynamo_compiling,
    logging,
)

logger = logging.get_logger(__name__)


def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)
    
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0] 
        top2_probs = sorted_probs[:, 1] 
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs 
    
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    
    return confidence, x0


# ── Grand Canonical Ensemble (GCE) decoding helpers ──────────────────────────
#
# Background
# ----------
# The softmax at position n is a Boltzmann distribution with partition function
#   Z_n = Σ_v exp(z_{n,v})
# The decoding energy (surprise of the mode) is:
#   E_n = -log p_max_n = -max_v z_{n,v} + logsumexp(z_n)  ≥ 0
# Zero means perfectly certain; log V means uniform.
#
# The GCE treats the number of tokens committed per step as a random variable K,
# with grand partition function
#   Z_GC(μ) = Σ_{K=0}^{M} exp(β · G_K),    G_K = Σ_{n=1}^{K} (μ - E_{(n)})
# where E_{(1)} ≤ … ≤ E_{(M)} are sorted energies and μ is the chemical
# potential (benefit per token emitted).  K=0 contributes exp(0)=1 (vacuum).
#
# Zero-temperature limit (β → ∞):
#   K* = argmax_K G_K
#      = |{n : E_n < μ}|
#      = |{n : p_max_n > τ}|  where  τ = exp(-μ)
# This is exactly confidence-threshold decoding.
#
# Equivalence to maskgit_plus:
#   maskgit_plus commits K_sched = ⌊M·(1-s/t)⌋ tokens with highest p_max.
#   GCE zero-temp commits all tokens with p_max > τ.  These produce the same
#   committed set iff τ equals the (K_sched)-th largest confidence value, i.e.
#   μ = -log c_{(K_sched)}, where c_{(k)} is the k-th order statistic of p_max.
#   In other words, maskgit_plus uses a data-adaptive μ tied to the diffusion
#   schedule, while GCE fixes μ as a hyperparameter and lets K adapt to the
#   confidence distribution.  GCE strictly generalises maskgit_plus.
#
# μ schedules
# -----------
# μ increases over denoising (progress 0 → 1), meaning τ = e^{-μ} decreases.
# Early in denoising the threshold is high (only very confident tokens committed);
# late in denoising the threshold is low (almost everything committed).
# The schedule shapes the acceleration curve of this relaxation.

def _gce_schedule_mu(cfg, progress: float) -> float:
    """Return the chemical potential μ at normalised denoising progress ∈ [0, 1].

    progress = 0: start of denoising (most positions still [MASK]).
    progress = 1: end of denoising (last step, just before forced commit).

    Schedules
    ---------
    constant : μ = cfg.gce_mu  (schedule-independent; ignores min/max)
    linear   : μ = μ_min + (μ_max - μ_min) · p
    cosine   : μ = μ_min + (μ_max - μ_min) · (1 - cos(π·p)) / 2
                   — symmetric S-curve; slow at both ends, fast in middle.
    sigmoid  : μ = μ_min + (μ_max - μ_min) · norm_σ(p; k)
                   where norm_σ normalises σ(k·(p-½)) to span [0,1],
                   k = cfg.gce_sigmoid_k.  Large k → sharp transition at p=½.

    In all scheduled variants μ_min = cfg.gce_mu_min, μ_max = cfg.gce_mu_max.
    """
    schedule = cfg.gce_mu_schedule

    if schedule == "constant":
        return float(cfg.gce_mu)

    mu_min = float(cfg.gce_mu_min)
    mu_max = float(cfg.gce_mu_max)
    p = float(progress)

    if schedule == "linear":
        return mu_min + (mu_max - mu_min) * p

    elif schedule == "cosine":
        # Standard half-cosine warmup shape: 0 at p=0, 1 at p=1.
        # Derivative is zero at both endpoints → smooth ramping.
        return mu_min + (mu_max - mu_min) * (1.0 - math.cos(math.pi * p)) / 2.0

    elif schedule == "sigmoid":
        k = float(cfg.gce_sigmoid_k)
        # σ(k·(p-½)) evaluated at the endpoints 0 and 1:
        sig_lo = 1.0 / (1.0 + math.exp( k / 2.0))   # σ(-k/2), value at p=0
        sig_hi = 1.0 / (1.0 + math.exp(-k / 2.0))   # σ(+k/2), value at p=1
        span = sig_hi - sig_lo
        if span < 1e-12:
            # k is so large the sigmoid degenerates; fall back to linear
            return mu_min + (mu_max - mu_min) * p
        sig_p = 1.0 / (1.0 + math.exp(-k * (p - 0.5)))
        norm  = (sig_p - sig_lo) / span          # normalised to [0, 1]
        return mu_min + (mu_max - mu_min) * norm

    else:
        raise ValueError(
            f"Unknown gce_mu_schedule '{schedule}'. "
            "Choose from: 'constant', 'linear', 'cosine', 'sigmoid'."
        )


def _gce_compute_K(
    energies: torch.Tensor,
    mu: float,
    beta: float,
    last_step: bool,
) -> int:
    """Compute the number of tokens to commit for one sequence in one step.

    Parameters
    ----------
    energies : Tensor [M]
        Decoding energies E_n = -log p_max_n for the M still-masked positions
        in this sequence.  Values are ≥ 0; smaller means more confident.
    mu : float
        Current chemical potential (from _gce_schedule_mu).
        Corresponding confidence threshold: τ = exp(-μ).
    beta : float
        Inverse temperature β.  Use float('inf') for the zero-temperature
        (deterministic threshold) rule; smaller values allow stochastic K.
    last_step : bool
        If True, return M unconditionally (forced commit: all remaining
        masked tokens are committed regardless of μ and β).

    Returns
    -------
    int
        K ∈ [0, M]: number of tokens to commit.  The caller should then
        select the K positions with highest confidence (lowest energy).

    Algorithm
    ---------
    1. Sort energies ascending: E_{(1)} ≤ E_{(2)} ≤ … ≤ E_{(M)}.
    2. Compute cumulative net benefit:
           G_0 = 0  (commit nothing)
           G_K = Σ_{n=1}^{K} (μ - E_{(n)})
    3. Zero-temperature:  K* = argmax_K G_K
                              = |{n : E_n < μ}|
       Finite-temperature: K ~ Categorical( softmax(β · [G_0, G_1, …, G_M]) )
    4. Safety floor: if K=0 and M>0, set K=1 to guarantee progress.
    """
    M = int(energies.shape[0])
    if M == 0:
        return 0
    if last_step:
        # Final step: forced commit — every remaining [MASK] position is resolved.
        return M

    # ── Select K ──────────────────────────────────────────────────────────
    if math.isinf(beta) or beta > 1e8:
        # Zero temperature: direct threshold rule.
        # K* = |{n : E_n < μ}| — commit every position more confident than τ.
        #
        # This is mathematically equivalent to argmax_K G_K (the cumulative
        # net-benefit formulation), but avoids a cumsum that drifts in
        # bfloat16 when M is large, causing off-by-a-few errors at the
        # boundary where E_n ≈ μ.  The threshold rule is exact regardless
        # of dtype because it tests each energy independently.
        K_star = int((energies < mu).sum().item())
    else:
        # Finite temperature: sample K from the GCE distribution.
        # P(K) ∝ exp(β · G_K), where G_K = Σ_{n=1}^{K} (μ − E_{(n)}).
        # Higher β → more concentrated near the threshold K*.
        #
        # Cast to float32 for the cumsum to avoid bfloat16 drift.
        sorted_e, _ = energies.float().sort()              # [M] ascending, fp32
        net  = mu - sorted_e                               # [M]
        G_K  = net.cumsum(0)                               # [M]
        G    = torch.cat([
            torch.zeros(1, device=G_K.device, dtype=G_K.dtype),
            G_K
        ])                                                 # [M+1]
        log_weights = beta * G                             # [M+1]
        K_star = int(
            torch.distributions.Categorical(logits=log_weights).sample().item()
        )

    # ── Step 4: safety floor — commit at least 1 to guarantee progress
    if K_star == 0:
        K_star = 1

    return K_star


# ── ImprovIsing: 1D Ising model for correlated token commitment ───────────────
#
# ImprovIsing extends GCE decoding by adding nearest-neighbour ferromagnetic
# coupling J between adjacent masked positions in the denoising chain.
#
# Non-interacting GCE (ideal gas):
#   H_GCE(σ) = −Σ_n h_n σ_n,   σ_n ∈ {±1=commit/mask}
#   → each position commits independently when p_max_n > e^{−μ}
#
# ImprovIsing (this section):
#   H_Ising(σ) = −Σ_n h_n σ_n − Σ_n J_n σ_n σ_{n+1}
#   → adjacent masked positions prefer the SAME state (both committed or
#     both masked), encouraging contiguous "islands" of commitment.
#   → J = 0 recovers standard GCE exactly.
#
# Local field:   h_n = μ(n) − E_n,   E_n = −log(p_max_n)
# Nucleation μ:  μ(n) = μ_0 · λ^{d(n)},  d(n)=min distance to a committed pos.
#   → near committed tokens: μ ≈ μ_0 (easy to commit)
#   → far from any context:  μ ≈ 0   (needs very high confidence to nucleate)
# Distance coupling: J_n = J_0 · exp(−γ · max(gap_n − 1, 0))
#   gap_n = seq_pos(n+1) − seq_pos(n); fully adjacent masked pos → J = J_0.
#
# Exact inference via transfer-matrix forward-backward: O(M) time and space,
# negligible compared to the transformer forward pass.
# β → ∞: Viterbi (joint MAP = most probable commitment pattern).
# Finite β: marginal thresholding (softer, stochastic commitment).

def _ising_forward_backward(
    h: torch.Tensor,    # [M] local fields
    J: torch.Tensor,    # [M-1] ferromagnetic couplings
    beta: float,        # inverse temperature
) -> torch.Tensor:
    """
    Exact marginal commit probabilities for the 1D Ising chain.

    Uses log-space transfer-matrix forward-backward to compute
    P(σ_n = +1) for each masked position n, marginalising over all others.

    Hamiltonian: H = −Σ_n h_n σ_n − Σ_n J_n σ_n σ_{n+1}
    Gibbs:       P(σ) ∝ exp(−β H(σ))
    Index convention: dim-index 0 = σ=+1 (commit), 1 = σ=−1 (mask).

    Args:
        h    [M]   local fields; h_n = μ(n) − E_n (positive favours commit)
        J    [M-1] bond couplings; J_n ≥ 0 (ferromagnetic: same-state preferred)
        beta       inverse temperature

    Returns:
        commit_prob [M]: P(σ_n = +1) for each masked position
    """
    M = h.shape[0]
    device, dtype = h.device, h.dtype

    if M == 0:
        return torch.empty(0, device=device, dtype=dtype)
    if M == 1:
        # Single site: P(+1) = sigmoid(2β h)
        return torch.sigmoid(2.0 * beta * h)   # [1]

    # Log site weights: [M, 2]   index 0 = +1 (commit), index 1 = −1 (mask)
    bh = beta * h
    log_site = torch.stack([bh, -bh], dim=-1)                   # [M, 2]

    # Log bond weights: log_bond[n, s, s'] = β J_n σ(s) σ(s')
    # Same-index pairs (++, −−) give +βJ; cross-index (+−, −+) give −βJ.
    bJ = beta * J                                                # [M-1]
    log_bond = torch.stack([
        torch.stack([ bJ, -bJ], dim=-1),   # row s=0 (+1): (++→+J, +−→−J)
        torch.stack([-bJ,  bJ], dim=-1),   # row s=1 (−1): (−+→−J, −−→+J)
    ], dim=-2)                                                   # [M-1, 2, 2]

    # ── Forward pass (log-space) ──────────────────────────────────────────
    # log_alpha[n][s] = log P(σ_1,...,σ_{n−1}, σ_n=s)  (unnormalised)
    log_alpha = torch.zeros(M, 2, device=device, dtype=dtype)
    log_alpha[0] = log_site[0]
    for n in range(M - 1):
        # incoming[s, s'] = log_alpha[n][s] + log_bond[n][s, s']
        incoming = log_alpha[n].unsqueeze(1) + log_bond[n]      # [2, 2]
        # log_alpha[n+1][s'] = logsumexp_s(incoming[:, s']) + log_site[n+1][s']
        log_alpha[n + 1] = torch.logsumexp(incoming, dim=0) + log_site[n + 1]

    # ── Backward pass (log-space) ─────────────────────────────────────────
    # log_beta[n][s] = log P(σ_{n+1},...,σ_M | σ_n=s)  (unnormalised)
    log_beta = torch.zeros(M, 2, device=device, dtype=dtype)    # boundary log(1)=0
    for n in range(M - 2, -1, -1):
        # outgoing[s, s'] = log_bond[n][s,s'] + log_site[n+1][s'] + log_beta[n+1][s']
        outgoing = log_bond[n] + (log_site[n + 1] + log_beta[n + 1]).unsqueeze(0)
        # log_beta[n][s] = logsumexp_s'(outgoing[s, :])
        log_beta[n] = torch.logsumexp(outgoing, dim=1)          # [2]

    # ── Marginals ─────────────────────────────────────────────────────────
    log_unnorm = log_alpha + log_beta                            # [M, 2]
    log_Z      = torch.logsumexp(log_unnorm, dim=1, keepdim=True)
    log_probs  = log_unnorm - log_Z                             # [M, 2]
    return log_probs[:, 0].exp()                                 # [M]: P(commit)


def _ising_viterbi(
    h: torch.Tensor,    # [M] local fields
    J: torch.Tensor,    # [M-1] ferromagnetic couplings
    beta: float = 1.0,  # scale (sign structure determines MAP; β > 0 required)
) -> torch.Tensor:
    """
    Joint MAP configuration for the 1D Ising chain via Viterbi (max-product).

    Finds σ* = argmax_σ [ Σ_n h_n σ_n + Σ_n J_n σ_n σ_{n+1} ]
    i.e. the zero-temperature limit of the Gibbs distribution.
    Runs in log space for numerical stability.

    Args:
        h    [M]   local fields
        J    [M-1] couplings (ferromagnetic; J_n ≥ 0)
        beta       positive scale (does not affect the argmax for β > 0)

    Returns:
        commit_mask [M] bool: True = commit (σ_n = +1)
    """
    M = h.shape[0]
    device, dtype = h.device, h.dtype

    if M == 0:
        return torch.empty(0, dtype=torch.bool, device=device)
    if M == 1:
        return (h > 0)                                           # [1] bool

    bh = beta * h
    log_site  = torch.stack([bh, -bh], dim=-1)                  # [M, 2]
    bJ = beta * J
    log_bond  = torch.stack([
        torch.stack([ bJ, -bJ], dim=-1),
        torch.stack([-bJ,  bJ], dim=-1),
    ], dim=-2)                                                   # [M-1, 2, 2]

    # ── Forward Viterbi ───────────────────────────────────────────────────
    V       = torch.zeros(M, 2, device=device, dtype=dtype)
    backptr = torch.zeros(M - 1, 2, dtype=torch.long, device=device)
    V[0] = log_site[0]
    for n in range(M - 1):
        # candidates[s, s'] = V[n][s] + log_bond[n][s, s']
        candidates = V[n].unsqueeze(1) + log_bond[n]            # [2, 2]
        best_val, best_from = candidates.max(dim=0)              # [2], [2]
        V[n + 1] = best_val + log_site[n + 1]
        backptr[n] = best_from

    # ── Backtrace ─────────────────────────────────────────────────────────
    path = torch.zeros(M, dtype=torch.long, device=device)
    path[M - 1] = V[M - 1].argmax()
    for n in range(M - 2, -1, -1):
        path[n] = backptr[n, path[n + 1]]
    return path == 0                                             # True = commit


def _improvising_commit_step(
    confidence: torch.Tensor,           # [M_b] p_max at each masked position
    energies: torch.Tensor,             # [M_b] E_n = −log(p_max_n)
    seq_positions: torch.Tensor,        # [M_b] absolute sequence indices
    committed_positions: torch.Tensor,  # [C]   absolute sequence indices (committed)
    mu_0: float,                        # base chemical potential (from μ schedule)
    lam: float,                         # nucleation decay λ ∈ (0, 1)
    J0: float,                          # base coupling J_0 ≥ 0
    gamma: float,                       # coupling distance decay γ ≥ 0
    beta: float,                        # inverse temperature (inf → Viterbi)
) -> torch.Tensor:
    """
    ImprovIsing commitment mask for one sequence in one denoising step.

    Builds the 1D Ising model from the local fields and couplings, then
    solves it exactly (Viterbi for β→∞; marginal thresholding for finite β)
    to return a jointly-optimal commitment mask.

    Local field h_n = μ(n) − E_n:
        μ(n) = μ_0 · λ^{d(n)},  d(n) = min distance to any committed position.
        h_n > 0 favours commitment; h_n < 0 favours remaining masked.

    Coupling J_n = J_0 · exp(−γ · max(gap_n − 1, 0)):
        gap_n = seq_pos(n+1) − seq_pos(n) (sequence gap between masked positions).
        Truly adjacent masked positions → J_n = J_0 (full coupling).
        Separated by d committed tokens → J_n = J_0 · e^{−γ d} (screened).

    Returns:
        commit_mask [M_b] bool: True = commit this position
    """
    M      = int(confidence.shape[0])
    device = confidence.device
    ftype  = confidence.dtype

    if M == 0:
        return torch.empty(0, dtype=torch.bool, device=device)

    # ── Position-dependent μ(n) = μ_0 · λ^{d(n)} ─────────────────────────
    # d(n): distance from masked position n to the nearest committed position.
    # Prompt tokens are always committed → the first response positions start
    # close to a committed boundary, naturally biasing nucleation near the prompt.
    if committed_positions.numel() > 0:
        dists = (seq_positions.float().unsqueeze(1) -
                 committed_positions.float().unsqueeze(0)).abs()    # [M_b, C]
        d_n = dists.min(dim=1).values                               # [M_b]
    else:
        # Nothing committed yet: all positions at large symbolic distance.
        # Only very confident positions will nucleate (μ ≈ 0 everywhere).
        d_n = torch.full((M,), 50.0, device=device, dtype=ftype)

    lam_t  = torch.tensor(lam, device=device, dtype=ftype)
    mu_n   = mu_0 * torch.pow(lam_t, d_n.clamp(max=50.0))          # [M_b]

    # ── Local fields h_n = μ(n) − E_n ────────────────────────────────────
    h = (mu_n - energies).to(ftype)                                 # [M_b]

    # ── Distance-decayed couplings J_n ───────────────────────────────────
    # gap = 1: truly adjacent in sequence → J = J_0
    # gap = d+1: d committed tokens between them → J = J_0 · e^{−γ d}
    if M >= 2:
        gaps = (seq_positions[1:] - seq_positions[:-1]).float()     # [M_b − 1]
        J    = J0 * torch.exp(-gamma * (gaps - 1.0).clamp(min=0.0))
        J    = J.to(ftype)                                          # [M_b − 1]
    else:
        J = torch.empty(0, device=device, dtype=ftype)

    # ── Ising inference ───────────────────────────────────────────────────
    if math.isinf(beta) or beta > 1e6:
        # Zero-temperature limit: Viterbi finds the joint MAP configuration.
        commit_mask = _ising_viterbi(h, J, beta=1.0)
    else:
        # Finite temperature: marginal thresholding at P(commit) > 0.5.
        commit_prob = _ising_forward_backward(h, J, beta)
        commit_mask = commit_prob > 0.5

    # ── Safety floor: commit at least 1 position ─────────────────────────
    # Prevents stalling when the Ising solution commits nothing (all h_n < 0
    # and J small). Analogous to GCE's K_star = 1 floor.
    if not commit_mask.any():
        commit_mask = torch.zeros(M, dtype=torch.bool, device=device)
        commit_mask[energies.argmin()] = True   # most confident position

    return commit_mask


@dataclass
class DreamModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None


class DreamGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        self.temperature: float = kwargs.pop("temperature", 0.0)
        self.top_p: Optional[float] = kwargs.pop("top_p", None)
        self.top_k: Optional[int] = kwargs.pop("top_k", None)
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        # diffusion specific params
        self.eps: float = kwargs.pop("eps", 1e-3)
        self.steps: int = kwargs.pop("steps", 512)
        self.alg: str = kwargs.pop("alg", 'origin')
        self.alg_temp: Optional[float] = kwargs.pop("alg_temp", None)

        # ── Grand Canonical Ensemble (GCE) decoding ───────────────────────────
        # alg='gce' uses a chemical-potential threshold to decide per-sample how
        # many tokens to commit at each denoising step, rather than using the
        # diffusion schedule directly.
        #
        # Mathematical correspondence
        # ---------------------------
        # Single-position Boltzmann:  p_{n,v} = softmax(z_n)_v
        # Decoding energy:            E_n = -log p_max_n  (∈ [0, log V])
        # Chemical potential:         μ  (in nats)
        # Confidence threshold:       τ = e^{-μ}  (τ ∈ (0,1))
        #
        # Zero-temperature GCE rule:  commit position n  iff  E_n < μ
        #                                                 iff  p_max_n > τ
        #
        # Equivalence to maskgit_plus:
        #   maskgit_plus fixes K = ⌊M·(1-s/t)⌋ and picks the top-K tokens.
        #   GCE fixes μ and picks all tokens with E_n < μ.  The two rules give
        #   the same committed set when μ = -log c_{(K)}, i.e. when the threshold
        #   equals the confidence of the last token in maskgit_plus's top-K.
        #   GCE subsumes maskgit_plus: it decouples the commit count from the
        #   diffusion schedule, letting the confidence distribution itself
        #   determine how many tokens are ready.
        #
        # Schedules:  μ increases over denoising (progress 0→1), meaning τ
        #   decreases — the model starts conservative and becomes increasingly
        #   aggressive.  Available schedules: 'constant', 'linear', 'cosine',
        #   'sigmoid'.  The 'constant' schedule uses only gce_mu; the others
        #   interpolate from gce_mu_min to gce_mu_max.
        #
        # β (inverse temperature on K):
        #   float('inf')  → deterministic threshold rule  (default)
        #   finite β      → sample K stochastically from the GCE distribution
        #                   P(K) ∝ exp(β · G_K), adding variance to block size.
        #
        # Note: alg_temp still applies inside sample_tokens (randomises WHICH
        # token wins at each position), while gce_beta randomises HOW MANY
        # positions are committed.  They are orthogonal.
        self.gce_mu: float = kwargs.pop("gce_mu", 0.105)
        # ^ constant-schedule μ; default 0.105 ≈ -log(0.90), i.e. τ = 0.90
        self.gce_mu_min: float = kwargs.pop("gce_mu_min", 0.01)
        # ^ start of scheduled range (conservative; τ_max ≈ 0.99)
        self.gce_mu_max: float = kwargs.pop("gce_mu_max", 0.693)
        # ^ end of scheduled range (aggressive; τ_min ≈ 0.50)
        self.gce_mu_schedule: str = kwargs.pop("gce_mu_schedule", "constant")
        # ^ one of: 'constant' | 'linear' | 'cosine' | 'sigmoid'
        self.gce_beta: float = kwargs.pop("gce_beta", float('inf'))
        # ^ inverse temperature β on the K distribution.  'inf' = threshold rule.
        self.gce_sigmoid_k: float = kwargs.pop("gce_sigmoid_k", 8.0)
        # ^ steepness of the sigmoid schedule.  8.0 gives a moderately sharp
        #   transition around p=0.5; increase for a sharper step.

        # ── EOS logit bias ────────────────────────────────────────────────
        # Applied to every denoising step before sampling / commitment.
        # Negative values penalise EOS, making it less likely to win a
        # high-confidence slot early and thus preventing premature termination.
        # Typical range: −3 to −8 (bias of −3 ≈ reduces EOS prob by ~20×).
        # 0.0 = no adjustment (default; backward-compatible).
        self.eos_bias: float = kwargs.pop("eos_bias", 0.0)

        # ── Minimum response length ──────────────────────────────────────
        # Position-specific EOS suppression.  At each denoising step, set
        # logits[:, prompt_len : prompt_len+min_new_tokens, EOS] = −∞
        # so EOS can NEVER be committed at the first min_new_tokens response
        # positions.  Later positions are unaffected — the model can still
        # place EOS correctly at the natural end of the response.
        #
        # Unlike eos_bias (a global penalty that degrades quality by
        # suppressing EOS everywhere), this is a hard positional mask that
        # only affects positions where content is desired.  Positions beyond
        # min_new_tokens predict EOS normally, preserving output quality.
        #
        # Set to 0 to disable (default; backward-compatible).
        self.min_new_tokens: int = kwargs.pop("min_new_tokens", 0)

        # ── ImprovIsing decoding parameters ───────────────────────────────
        # alg='improvising' extends GCE with a 1D ferromagnetic Ising model
        # over the masked positions.  The μ schedule parameters above
        # (gce_mu, gce_mu_schedule, gce_mu_min/max, gce_sigmoid_k) are
        # shared between GCE and ImprovIsing.  The β parameter (gce_beta)
        # controls inference mode: β=∞ → Viterbi (joint MAP), finite β →
        # marginal thresholding.
        self.improvising_J0: float = kwargs.pop("improvising_J0", 0.3)
        # ^ base coupling J_0 ≥ 0.  Sets the surface tension between committed
        #   and masked regions.  J_0=0 recovers standard GCE exactly.
        #   Typical range: 0.1–1.0; start at 0.3.
        self.improvising_gamma: float = kwargs.pop("improvising_gamma", 0.5)
        # ^ coupling distance decay γ ≥ 0.  J_n = J_0 · exp(−γ · (gap−1)),
        #   where gap is the sequence distance between adjacent masked positions.
        #   γ=0: no distance decay (uniform coupling).  γ=1: coupling halves
        #   every ~1.4 committed-token gap.
        self.improvising_lam: float = kwargs.pop("improvising_lam", 0.9)
        # ^ nucleation decay λ ∈ (0,1).  μ(n) = μ_0 · λ^{d(n)} where d(n)
        #   is the distance to the nearest committed position.  λ=0.9 gives
        #   correlation length ξ = −1/log(0.9) ≈ 9.5 tokens.  Smaller λ →
        #   shorter reach (commit only near existing context).

        # Parameters that define the output variables of `generate`
        self.num_return_sequences: int = kwargs.pop("num_return_sequences", 1)
        self.return_dict_in_generate: bool = kwargs.pop("return_dict_in_generate", False)
        self.output_history: bool = kwargs.pop("output_history", False)

        # Special tokens that can be used at generation time
        self.mask_token_id = kwargs.pop("mask_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # Wild card
        self.generation_kwargs = kwargs.pop("generation_kwargs", {})

        # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        # Validate the values of the attributes
        self.validate(is_init=True)

    def validate(self, is_init=False):
        pass

class DreamGenerationMixin:
    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
        # Do not call torch.repeat_interleave if expand_size is 1 because it clones
        # the input tensor and thus requires more memory although no change is applied
        if expand_size == 1:
            return input_ids, attention_mask
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)
        return input_ids, attention_mask

    def _validate_generated_length(self, generation_config, input_ids_length, has_default_max_length):
        """Performs validation related to the resulting generated length"""

        # Can't throw warnings/exceptions during compilation
        if is_torchdynamo_compiling():
            return

        # 1. Max length warnings related to poor parameterization
        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
            # 20 is the default max_length of the generation config
            warnings.warn(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the "
                "generation length. We recommend setting `max_new_tokens` to control the maximum length of the "
                "generation.",
                UserWarning,
            )
        if input_ids_length >= generation_config.max_length:
            input_ids_string = "input_ids"
            raise ValueError(
                f"Input length of {input_ids_string} is {input_ids_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_length` or, better yet, setting `max_new_tokens`."
            )

    def _prepare_generated_length(
        self,
        generation_config,
        has_default_max_length,
        input_ids_length,
    ):
        """Prepared max and min length in generation configs to avoid clashes between similar attributes"""

        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length

        elif has_default_max_length:
            if generation_config.max_length == DreamGenerationConfig().max_length:
                generation_config.max_length = generation_config.max_length + input_ids_length
                max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
                if max_position_embeddings is not None:
                    generation_config.max_length = min(generation_config.max_length, max_position_embeddings)

        return generation_config

    def _prepare_generation_config(
        self, generation_config: Optional[DreamGenerationConfig], **kwargs: Dict
    ) -> DreamGenerationConfig:
        """
        Prepares the base generation config, then applies any generation configuration options from kwargs. This
        function handles retrocompatibility with respect to configuration files.
        """
        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        using_model_generation_config = False
        if generation_config is None:
            generation_config = DreamGenerationConfig.from_model_config(self.config)
            using_model_generation_config = True

        # `torch.compile` can't compile `copy.deepcopy`, arguments in `kwargs` that are part of `generation_config`
        # will mutate the object with `.update`. As such, passing these arguments through `kwargs` is disabled -- an
        # exception will be raised in `_validate_model_kwargs`
        if not is_torchdynamo_compiling():
            generation_config = copy.deepcopy(generation_config)
            _kwargs = generation_config.update(**kwargs)
            # If `generation_config` is provided, let's fallback ALL special tokens to the default values for the model
            if not using_model_generation_config:
                if generation_config.bos_token_id is None:
                    generation_config.bos_token_id = self.generation_config.bos_token_id
                if generation_config.eos_token_id is None:
                    generation_config.eos_token_id = self.generation_config.eos_token_id
                if generation_config.pad_token_id is None:
                    generation_config.pad_token_id = self.generation_config.pad_token_id
                if generation_config.mask_token_id is None:
                    generation_config.mask_token_id = self.generation_config.mask_token_id

        return generation_config

    def _prepare_special_tokens(
        self,
        generation_config: DreamGenerationConfig,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Prepares the special tokens for generation, overwriting the generation config with their processed versions
        converted to tensor.

        Note that `generation_config` is changed in place and stops being serializable after this method is called.
        That is no problem if called within `generate` (`generation_config` is a local copy that doesn't leave the
        function). However, if called outside `generate`, consider creating a copy of `generation_config` first.
        """

        # Convert special tokens to tensors
        def _tensor_or_none(token, device=None):
            if token is None:
                return token

            device = device if device is not None else self.device
            if isinstance(token, torch.Tensor):
                return token.to(device)
            return torch.tensor(token, device=device, dtype=torch.long)

        bos_token_tensor = _tensor_or_none(generation_config.bos_token_id, device=device)
        eos_token_tensor = _tensor_or_none(generation_config.eos_token_id, device=device)
        pad_token_tensor = _tensor_or_none(generation_config.pad_token_id, device=device)
        mask_token_tensor = _tensor_or_none(generation_config.mask_token_id, device=device)

        # We can have more than one eos token. Always treat it as a 1D tensor (when it exists).
        if eos_token_tensor is not None and eos_token_tensor.ndim == 0:
            eos_token_tensor = eos_token_tensor.unsqueeze(0)

        # Set pad token if unset (and there are conditions to do so)
        if pad_token_tensor is None and eos_token_tensor is not None:
            pad_token_tensor = eos_token_tensor[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{pad_token_tensor} for open-end generation.")

        # Update generation config with the updated special tokens tensors
        # NOTE: this must be written into a different attribute name than the one holding the original special tokens
        # (in their non-tensor form), in order to enable end-to-end compilation. See
        # https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html#limitations
        generation_config._bos_token_tensor = bos_token_tensor
        generation_config._eos_token_tensor = eos_token_tensor
        generation_config._pad_token_tensor = pad_token_tensor
        generation_config._mask_token_tensor = mask_token_tensor

    @torch.no_grad()
    def diffusion_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[DreamGenerationConfig] = None,
        **kwargs,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        generation_config = self._prepare_generation_config(generation_config, **kwargs)
        generation_tokens_hook_func = kwargs.pop("generation_tokens_hook_func", lambda step, x, logits: x)
        generation_logits_hook_func = kwargs.pop("generation_logits_hook_func", lambda step, x, logits: logits)

        # 2. Define model inputs
        assert inputs is not None
        input_ids = inputs
        device = input_ids.device
        attention_mask = kwargs.pop("attention_mask", None)
        self._prepare_special_tokens(generation_config, device=device)

        # 3. Prepare `max_length`.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            input_ids_length=input_ids_length,
        )

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)
        
        # 4. Check input_ids
        if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )
        if (
            hasattr(generation_config, "pad_token_id") and
            torch.any(input_ids == generation_config.pad_token_id) and 
            attention_mask is None
        ):
            warnings.warn(
                "Padding was detected but no attention mask is passed here. For correct "
                "generation results, please set `attention_mask` when batch-padding inputs.",
                UserWarning,
            )

        input_ids, attention_mask = self._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask 
        )

        result = self._sample(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            generation_tokens_hook_func=generation_tokens_hook_func,
            generation_logits_hook_func=generation_logits_hook_func
        )
        return result

    def _sample(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        generation_tokens_hook_func,
        generation_logits_hook_func
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # init values
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k
        # GCE parameters (only used when alg='gce' or alg='improvising')
        gce_beta      = generation_config.gce_beta
        gce_sigmoid_k = generation_config.gce_sigmoid_k
        # ImprovIsing parameters (only used when alg='improvising')
        improvising_J0    = generation_config.improvising_J0
        improvising_gamma = generation_config.improvising_gamma
        improvising_lam   = generation_config.improvising_lam
        # EOS-based early stopping: stop once every sequence has an EOS in its
        # response area.  prompt_len marks where the response starts so we never
        # mistake a previous-turn EOS in the prompt for a completion signal.
        eos_token_id = generation_config.eos_token_id
        prompt_len   = input_ids.shape[1]
        # EOS logit bias — subtract from EOS logits at every step to delay
        # premature EOS commitment.  Negative values (penalty) make EOS tokens
        # less likely to rank highly in the confidence ordering, so content
        # tokens fill in before EOS is committed, preventing mid-sentence cutoff.
        eos_bias = generation_config.eos_bias

        histories = [] if (return_dict_in_generate and output_history) else None

        # pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            # attention_mask is of shape [B, N]
            # broadcast to [B, 1, N, N]
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)

        # this allows user-defined token control of the intermediate steps
        x = generation_tokens_hook_func(None, x, None)
        for i in range(steps):
            mask_index = (x == mask_token_id)
            logits = self(x, attention_mask, tok_idx).logits
            logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)

            # this allows user-defined logits control of the intermediate steps
            logits = generation_logits_hook_func(i, x, logits)

            # ── Optional EOS logit bias ───────────────────────────────────────
            # Applied after the user hook so external hooks see unbiased logits.
            # generation_config._eos_token_tensor is a 1-D LongTensor set by
            # _prepare_special_tokens(); it may hold multiple EOS ids.
            if eos_bias != 0.0 and generation_config._eos_token_tensor is not None:
                logits[:, :, generation_config._eos_token_tensor] += eos_bias

            # ── Position-specific EOS suppression (min_new_tokens) ────────────
            # Hard-mask EOS at the first min_new_tokens response positions so
            # the model CANNOT commit EOS there.  Positions beyond this cutoff
            # predict EOS normally, preserving output quality.
            min_new_tokens = generation_config.min_new_tokens
            if min_new_tokens > 0 and generation_config._eos_token_tensor is not None:
                suppress_end = min(prompt_len + min_new_tokens, logits.shape[1])
                logits[:, prompt_len:suppress_end, generation_config._eos_token_tensor] = (
                    torch.finfo(logits.dtype).min
                )

            mask_logits = logits[mask_index]
            t = timesteps[i]
            s = timesteps[i + 1]
        
            if alg == 'origin':
                p_transfer = 1 - s / t if i < steps - 1 else 1
                x0 = torch.zeros_like(x[mask_index], device=self.device, dtype=torch.long) + mask_token_id
                transfer_index_t_s = torch.rand(*x0.shape, device=self.device) < p_transfer
                _, x0[transfer_index_t_s]= sample_tokens(mask_logits[transfer_index_t_s], temperature=temperature, top_p=top_p, top_k=top_k)
                x[mask_index] = x0.clone()

            elif alg == 'gce':
                # ── Grand Canonical Ensemble decoding ────────────────────────
                # Uses p_max confidence (same signal as maskgit_plus) so that
                # E_n = -log p_max_n is the decoding energy in nats.
                #
                # Per-sample K: unlike the other algorithms, which use a single
                # batch-averaged K, GCE computes K independently for each item
                # in the batch.  This is correct because the confidence
                # distribution varies per sequence.
                #
                # Relationship to maskgit_plus:
                #   Both rank masked positions by p_max and commit the top-K.
                #   maskgit_plus: K = ⌊M·(1-s/t)⌋  (schedule-driven, batch-avg)
                #   GCE:          K = |{n : p_max_n > τ_t}|  (threshold-driven,
                #                     per-sample, where τ_t = e^{-μ_t})
                #   They coincide when τ equals the (1-s/t)-th quantile of
                #   confidence values, i.e. μ = -log c_{(K_sched)}.
                #
                # Step structure:
                #   1. Get p_max (confidence) and best token x0 for each masked
                #      position — shape [M] where M = total masked tokens.
                #   2. Use mask_index.nonzero() to recover which batch item and
                #      sequence position each of the M elements belongs to.
                #   3. For each batch item b independently:
                #      a. Compute μ_t from the chosen schedule.
                #      b. Compute K_b via _gce_compute_K (threshold or sampled).
                #      c. Select the K_b positions with highest confidence.
                #      d. Write the committed tokens directly into x[b, pos].

                confidence, x0 = sample_tokens(
                    mask_logits, temperature=temperature, top_p=top_p, top_k=top_k
                )
                # confidence[m] = p_max at the m-th masked position  (scalar ∈ (0,1])
                # x0[m]         = argmax / sampled token at that position

                # Decoding energy E_n = -log p_max_n  ∈ [0, ∞)
                energies = -torch.log(confidence.clamp(min=1e-10))  # [M]

                # normalised progress ∈ [0, 1]; 0 = first step, 1 = last step
                progress = i / max(steps - 1, 1)

                # μ at this step — uses the schedule chosen in DreamGenerationConfig
                mu = _gce_schedule_mu(generation_config, progress)

                # Decompose the flat [M] index back into (batch_item, seq_pos) pairs.
                # mask_index is [B, N] bool; nonzero returns a [M, 2] LongTensor
                # where column 0 = batch index, column 1 = sequence index.
                mask_positions = mask_index.nonzero(as_tuple=False)  # [M, 2]
                batch_idx_of_m = mask_positions[:, 0]                # [M]
                seq_idx_of_m   = mask_positions[:, 1]                # [M]

                is_last_step = (i == steps - 1)

                for b in range(x.shape[0]):
                    # Identify the subset of the M masked positions belonging to b.
                    in_b       = (batch_idx_of_m == b)    # [M] bool mask
                    if not in_b.any():
                        continue

                    conf_b     = confidence[in_b]          # [M_b]
                    x0_b       = x0[in_b]                  # [M_b]
                    energies_b = energies[in_b]            # [M_b]
                    seq_b      = seq_idx_of_m[in_b]        # [M_b] positions in [0, N)

                    # Compute per-sample K using the GCE rule.
                    K_b = _gce_compute_K(
                        energies_b, mu, gce_beta, last_step=is_last_step
                    )

                    if K_b > 0:
                        # Select the K_b tokens with highest confidence (lowest energy).
                        # torch.topk on conf_b returns LOCAL indices into conf_b.
                        _, top_local = torch.topk(conf_b, K_b)
                        # Map back to absolute sequence positions and write tokens.
                        x[b, seq_b[top_local]] = x0_b[top_local]

            elif alg == 'improvising':
                # ── ImprovIsing: 1D Ising model for correlated token commitment ──
                #
                # Like GCE, we compute p_max confidence and run the μ schedule.
                # Unlike GCE, the commit decision is not independent per site:
                # a 1D ferromagnetic Ising model couples adjacent masked positions
                # so they prefer to commit together (nucleation-and-growth).
                #
                # The Ising chain is solved exactly in O(M) time:
                #   β → ∞  (gce_beta slider at 100): Viterbi joint MAP.
                #   finite β: marginal thresholding — softer, stochastic.
                #
                # The μ schedule (gce_mu_schedule, gce_mu_min/max, etc.) is
                # shared with GCE, as is the gce_beta slider.
                # ImprovIsing-specific hyperparameters: J_0, γ, λ.

                confidence, x0 = sample_tokens(
                    mask_logits, temperature=temperature, top_p=top_p, top_k=top_k
                )
                # confidence[m] = p_max at masked position m;  x0[m] = argmax token
                energies = -torch.log(confidence.clamp(min=1e-10))   # [M]

                # Decompose flat [M] index into (batch_item, seq_pos) pairs.
                mask_positions = mask_index.nonzero(as_tuple=False)   # [M, 2]
                batch_idx_of_m = mask_positions[:, 0]                 # [M]
                seq_idx_of_m   = mask_positions[:, 1]                 # [M]

                progress     = i / max(steps - 1, 1)
                mu_t         = _gce_schedule_mu(generation_config, progress)
                is_last_step = (i == steps - 1)

                for b in range(x.shape[0]):
                    in_b = (batch_idx_of_m == b)
                    if not in_b.any():
                        continue

                    conf_b     = confidence[in_b]       # [M_b]
                    x0_b       = x0[in_b]               # [M_b]
                    energies_b = energies[in_b]         # [M_b]
                    seq_b      = seq_idx_of_m[in_b]     # [M_b] absolute seq positions

                    if is_last_step:
                        # Final step: force-commit all remaining positions.
                        x[b, seq_b] = x0_b
                        continue

                    # All currently committed sequence positions in this batch item.
                    # Prompt positions are always committed and act as the initial
                    # nucleus from which the response grows outward.
                    committed_b = (~mask_index[b]).nonzero(as_tuple=True)[0]  # [C]

                    commit_mask_b = _improvising_commit_step(
                        confidence        = conf_b,
                        energies          = energies_b,
                        seq_positions     = seq_b,
                        committed_positions = committed_b,
                        mu_0              = mu_t,
                        lam               = improvising_lam,
                        J0                = improvising_J0,
                        gamma             = improvising_gamma,
                        beta              = gce_beta,    # shared with GCE slider
                    )

                    if commit_mask_b.any():
                        x[b, seq_b[commit_mask_b]] = x0_b[commit_mask_b]

            else:
                if alg == 'maskgit_plus':
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
                elif alg == 'topk_margin':
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, margin_confidence=True)
                elif alg == 'entropy':
                    confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
                else:
                    raise RuntimeError(f"Unknown alg: {alg}")
                num_mask_token = mask_index.sum() / mask_index.shape[0]
                if i < steps - 1:
                    scheduled = int(num_mask_token * (1 - s / t))
                    # Safety floor: always commit at least 1 token when masks remain.
                    # Without this, int() truncation can produce 0 when the remaining
                    # mask count is much smaller than the remaining step budget (e.g.
                    # after high-confidence tail-EOS positions are committed early),
                    # causing the loop to spin with no progress for many steps.
                    number_transfer_tokens = max(scheduled, 1) if num_mask_token > 0 else 0
                else:
                    number_transfer_tokens = int(num_mask_token)
                full_confidence = torch.full_like(x, -torch.inf, device=self.device, dtype=logits.dtype)
                full_confidence[mask_index] = confidence
                if number_transfer_tokens > 0:
                    if alg_temp is None or alg_temp == 0:
                        _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
                    else:
                        full_confidence = full_confidence / alg_temp
                        full_confidence = F.softmax(full_confidence, dim=-1)
                        transfer_index = torch.multinomial(full_confidence, num_samples=number_transfer_tokens)
                    x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id
                    x_[mask_index] = x0.clone()
                    row_indices = torch.arange(x.size(0), device=self.device).unsqueeze(1).expand_as(transfer_index)
                    x[row_indices,transfer_index] = x_[row_indices,transfer_index]

            # this allows user-defined token control of the intermediate steps
            x = generation_tokens_hook_func(i, x, logits)

            if histories is not None:
                histories.append(x.clone())

            # ── All-masks-gone early exit ─────────────────────────────────────
            # If every position is committed (no [MASK] tokens anywhere in x),
            # there is nothing left to do — skip the remaining steps.
            if not (x == mask_token_id).any():
                break

            # ── EOS-based early exit ──────────────────────────────────────────
            # The naive check "any EOS in response area" fires immediately
            # because diffusion models correctly predict EOS at the END of
            # the response from the very first step (before content is filled).
            #
            # The correct condition: for each sequence, find the first committed
            # EOS in the response area.  The response is complete when all
            # positions BEFORE that EOS are also committed (non-MASK).  That
            # means the full content has solidified and EOS marks the true end.
            #
            # Guard: require first_eos >= 1 so that a spurious EOS at position
            # 0 (empty response) does not trigger early exit — those cases just
            # run to completion harmlessly.
            #
            # x[:, :prompt_len] is never examined, so previous-turn EOS tokens
            # inside the prompt cannot trigger a false stop.
            if eos_token_id is not None:
                done = True
                for b in range(x.shape[0]):
                    resp = x[b, prompt_len:]                                  # [R]
                    eos_locs = (resp == eos_token_id).nonzero(as_tuple=True)[0]
                    if len(eos_locs) == 0:                                    # no EOS yet
                        done = False; break
                    first_eos = int(eos_locs[0])
                    if first_eos == 0:                                        # edge-case guard
                        done = False; break
                    if first_eos < min_new_tokens:                            # below min length
                        done = False; break
                    if (resp[:first_eos] == mask_token_id).any():             # content still masked
                        done = False; break
                if done:
                    # Fill everything after the last content token with EOS so
                    # there are no trailing MASK tokens in the output.
                    x = x.masked_fill(x == mask_token_id, eos_token_id)
                    break

        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=x,
                history=histories,
            )
        else:
            return x