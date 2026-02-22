# ImprovIsing: Locality-Aware Token Commitment via a 1D Ising Model

> A self-contained technical reference for the ImprovIsing decoding algorithm,
> grounded in the Dream-v0 masked diffusion implementation.
> Assumes familiarity with softmax and basic statistical mechanics; all else is derived here.

---

## Table of Contents

1. [The Commitment Problem in Masked Diffusion](#1-the-commitment-problem-in-masked-diffusion)
2. [From Softmax to Boltzmann: Decoding as Statistical Mechanics](#2-from-softmax-to-boltzmann-decoding-as-statistical-mechanics)
3. [Grand Canonical Ensemble (GCE): Committing a Variable Number of Tokens](#3-grand-canonical-ensemble-gce-committing-a-variable-number-of-tokens)
4. [The GCE Ideal Gas: Independent-Site Commitment](#4-the-gce-ideal-gas-independent-site-commitment)
5. [Existing Algorithms as Special Cases](#5-existing-algorithms-as-special-cases)
6. [Why Independence Fails: The Case for Locality](#6-why-independence-fails-the-case-for-locality)
7. [ImprovIsing: The 1D Ising Model for Commitment](#7-improvising-the-1d-ising-model-for-commitment)
8. [The Two Distance Parameters: λ vs γ](#8-the-two-distance-parameters-λ-vs-γ)
9. [Exact Inference: Forward-Backward and Viterbi](#9-exact-inference-forward-backward-and-viterbi)
10. [Implementation in Code](#10-implementation-in-code)
11. [Hyperparameter Guide](#11-hyperparameter-guide)
12. [Side-by-Side Algorithm Comparison](#12-side-by-side-algorithm-comparison)

---

## 1. The Commitment Problem in Masked Diffusion

A masked diffusion language model (DLM) generates text by *iteratively unmasking* a pre-allocated response area. At the start of generation, every output position is a `[MASK]` token:

```
Step 0:   [prompt | MASK MASK MASK MASK MASK MASK MASK MASK]
Step 1:   [prompt | MASK the  MASK MASK MASK MASK MASK MASK]
Step 2:   [prompt | MASK the  cat  MASK MASK MASK MASK MASK]
...
Step T:   [prompt | Once the  cat  sat  on   the  mat  .   ]
```

At each step, the model runs a **single full forward pass** over all positions simultaneously, producing a distribution over the vocabulary at every `[MASK]` position. The central design question is:

> **Which masked positions should be "committed" (unmasked) at each step, and in what order?**

This is the *commitment problem*, and different decoding algorithms answer it differently. ImprovIsing is a new answer that accounts for the spatial structure of language: words that are semantically and syntactically related tend to appear *near each other*, so commitment should spread contiguously rather than happening independently at isolated positions.

---

## 2. From Softmax to Boltzmann: Decoding as Statistical Mechanics

### 2.1 The softmax as a Boltzmann distribution

At each masked position $n$, the model outputs a logit vector $z_n \in \mathbb{R}^V$. The softmax produces:

$$p_{n,v} = \frac{e^{z_{n,v}/T}}{\sum_{v'} e^{z_{n,v'}/T}} \equiv \frac{e^{-\beta \tilde{z}_{n,v}}}{Z_n}$$

where $T$ is sampling temperature, $\beta = 1/T$ is **inverse temperature**, and $\tilde{z}_{n,v} = -z_{n,v}$ plays the role of an energy. This is precisely a **Boltzmann (Gibbs) distribution** over the vocabulary: low-energy (high-logit) tokens have high probability.

The **partition function** $Z_n = \sum_v e^{z_{n,v}/T}$ normalises the distribution. Its log is the *log-partition function* or *free energy*:

$$\log Z_n = T \cdot \log \sum_v e^{z_{n,v}/T} = T \cdot \text{logsumexp}(z_n / T)$$

### 2.2 The decoding energy

The **decoding energy** of position $n$ is the surprise of the most likely token:

$$E_n = -\log p_{n,v^*} = -\log \max_v p_{n,v} \in [0, \log V]$$

where $v^* = \arg\max_v p_{n,v}$. Intuitively:

| $E_n$ | $c_n = e^{-E_n}$ | Interpretation |
|--------|-----------|----------------|
| $0$ | $1.0$ | Model is perfectly certain; one token has all the probability |
| $0.1$ | $0.90$ | Very confident — top token has 90% mass |
| $0.7$ | $0.50$ | Moderate — top token barely preferred |
| $2.3$ | $0.10$ | Uncertain — ten plausible tokens |
| $\log V$ | $1/V$ | Uniform distribution — no idea |

$E_n$ is the natural "cost" of committing position $n$: it measures how much information we need to resolve that position.

### 2.3 The Legendre transform: from per-token to per-step

Committing $K$ tokens costs a total energy $\sum_{k=1}^K E_{(k)}$ (where $E_{(1)} \le \cdots \le E_{(M)}$ are sorted energies). The benefit of committing $K$ tokens, at chemical potential $\mu$ per token, is:

$$G_K = \sum_{k=1}^K (\mu - E_{(k)}) = K\mu - \sum_{k=1}^K E_{(k)}, \qquad G_0 = 0$$

This is a **Legendre transform** of the energy function: we're trading between "number of tokens committed" $K$ and "net free energy gained" $G_K$. The chemical potential $\mu$ acts as the Lagrange multiplier conjugate to $K$.

More formally: define the *microcanonical free energy* $F(K) = \sum_{k=1}^K E_{(k)}$ (minimum energy cost of committing exactly $K$ tokens). Then:

$$G_K = K\mu - F(K)$$

is the Legendre-Fenchel conjugate evaluated at $\mu$. Maximising over $K$ gives the optimal number of tokens:

$$K^*(\mu) = \arg\max_K G_K = \left|\{n : E_n < \mu\}\right|$$

This is the **confidence threshold rule**: commit every position where the decoding energy is below the chemical potential, equivalently every position where $p_{\max} > e^{-\mu} = \tau$.

---

## 3. Grand Canonical Ensemble (GCE): Committing a Variable Number of Tokens

### 3.1 The GCE distribution over K

In the **grand canonical ensemble** we don't fix $K$ — instead we associate each possible $K$ with a statistical weight:

$$P_\beta(K) = \frac{e^{\beta G_K}}{Z_\text{GC}(\mu)}, \qquad Z_\text{GC}(\mu) = \sum_{K=0}^M e^{\beta G_K}$$

This is a Gibbs distribution over *how many tokens to commit*. The inverse temperature $\beta$ controls concentration:

- $\beta \to \infty$: all weight on $K^* = \arg\max G_K$ — **deterministic threshold rule**.
- Finite $\beta$: a soft distribution around $K^*$ — **stochastic commit count**.

### 3.2 The threshold rule (zero temperature)

At $\beta \to \infty$:

$$K^* = \arg\max_K G_K = \bigl|\{n : E_n < \mu\}\bigr| = \bigl|\{n : c_n > \tau\}\bigr|, \quad \tau = e^{-\mu}$$

Because $G_K - G_{K-1} = \mu - E_{(K)}$: $G_K$ increases when $E_{(K)} < \mu$ and decreases thereafter. The maximum is at the last $K$ for which $E_{(K)} < \mu$.

```python
sorted_e, _ = energies.sort()          # ascending: cheapest first
net  = mu - sorted_e                   # positive = beneficial to commit
G    = torch.cat([zeros(1), net.cumsum(0)])   # G[0]=0, G[k] = Σ_{i<k} net_i
K_star = int(G.argmax())               # last index where gains outweigh costs
```

**Visualisation of G_K:**

```
Sorted energies:  0.05  0.08  0.12  0.20  0.35  0.60  0.90  1.20
μ = 0.30:         +0.25 +0.22 +0.18 +0.10 -0.05 -0.30 -0.60 -0.90
G_K:          0   0.25  0.47  0.65  0.75  0.70  0.40 -0.20 -1.10
                                           ↑
                                         K* = 4
```

The first four positions (energies 0.05–0.20) have positive net gains; the fifth (energy 0.35 > μ=0.30) would reduce the total, so we stop there. GCE naturally commits a variable number of tokens per step based on the actual confidence distribution — no schedule required.

---

## 4. The GCE Ideal Gas: Independent-Site Commitment

### 4.1 Factorisation

At finite $\beta$, the GCE partition function factorises beautifully:

$$Z_\text{GC}(\mu) = \sum_{K=0}^M e^{\beta G_K}$$

$$= \sum_{\sigma \in \{0,1\}^M} \exp\!\left(\beta \sum_{n=1}^M \sigma_n (\mu - E_n)\right)$$

$$= \prod_{n=1}^M \left(1 + e^{\beta(\mu - E_n)}\right)$$

This is a product of **independent Bernoulli factors** — the GCE is equivalent to independently deciding, for each masked position $n$:

$$P(\text{commit } n) = \sigma\!\bigl(\beta(\mu - E_n)\bigr) = \frac{1}{1 + e^{-\beta(\mu - E_n)}}$$

In physics language: the GCE is a **non-interacting (ideal) gas** of "commitment particles." Each site independently decides to commit based only on its own energy $E_n$ and the global chemical potential $\mu$.

### 4.2 The Hamiltonian picture

Defining $\sigma_n = +1$ (commit) or $-1$ (keep masked), the GCE energy is:

$$H_\text{GCE}(\sigma) = -\sum_{n=1}^M h_n \sigma_n, \qquad h_n = \mu - E_n$$

where $h_n$ is the **local field** at site $n$: positive means the model favours committing, negative means keep masked.

The Boltzmann weight is $e^{-\beta H(\sigma)} = e^{\beta \sum_n h_n \sigma_n}$, which factorises over sites — confirming independence.

**Key limitation:** the ideal gas has no spatial awareness. A position deep in the middle of an unresolved patch is treated identically to a position adjacent to already-committed text. Language doesn't work this way — context propagates locally.

---

## 5. Existing Algorithms as Special Cases

| Algorithm | Commitment signal | K source | Per-sample? | Spatial awareness |
|-----------|------------------|----------|-------------|-------------------|
| `origin` | random flip | Bernoulli($1-s/t$) | Per-position | None |
| `maskgit_plus` | $p_{\max}$ | schedule $\lfloor \bar{M}(1-s/t)\rfloor$ | **No** (batch-global) | None |
| `topk_margin` | $p_1 - p_2$ | schedule | **No** | None |
| `entropy` | $-H(p)$ | schedule | **No** | None |
| `gce` | $p_{\max}$ | chemical potential $\mu$ | **Yes** | **None** |
| **`improvising`** | $p_{\max}$ + Ising | Viterbi / marginals | **Yes** | **Yes** |

All five existing algorithms share a critical property: **commitment decisions at different positions are independent conditioned on the model's outputs**. ImprovIsing breaks this.

### maskgit_plus as a GCE special case

`maskgit_plus` commits $K_\text{sched} = \lfloor \bar{M}(1-s/t) \rfloor$ tokens globally. At zero temperature, GCE commits $K^* = |\{n : c_n > \tau\}|$ per sample. They select the *same set* when:

$$\tau = c_{(K_\text{sched})} \quad \Longleftrightarrow \quad \mu = -\log c_{(K_\text{sched})}$$

i.e., when the GCE threshold equals the confidence of the last committed token in maskgit\_plus. GCE strictly subsumes maskgit\_plus: it replaces the schedule-driven $K$ with a confidence-adaptive $K^*$.

---

## 6. Why Independence Fails: The Case for Locality

### 6.1 The "island problem"

Consider this partially-denoised response after step 3:

```
[prompt | MASK the  MASK quick MASK MASK jumped MASK MASK MASK]
  pos:    0    1    2    3     4    5    6       7    8    9
```

The model has committed "the" (pos 1), "quick" (pos 3), "jumped" (pos 6). Now at step 4, it must decide which remaining masks to commit. The ideal gas (GCE) computes:

```
pos 0: E_0 = 0.15   (fairly confident → "Once" or "The")
pos 2: E_2 = 0.30   (moderate → "brown" or "lazy")
pos 4: E_4 = 0.80   (uncertain → "fox" or "dog" or "rabbit")
pos 5: E_5 = 0.20   (confident → "fox")
pos 7: E_7 = 0.25   (confident → "over")
pos 8: E_8 = 0.60   (uncertain → "the" or "a")
pos 9: E_9 = 0.15   (confident → "fence")
```

With $\mu = 0.35$, GCE commits positions 0, 2, 5, 7, 9 (energies < 0.35):

```
[prompt | Once the  brown quick MASK fox  jumped over MASK fence]
```

But positions 5 ("fox") and 7 ("over") are **isolated**: they've jumped over positions 4 and 8 that are still uncertain. The resulting intermediate text is:

```
"Once the brown quick [MASK] fox jumped over [MASK] fence"
```

This is linguistically fine — fox will likely resolve to "fox" given context. But committing position 5 ("fox") without position 4 ("quick **fox**") creates a structural hole. ImprovIsing recognises that positions 4 and 5 are adjacent in the mask chain and tries to commit them together.

### 6.2 Locality in language

Linguistic coherence is inherently **local**: grammar operates within phrases, phrases within clauses, clauses within sentences. When the model is uncertain about a position (high $E_n$), it's often because the surrounding context is still unresolved. Committing neighbouring positions *jointly* — even at slightly higher cost — can unlock coherent phrase-level commitment that independent decoding misses.

---

## 7. ImprovIsing: The 1D Ising Model for Commitment

### 7.1 The Hamiltonian

ImprovIsing extends the GCE ideal-gas Hamiltonian with a **ferromagnetic nearest-neighbour coupling** between adjacent masked positions:

$$H_\text{Ising}(\sigma) = -\sum_{n=1}^M h_n \sigma_n \;-\; \sum_{n=1}^{M-1} J_n \sigma_n \sigma_{n+1}$$

where:

- $\sigma_n \in \{+1, -1\}$: spin $+1$ = commit, spin $-1$ = keep masked
- $h_n = \mu(n) - E_n$: local field at position $n$ (positive favours committing)
- $J_n \geq 0$: ferromagnetic coupling between adjacent positions $n$ and $n+1$ (positive = same-state preferred)

Setting $J_n = 0$ recovers $H_\text{GCE}$ exactly. The coupling term $-J_n \sigma_n \sigma_{n+1}$ lowers energy when $\sigma_n = \sigma_{n+1}$ (both commit or both stay masked), which creates **domain walls at commitment boundaries** rather than scattered isolated commitments.

### 7.2 The nucleation-and-growth picture

The Ising model on $\sigma \in \{+1,-1\}^M$ has two stable phases:

| Phase | $\sigma$ | Physical analogy | In decoding |
|-------|----------|-----------------|-------------|
| All $+1$ | all commit | ferromagnet aligned with field | full step commitment |
| All $-1$ | all mask | ferromagnet opposing field | nothing committed |

The interesting dynamics happen at intermediate $h$ and $J$:

1. **Nucleation:** A region with sufficiently positive $h_n$ (high confidence) becomes a seed — it spontaneously flips to $+1$ (commits).
2. **Growth:** Ferromagnetic coupling $J_n$ makes it energetically favourable for neighbours of a committed site to also commit, even if their $h_n$ alone would not suffice.
3. **Propagation:** The committed island grows outward until it hits sites where $h_n$ is too negative (too uncertain) to sustain growth.

This is the **nucleation-and-growth** dynamics of phase transitions — imported directly into the masked diffusion commitment step.

### 7.3 Two distance-dependent parameters

Both $h_n$ and $J_n$ depend on **distances** in the sequence, but they measure different kinds of distance:

**$h_n$: distance to nearest committed token (controlled by $\lambda$)**

$$\mu(n) = \mu_0 \cdot \lambda^{d(n)}, \qquad d(n) = \min_{c \in \text{committed}} |n - c|$$
$$h_n = \mu(n) - E_n = \mu_0 \lambda^{d(n)} - E_n$$

**$J_n$: gap between adjacent masked positions (controlled by $\gamma$)**

$$J_n = J_0 \cdot \exp\!\bigl(-\gamma \cdot \max(\text{gap}_n - 1,\, 0)\bigr)$$
$$\text{gap}_n = \text{seq\_pos}(n+1) - \text{seq\_pos}(n)$$

These are covered in detail in [Section 8](#8-the-two-distance-parameters-λ-vs-γ).

### 7.4 Exact ground states

The 1D Ising model is one of the few exactly solvable models in statistical physics. We exploit this:

- **Zero temperature ($\beta \to \infty$, Viterbi):** the unique ground-state configuration $\sigma^* = \arg\max_\sigma [-H(\sigma)]$ can be found in $O(M)$ time via **Viterbi** (max-product belief propagation).
- **Finite temperature, marginals (forward-backward):** the marginal probabilities $P(\sigma_n = +1)$ can be computed in $O(M)$ time via the **transfer-matrix forward-backward algorithm**, giving a soft commit probability at each position.

Both algorithms are $O(M)$ — negligible relative to the transformer forward pass — so ImprovIsing adds essentially zero computational overhead.

---

## 8. The Two Distance Parameters: λ vs γ

This is the most important conceptual distinction in ImprovIsing. $\lambda$ and $\gamma$ both involve distances, but they affect completely different parts of the Hamiltonian.

```
H(σ) = − Σ_n  h_n  σ_n  −  Σ_n  J_n  σ_n σ_{n+1}
               ↑                    ↑
         local field           coupling
         λ controls           γ controls
         "where to nucleate"  "how far commitment spreads"
```

### 8.1 λ — nucleation reach (local field decay)

**What it is:** the spatial decay of the chemical potential $\mu(n)$ as a function of distance from the nearest *committed* token.

$$\mu(n) = \mu_0 \cdot \lambda^{d(n)}, \qquad d(n) = \min_{c \in \text{committed}} |n - c|$$

**What it controls:** whether a masked position has a strong *intrinsic drive* to commit, based on how close it is to already-committed context.

```
Committed positions:   [committed | MASK MASK MASK MASK MASK MASK]
                                     ↑    ↑    ↑    ↑    ↑    ↑
d(n):                                1    2    3    4    5    6
μ(n) with λ=0.9:                   0.9μ₀ 0.81 0.73 0.66 0.59 0.53
μ(n) with λ=0.7:                   0.7μ₀ 0.49 0.34 0.24 0.17 0.12
```

**Intuition:** $\lambda$ is the "nucleation reach" — how far from an existing committed region a new commitment island can spontaneously form.

- `λ → 1.0`: $\mu(n) \approx \mu_0$ everywhere; distance doesn't matter → standard GCE. Every position is equally willing to nucleate.
- `λ = 0.9`: correlation length $\xi = -1/\log(0.9) \approx 9.5$ tokens. Positions within ~10 tokens of a committed boundary commit easily; positions farther away need much higher intrinsic confidence.
- `λ = 0.7`: short reach $\xi \approx 3$ tokens. Only immediate neighbours of committed regions get a boost.

**What it does NOT do:** $\lambda$ does not directly couple masked positions to each other. It acts on each masked site *independently*, as a spatially-varying external field.

### 8.2 γ — coupling decay (bond strength screening)

**What it is:** the decay of the Ising coupling $J_n$ between consecutive masked positions as a function of the sequence gap between them.

$$J_n = J_0 \cdot \exp\!\bigl(-\gamma \cdot \max(\text{gap}_n - 1,\, 0)\bigr)$$

where $\text{gap}_n = \text{seq\_pos}(n+1) - \text{seq\_pos}(n)$ is the number of tokens between masked positions $n$ and $n+1$ in the original sequence.

```
Sequence:  MASK  MASK  committed  committed  MASK  MASK
           n=0   n=1   (gap=3)               n=2   n=3
Ising chain:  n=0 ←— J_01 = J₀·e^(-γ·(3-1)) = J₀·e^(-2γ) —→ n=2
              n=2 ←— J_23 = J₀·e^(-γ·(1-1)) = J₀·1 (adjacent) —→ n=3
```

**Intuition:** $\gamma$ determines how effectively ferromagnetic coupling "reaches across" committed tokens to link two groups of masked positions.

- `γ = 0`: $J_n = J_0$ regardless of gap. Two groups of masked positions separated by 20 committed tokens are still fully coupled — they will try to commit or stay masked together.
- `γ = 0.5`: each additional committed-token gap between masked positions reduces the coupling by $e^{-0.5} \approx 0.6\times$. Coupling falls off moderately.
- `γ → ∞`: only truly adjacent masked positions ($\text{gap} = 1$) are coupled; separated groups behave independently.

**What it does NOT do:** $\gamma$ does not affect whether individual positions want to commit on their own; it only affects how strongly they influence each other.

### 8.3 Concrete contrast

Consider a response at mid-denoising with two clusters of masked positions:

```
Sequence:  "MASK the MASK quick MASK MASK jumped MASK MASK"
Committed:        ↑         ↑              ↑
MASK chain: [0, 2, 4, 5, 7, 8]
Gaps:          2  2  1  2  1
```

**Effect of λ:** Position 0 ("Once"?) is at distance 0 from the start of the response, but at distance 1 from "the" (committed). Position 8 is at distance 1 from "jumped." With small $\lambda$, position 0 (far from later committed tokens) gets lower $\mu(0)$ and needs higher intrinsic confidence to commit. This biases commitment toward positions close to the "growing front" of committed text.

**Effect of γ:** Positions 4 and 5 are adjacent in the MASK chain (gap=1 → $J = J_0$, full coupling). Positions 2 and 4 are separated by "quick" (gap=2 → $J = J_0 e^{-\gamma}$, reduced coupling). With small $\gamma$, the entire MASK chain is nearly fully coupled — commitment of any one position propagates across the whole sequence. With large $\gamma$, only local clusters (4,5) and (7,8) are tightly coupled; the two clusters decide independently.

### 8.4 Summary table

| Parameter | Controls | Acts on | Distance measured to | Effect of increasing |
|-----------|----------|---------|---------------------|----------------------|
| **λ** | Nucleation reach | Local field $h_n$ | Nearest **committed** position | Shorter reach; nucleation only near committed context |
| **γ** | Coupling decay | Bond strength $J_n$ | Gap between **adjacent masked** positions | Shorter coupling range; groups of masked positions decouple |
| $J_0$ | Coupling strength | Bond strength $J_n$ | — | Stronger preference for contiguous commitment |
| $\mu_0$ | Commitment threshold | Local field $h_n$ | — | More tokens committed per step (shared with GCE $\mu$) |

---

## 9. Exact Inference: Forward-Backward and Viterbi

The 1D Ising model admits exact inference in $O(M)$ time via the **transfer matrix method**. Both algorithms represent probability in log-space to avoid floating-point underflow.

### 9.1 Log-space formulation

Define site energies (index 0 = $\sigma=+1$, index 1 = $\sigma=-1$):

$$\log \phi_n(s) = \begin{cases} +\beta h_n & s = 0 \;(+1) \\ -\beta h_n & s = 1 \;(-1) \end{cases}$$

Define bond energies:

$$\log \psi_n(s, s') = \begin{cases} +\beta J_n & s = s' \quad (\text{same state}) \\ -\beta J_n & s \neq s' \quad (\text{different state}) \end{cases}$$

### 9.2 Viterbi: joint MAP (β → ∞)

The **Viterbi algorithm** finds $\sigma^* = \arg\max_\sigma \exp(-\beta H(\sigma))$ in $O(M)$ via dynamic programming.

**Forward pass** (max-product):

```python
V[0] = log_site[0]                              # initialise at position 0
for n in range(M - 1):
    candidates = V[n].unsqueeze(1) + log_bond[n]   # [2 incoming] × [2 outgoing]
    best_val, best_from = candidates.max(dim=0)     # best incoming state for each s'
    V[n+1] = best_val + log_site[n+1]
    backptr[n] = best_from                          # remember which s led to best
```

**Backward trace** (reading off the MAP path):

```python
path[M-1] = V[M-1].argmax()                    # best final state
for n in range(M-2, -1, -1):
    path[n] = backptr[n, path[n+1]]             # follow optimal predecessor
commit_mask = (path == 0)                       # σ = +1 means commit
```

At $\beta \to \infty$, the site and bond log-weights become:

```
log_site[n, 0] = +∞ · h_n  →  pure sign comparison:  σ_n = sign(h_n + J·σ_prev)
```

The Viterbi MAP is equivalent to finding the minimum-cost cut between the committed (+1) and masked (−1) domains — with domain walls costing $2J_n$ each and field misalignment costing $2|h_n|$.

### 9.3 Forward-backward: marginals (finite β)

The **forward-backward algorithm** computes $P(\sigma_n = +1)$ for each site while marginalising over all other sites.

**Forward messages:**

$$\alpha_n(s) = \phi_n(s) \cdot \sum_{s'} \psi_{n-1}(s', s) \cdot \alpha_{n-1}(s')$$

In log-space:

```python
log_alpha[0] = log_site[0]
for n in range(M - 1):
    incoming = log_alpha[n].unsqueeze(1) + log_bond[n]     # [2, 2]
    log_alpha[n+1] = logsumexp(incoming, dim=0) + log_site[n+1]
```

**Backward messages:**

$$\beta_n(s) = \sum_{s'} \psi_n(s, s') \cdot \phi_{n+1}(s') \cdot \beta_{n+1}(s')$$

In log-space:

```python
log_beta[M-1] = 0                               # boundary: log(1) = 0
for n in range(M-2, -1, -1):
    outgoing = log_bond[n] + (log_site[n+1] + log_beta[n+1]).unsqueeze(0)  # [2, 2]
    log_beta[n] = logsumexp(outgoing, dim=1)
```

**Marginals:**

$$P(\sigma_n = +1) = \frac{\alpha_n(+1)\beta_n(+1)}{\alpha_n(+1)\beta_n(+1) + \alpha_n(-1)\beta_n(-1)}$$

```python
log_unnorm = log_alpha + log_beta               # [M, 2]
log_Z = logsumexp(log_unnorm, dim=1, keepdim=True)
commit_prob = (log_unnorm - log_Z)[:, 0].exp()  # P(σ_n = +1)
commit_mask = commit_prob > 0.5
```

### 9.4 When to use each

| Mode | β slider | Behaviour |
|------|----------|-----------|
| Viterbi (β→∞) | β = 100 (UI max) | Joint MAP: commitment pattern optimises the full chain simultaneously. Sharp boundaries. |
| Marginals (finite β) | β = 1–20 | Per-site soft thresholding at P>0.5. Softer boundaries; commitment can vary more between steps. |

Both give $K \ge 1$ by the safety floor (see below).

---

## 10. Implementation in Code

### 10.1 Building fields and couplings

```python
def _improvising_commit_step(
    confidence, energies, seq_positions, committed_positions,
    mu_0, lam, J0, gamma, beta
):
    M = confidence.shape[0]

    # ── Position-dependent chemical potential μ(n) = μ₀ · λ^{d(n)} ──────
    if committed_positions.numel() > 0:
        dists = (seq_positions.float().unsqueeze(1) -
                 committed_positions.float().unsqueeze(0)).abs()   # [M, C]
        d_n = dists.min(dim=1).values                              # [M]
    else:
        d_n = torch.full((M,), 50.0)   # no committed tokens: extreme distance

    mu_n = mu_0 * lam ** d_n.clamp(max=50.0)    # [M]  μ(n)

    # ── Local field h_n = μ(n) - E_n ──────────────────────────────────
    h = mu_n - energies                          # [M]  positive → wants to commit

    # ── Distance-decayed bond couplings J_n ───────────────────────────
    # gap_n = seq_positions[n+1] - seq_positions[n]
    # gap=1: truly adjacent in sequence (no committed tokens between them)
    gaps = (seq_positions[1:] - seq_positions[:-1]).float()   # [M-1]
    J = J0 * torch.exp(-gamma * (gaps - 1.0).clamp(min=0.0)) # [M-1]

    # ── Ising inference ────────────────────────────────────────────────
    if beta > 1e6:   # zero temperature
        commit_mask = _ising_viterbi(h, J, beta=1.0)
    else:            # finite temperature
        commit_prob = _ising_forward_backward(h, J, beta)
        commit_mask = commit_prob > 0.5

    # ── Safety floor: always commit at least 1 ────────────────────────
    if not commit_mask.any():
        commit_mask = torch.zeros(M, dtype=torch.bool)
        commit_mask[energies.argmin()] = True   # most confident position

    return commit_mask
```

### 10.2 Integration into the denoising loop

```python
elif alg == 'improvising':
    confidence, x0 = sample_tokens(mask_logits, temperature, top_p, top_k)
    energies = -torch.log(confidence.clamp(min=1e-10))

    mask_positions = mask_index.nonzero(as_tuple=False)   # [M, 2]
    batch_idx = mask_positions[:, 0]
    seq_idx   = mask_positions[:, 1]

    progress = i / max(steps - 1, 1)
    mu_t = _gce_schedule_mu(generation_config, progress)  # shared with GCE

    for b in range(B):
        in_b = (batch_idx == b)
        if not in_b.any(): continue

        conf_b     = confidence[in_b]     # [M_b]  p_max at masked positions
        x0_b       = x0[in_b]             # [M_b]  predicted tokens
        energies_b = energies[in_b]       # [M_b]
        seq_b      = seq_idx[in_b]        # [M_b]  absolute sequence positions

        if i == steps - 1:               # final step: force-commit all
            x[b, seq_b] = x0_b; continue

        committed_b = (~mask_index[b]).nonzero(as_tuple=True)[0]   # [C]

        commit_mask_b = _improvising_commit_step(
            confidence=conf_b, energies=energies_b,
            seq_positions=seq_b, committed_positions=committed_b,
            mu_0=mu_t, lam=improvising_lam,
            J0=improvising_J0, gamma=improvising_gamma, beta=gce_beta,
        )

        if commit_mask_b.any():
            x[b, seq_b[commit_mask_b]] = x0_b[commit_mask_b]
```

### 10.3 Data flow summary

```
mask_logits [M, V]
     ↓ sample_tokens
confidence  [M]      p_max at each masked position
energies    [M]      E_n = -log(p_max_n)
     ↓ per batch item b
conf_b      [M_b]
energies_b  [M_b]
seq_b       [M_b]    absolute sequence positions of masked tokens in item b
committed_b [C]      absolute sequence positions of all committed tokens in item b
     ↓ _improvising_commit_step
d_n         [M_b]    distance to nearest committed position  (for λ)
mu_n        [M_b]    μ(n) = μ₀ · λ^{d_n}
h           [M_b]    local field = μ(n) - E_n
gaps        [M_b-1]  sequence gaps between consecutive masked positions
J           [M_b-1]  couplings = J₀ · exp(-γ · (gap-1))
     ↓ _ising_viterbi or _ising_forward_backward
commit_mask [M_b]    bool: True = commit this position
     ↓
x[b, seq_b[commit_mask]] = x0_b[commit_mask]
```

---

## 11. Hyperparameter Guide

### J₀ — base coupling strength

| $J_0$ | Behaviour |
|-------|-----------|
| `0.0` | No coupling → identical to standard GCE |
| `0.1–0.3` | Mild coupling: slight preference for contiguous commitment |
| `0.3–0.5` | Moderate (recommended). Typical $|h_n|$ range is ~0.1–0.7; J in this range competes with the field. |
| `> 1.0` | Strong: Ising model wants all-or-nothing commitment. Can cause entire sequences to commit or none to commit in a single step. |

The balance point is roughly $J_0 \sim \mu_0$: coupling of the same order as the chemical potential means the Ising term can flip marginal sites (those with $|h_n| < J_0$) when their neighbours commit.

### λ — nucleation reach

| $\lambda$ | Correlation length $\xi = -1/\log\lambda$ | Interpretation |
|-----------|------------------------------------------|----------------|
| `1.00` | $\infty$ | No decay → standard GCE |
| `0.95` | 19 tokens | Very long-range; almost no decay |
| `0.90` | 9.5 tokens | Default. Commitment spreads easily within ~10 tokens of context. |
| `0.80` | 4.5 tokens | Medium. Only nearby positions benefit. |
| `0.70` | 3 tokens | Short. Effectively local only. |
| `0.50` | 1.4 tokens | Very short. Almost no propagation. |

Note that at the start of generation, all prompt tokens are committed. The first response position (immediately after the prompt) always has $d(0) = 1$, so it gets $\mu(0) = \mu_0 \cdot \lambda$ — nearly full drive. Generation naturally starts at the left and grows rightward.

### γ — coupling decay rate

| $\gamma$ | Coupling at gap=2 | Coupling at gap=5 | Interpretation |
|----------|-----------------|-----------------|----------------|
| `0.0` | $J_0$ | $J_0$ | Uniform coupling; groups far apart still interact |
| `0.5` | $0.61\,J_0$ | $0.14\,J_0$ | Moderate screening; effective range ~3 gaps |
| `1.0` | $0.37\,J_0$ | $0.018\,J_0$ | Strong screening; coupling negligible beyond 2 gaps |
| `2.0` | $0.14\,J_0$ | $\approx 0$ | Very short range; only adjacent masked positions coupled |

### μ schedule (shared with GCE)

ImprovIsing uses the same $\mu(t)$ schedule as GCE: constant, linear, cosine, or sigmoid. The μ schedule controls the *global* commitment rate; ImprovIsing's spatial parameters $J_0, \lambda, \gamma$ modulate *where* and *how contiguously* tokens commit at each step.

Recommended starting configuration:
```
alg:          improvising
mu schedule:  linear,  μ_min=0.1 → μ_max=0.8
J₀:           0.3
γ:            0.5
λ:            0.9
β:            100 (Viterbi)
```

---

## 12. Side-by-Side Algorithm Comparison

### Commitment rule at step $i$ (single sequence, $M_b$ masked positions)

**`maskgit_plus`**

```
K = floor(M_b * (1 - s/t))           # schedule-driven, same for all sequences
top_K indices by p_max                # independently ranked
```
No spatial awareness. K fixed by diffusion schedule. Batch-global in practice.

---

**`topk_margin`**

```
K = floor(M_b * (1 - s/t))           # schedule-driven
top_K indices by (p_1 - p_2)         # disambiguation signal
```
Same structure as maskgit\_plus, different ranking signal. Better for multi-modal distributions.

---

**`gce` (ideal gas)**

```
K* = |{n : E_n < μ(t)}|              # adaptive per-sample count
top_K* indices by confidence          # independently ranked
```
Per-sample K. Threshold $\tau = e^{-\mu}$ replaces schedule. Still independent.

---

**`improvising` (interacting)**

```
h_n = μ(n) - E_n,  μ(n) = μ₀ · λ^{d(n)}   # position-dependent field
J_n = J₀ · exp(-γ · (gap_n - 1))           # position-dependent coupling
σ* = Viterbi([h_1,...,h_M], [J_1,...,J_{M-1}])  # jointly optimal commitment
```
Per-sample. Spatially aware. K determined by the Ising solution (not a schedule or threshold directly).

### Visual comparison: commitment pattern evolution

With $J_0 = 0$ (GCE), a typical step might commit isolated high-confidence positions:
```
Before: MASK MASK MASK MASK MASK MASK MASK MASK MASK MASK
GCE:    MASK  the MASK MASK  fox MASK MASK  sat MASK MASK
```

With $J_0 = 0.4$, ImprovIsing prefers to grow contiguous islands:
```
Before: MASK MASK MASK MASK MASK MASK MASK MASK MASK MASK
Improv: MASK  the  cat MASK  fox MASK MASK  sat  on MASK
```

The isolated "fox" (high confidence alone) now brings its neighbour "cat" (moderate confidence) along. The cost — slightly larger domain walls — is offset by the benefit of locally coherent phrase commitment.

---

*ImprovIsing is implemented in `generation_utils.py` (`_ising_forward_backward`, `_ising_viterbi`, `_improvising_commit_step`, and the `elif alg == 'improvising':` branch of `_sample`). UI controls are in `app.py` (ImprovIsing sub-group inside the GCE accordion).*
