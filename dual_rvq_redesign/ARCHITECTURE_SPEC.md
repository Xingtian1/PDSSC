# Dual-RVQ Redesign Spec

This folder is for the new project direction, separate from the conference-code path.

## 1. Confirmed high-level decisions

The new system is **not** a patch on top of the current conference implementation.

Confirmed redesign direction:

- keep the **RVQ-style** quantization idea
- allow each branch to use **multiple RVQ layers**, not only one layer
- split the main path into:
  - semantic codebook branch
  - non-semantic / acoustic / timbre codebook branch
- each RVQ layer has its **own parent correction codebook**
  - one RVQ layer -> one parent codebook
  - parent codebooks are aligned one-to-one with transmitted RVQ layers
- use a **modulation-channel** setting, so symbol errors can happen
- distance-based **detection / replacement** is part of the method
- this should be developed as a **new project folder**, not constrained by the old paper code

## 2. Proposed system structure

Current default version:

```text
waveform x
  -> shared encoder E
  -> split latent h into:
       h_sem
       h_ac

h_sem
  -> semantic RVQ stack
  -> fine semantic codes q_sem
  -> semantic layer-wise parent code t_sem^(l)

h_ac
  -> acoustic RVQ stack
  -> fine acoustic codes q_ac
  -> acoustic layer-wise parent code t_ac^(l)

transmit:
  q_sem, q_ac, t_sem^(l), t_ac^(l)
  -> source/channel coding
  -> modulation
  -> noisy channel
  -> demodulation

receiver:
  fine-code candidate recovery
  + parent-code consistency check
  + distance-based detection
  + fallback replacement using parent code
  -> latent reconstruction
  -> decoder
  -> optional flow refinement
  -> waveform x_hat
```

## 3. Branch design

### 3.1 Shared encoder + split latent

Recommended first implementation:

- keep one shared encoder backbone
- project encoder output into two subspaces
  - `h_sem = P_sem(h)`
  - `h_ac = P_ac(h)`

Reason:

- cheaper than training two full encoders
- keeps semantic/acoustic factorization explicit
- easier to compare against the current conference system

### 3.2 Semantic RVQ branch

Role:

- content
- intelligibility
- coarse linguistic structure

Suggested properties:

- fewer layers than acoustic branch
- stronger semantic supervision
- stronger protection in transmission

### 3.3 Acoustic RVQ branch

Role:

- timbre
- prosody
- residual spectral detail

Suggested properties:

- can use equal or more layers than semantic branch
- weaker semantic constraint
- parent code is especially useful here for graceful degradation

## 4. Correction parent codebooks

Each RVQ layer uses one parent correction codebook.

Structure:

- semantic fine layer code: `q_sem^(l)`
- semantic parent layer code: `t_sem^(l)`
- acoustic fine layer code: `q_ac^(l)`
- acoustic parent layer code: `t_ac^(l)`

Interpretation:

- `q_*^(l)` is the fine quantized symbol of RVQ layer `l`
- `t_*^(l)` is the coarse representative class for the same RVQ layer

Recommended mapping:

```text
q_sem^(l) -> t_sem^(l)
q_ac^(l)  -> t_ac^(l)
```

This should be treated as a **hierarchical codebook relation**, not as two unrelated labels.

## 5. Detection and correction logic

This is the core novelty candidate.

### 5.1 Why parent codes are useful

In a modulation channel, the received fine symbol may be wrong rather than missing.

Then parent codes help in two ways:

1. detection:
   check whether the received fine symbol is consistent with its parent region
2. correction / fallback:
   if not consistent, replace the fine symbol with a parent-level reconstruction

### 5.2 Recommended first decision rule

Do not start with a fully hand-designed global threshold.

Use a two-stage rule:

1. symbol confidence from the demodulation stage
2. parent-consistency distance in embedding space

Example:

```text
if symbol_confidence is high and parent_distance is small:
    keep q
else:
    replace q by parent-based fallback
```

### 5.3 Where to measure distance

Recommended first choice:

- measure distance in the **code embedding space**
- not directly in waveform space
- not directly in raw encoder feature space

Candidate distances:

- cosine distance
- Mahalanobis distance
- learned energy score

Recommended order:

1. cosine distance baseline
2. learned metric later

### 5.4 Parent-based fallback

Two simple options:

1. replace wrong `q` by parent center embedding directly
2. sample or select a representative child code under the parent

Recommended first version:

- use parent center embedding directly

This is simpler and easier to stabilize.

## 6. RVQ design guidance

The new system still uses RVQ, but it should no longer assume all useful progression comes from one single 8-layer chain.

Recommended redesign:

- semantic branch has its own RVQ stack
- acoustic branch has its own RVQ stack

So instead of:

```text
one RVQ stack with L layers
```

use:

```text
semantic RVQ stack with L_sem layers
acoustic RVQ stack with L_ac layers
```

Total bitrate becomes:

```text
R_total
= R_sem_fine + R_ac_fine + R_sem_parent + R_ac_parent
```

If frame rates differ by branch:

```text
R = fps * log2(K) * num_layers
```

for each branch separately.

## 7. Current default configuration

Current default bitrate-controlled setup:

- sample rate: `16 kHz`
- encoder downsample ratio: `500`
- frame rate: `32 Hz`
- semantic fine branch: `L=1`, `K=(1024,)`
- acoustic fine branch: `L=2`, `K=(512, 256)`
- semantic parent branch: `K_parent=(16,)`
- acoustic parent branch: `K_parent=(8, 8)`
- estimated total bitrate: `1.184 kbps`

Bit allocation:

- semantic fine: `32 * log2(1024) = 320 bps`
- acoustic fine: `32 * log2(512) + 32 * log2(256) = 544 bps`
- semantic parent: `32 * log2(16) = 128 bps`
- acoustic parent: `2 * 32 * log2(8) = 192 bps`
- total: `1184 bps`

## 8. Suggested next-round parameter search

Do not search everything at once.

First round should fix frame rate and only search branch depth and codebook size.

Suggested first grid:

- `L_sem in {1, 2}`
- `L_ac in {1, 2, 3}`
- `K_sem in {512, 1024}`
- `K_ac in {512, 1024}`
- `T_sem^(l) in {8, 16, 32}`
- `T_ac^(l) in {8, 16, 32}`

Where:

- `K_*` = fine codebook size
- `T_*` = parent correction codebook size

Only after this should frame rate changes be explored.

## 9. Training roadmap

Recommended staged training:

### Stage A: representation learning

Train encoder + semantic RVQ + acoustic RVQ + decoder.

Losses:

- waveform reconstruction
- mel / STFT reconstruction
- RVQ commitment losses
- semantic distillation loss on semantic branch
- disentanglement regularization between semantic and acoustic branches

### Stage B: parent correction codebook learning

Train or jointly refine parent codebooks so that:

- each fine code maps to a stable parent class
- parent embedding is a valid coarse fallback

Losses:

- parent assignment consistency
- parent-only reconstruction fallback loss
- compactness / separation regularization

### Stage C: modulation-channel robustness

Inject symbol errors through the modulation channel.

Train detection / replacement policy with:

- symbol corruption simulation
- distance-based consistency objective
- corrected reconstruction loss

### Stage D: optional flow refinement

If flow is kept:

- use it only after parent-based correction
- position it as refinement, not sole recovery mechanism

## 9. What should not be done in v1

To keep the project controllable, v1 should avoid:

- changing frame rate, branch structure, distance metric, and modulation type all at once
- adding run-length compression for parent codes
- adding two-level parent hierarchies immediately
- keeping the old packet-loss pipeline as the main experimental path

## 10. Immediate open decisions

These still need to be fixed before implementation:

1. target bitrate range
   - low-only: `1-2 kbps`
   - broader: `1-4 kbps`

2. first modulation setting
   - recommended default: `QPSK + AWGN`

3. branch frame rate
   - same frame rate first
   - different frame rates later

4. flow module status
   - keep as optional refinement
   - or remove in v1 for cleaner ablation

## 11. Recommended v1 baseline

If no further decision is made, use this as the first implementation target:

- shared encoder
- split latent into semantic/acoustic branches
- semantic RVQ: `L_sem = 1`, `K_sem = 1024`
- acoustic RVQ: `L_ac = 2`, `K_ac = 256`
- semantic parent codebook: `T_sem = 16`
- acoustic parent codebook: `T_ac = 16`
- same frame rate for both branches
- modulation channel: `QPSK + AWGN`
- detection distance: cosine distance
- correction: parent-center fallback
- flow: optional refinement only

This version is simple enough to build and strong enough to test the main idea,
while keeping the default total bitrate inside the `1-2 kbps` target range.
