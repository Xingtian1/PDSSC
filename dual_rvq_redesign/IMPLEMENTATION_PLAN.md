# Implementation Plan

## Goal

Build a clean new project path for the dual-RVQ + parent-correction design.

## Folder policy

This folder is independent from the conference-paper path.

Do not modify the old paper scripts first.

## Suggested code layout

```text
dual_rvq_redesign/
  ARCHITECTURE_SPEC.md
  IMPLEMENTATION_PLAN.md
  src/
    models/
    quantizers/
    channels/
    training/
    eval/
```

## Build order

1. encoder-decoder backbone
2. branch split module
3. semantic RVQ stack
4. acoustic RVQ stack
5. semantic parent codebook
6. acoustic parent codebook
7. modulation channel simulator
8. distance-based detection module
9. parent fallback reconstruction
10. optional flow refinement

## First coding milestones

### M1

Implement:

- shared encoder
- latent split
- dual RVQ branches
- decoder reconstruction

Success condition:

- clean reconstruction with no channel corruption

### M2

Implement:

- parent codebooks
- parent mapping
- parent-only fallback decode

Success condition:

- parent fallback gives degraded but usable reconstruction

### M3

Implement:

- QPSK + AWGN channel simulator
- symbol corruption pipeline
- distance-based detection

Success condition:

- wrong fine codes can be detected better than random

### M4

Implement:

- correction policy
- full end-to-end corrupted-channel reconstruction

Success condition:

- corrected reconstruction beats no-correction baseline

### M5

Optional:

- add flow refinement after correction

Success condition:

- flow improves corrected reconstruction further

## Immediate next step

Start coding `M1`.
