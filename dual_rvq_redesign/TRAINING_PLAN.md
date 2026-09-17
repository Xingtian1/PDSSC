# Training Plan

This file defines the staged training and validation strategy for the new dual-RVQ project.

## Overview

The project uses two main stages:

1. `Stage 1`: train shared encoder + dual RVQ branches + decoder
2. `Stage 2`: train channel robustness + parent correction codebooks + flow recovery

Each stage must have:

- one train entrypoint
- one smoke test
- one stage-local validation signal

## Stage 1

Goal:

- learn semantic branch + acoustic branch representation
- obtain stable fine-code reconstruction

Main losses:

- waveform L1
- latent branch balance loss
- RVQ commitment loss

Validation:

- forward pass works
- reconstructed waveform shape matches input
- loss decreases on a tiny subset

## Stage 2

Goal:

- simulate symbol errors through `QPSK + AWGN`
- train semantic/acoustic parent codebooks
- train flow model on corrupted / fallback latent restoration

Main losses:

- corrected / flow-restored waveform L1
- parent consistency loss
- flow latent velocity or latent reconstruction loss
- confidence-aware penalty

Validation:

- corrupted symbols differ from transmitted symbols at low enough `Eb/N0`
- parent fallback path runs end-to-end
- flow path runs end-to-end
- corrected reconstruction is finite

## Execution policy

The implementation should support:

- `--stage stage1_ed_fine`
- `--stage stage2_channel_parent_flow`

And tests should support:

- `--stage stage1_ed_fine`
- `--stage stage2_channel_parent_flow`
- `--stage all`
