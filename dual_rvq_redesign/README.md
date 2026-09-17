# Dual RVQ Redesign

This directory contains the new project path for the dual-branch RVQ system.

Current target:

- bitrate range: `1-2 kbps`
- channel: `QPSK + AWGN`
- training: staged

Current default config target:

- semantic fine: `1 x 1024`
- acoustic fine: `(512, 256)`
- semantic parent: `1 x 16`
- acoustic parent: `2 x 8`
- encoder downsample ratio: `500`, frame rate `16000 / 500 = 32 Hz`
- estimated total bitrate: `1.184 kbps`

Code layout:

- `src/config.py`: model / branch / channel config
- `src/models/dual_rvq_model.py`: shared encoder + semantic/acoustic RVQ branches
- `src/quantizers/rvq_branch.py`: branch-local RVQ wrapper
- `src/quantizers/parent_codebook.py`: parent correction codebook
- `src/channels/qpsk_awgn.py`: modulation-channel simulator
- `src/training/stages.py`: staged-training defaults
- `src/training/trainer.py`: staged training entrypoint
- `stage_smoke_tests.py`: per-stage smoke tests

Smoke tests:

```bash
python dual_rvq_redesign/stage_smoke_tests.py --stage all --ebno_db 0.0
```

Training entry examples:

```bash
python dual_rvq_redesign/src/training/trainer.py --stage stage1_ed_fine
python dual_rvq_redesign/src/training/trainer.py --stage stage2_channel_parent_flow --ebno_db 6.0
```

This is intentionally separate from the conference-code path.
