# ESSC Transfer Notes

Target standalone project folder:

- `C:\Users\cheng\Desktop\ESSC`

What should be transferred:

- project docs:
  - `README.md`
  - `ARCHITECTURE_SPEC.md`
  - `IMPLEMENTATION_PLAN.md`
  - `TRAINING_PLAN.md`
- runnable scripts:
  - `smoke_test.py`
  - `stage_smoke_tests.py`
- source tree:
  - `src/`

What should NOT be transferred:

- `__pycache__/`
- any temporary outputs

What the standalone ESSC folder still depends on for now:

- local Python environment with current repo dependencies
- existing `speechtokenizer` package code in the current workspace import path

So after transfer, ESSC is a clean project folder for continued development,
but it is not yet fully dependency-vendored.
