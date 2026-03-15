# AGENTS.md

## Cursor Cloud specific instructions

### Overview
OpenTau is a Python 3.10 VLA (Vision-Language-Action) model training toolchain. It is a single Python package (`opentau`) managed by `uv` — no separate frontend/backend services, databases, or Docker Compose.

### Running lint
```
ruff check src/ tests/
ruff format --check src/ tests/
```
Note: `zero_to_fp32.py` and `grpc/robot_inference_pb2*.py` are excluded from lint in `.pre-commit-config.yaml`, so ruff will report issues in those files but they are expected.

### Running tests
```
source .venv/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl WANDB_MODE=offline
pytest -m "not gpu" -v tests/
```
- `tests/utils/test_hub.py` requires `HUGGINGFACE_HUB_TOKEN` env var; skip if unavailable.
- `tests/planner/test_planner.py` requires `OPENAI_API_KEY`; skip if unavailable.
- `tests/utils/test_libero_utils.py` and certain `test_factory.py` vector-env tests are skipped in CI (see `.github/workflows/cpu_test.yml`).
- GPU tests: `pytest -m "gpu"` — requires NVIDIA GPU + CUDA.

### CLI entry points
All installed via `pyproject.toml [project.scripts]`: `opentau-train`, `opentau-eval`, `opentau-export`, `opentau-dataset-viz`, `opentau-hierarchical-eval`.

### Environment variables
- `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl` are required for headless MuJoCo/OpenGL rendering in tests/eval.
- `WANDB_MODE=offline` prevents wandb from requiring authentication during tests.

### System dependencies
Python 3.10 (strict `==3.10.*`), `ffmpeg`, `libegl1`, `libegl-mesa0`, `libgl1`, `libglx-mesa0`, `libgles2`, `mesa-utils`, `cmake`, `build-essential` must be installed at the system level.
