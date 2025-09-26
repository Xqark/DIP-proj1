# Repository Guidelines

## Project Structure & Module Organization
- `scripts/` holds Python utilities; `analyze_blue_ncc.py` runs colour-aware NCC diagnostics.
- `tracker/` contains reusable NCC + colour-fusion tracking utilities shared across sequences. Key modules:
  - `fusion.py` (NCC+HSV fusion helpers, response resizing, bbox extraction)
  - `sequences.py` (sequence configs, frame/template loaders)
  - `ncc_colour_tracker.py` (tracking loop consuming the shared helpers)
- `sequences/` stores raw frame data (`red/` and `blue/` subfolders with numbered JPEGs).
- `analysis/` captures generated plots and overlays. Regenerate artifacts rather than committing large binaries when possible.
- Root contains `project1.md` (assignment brief) and `AGENTS.md` (this guide). Add new modules in dedicated folders to keep the top level tidy.

## Build, Test, and Development Commands
- Create or reuse the project venv: `UV_CACHE_DIR=.uv-cache uv venv .venv`.
- Activate: `source .venv/bin/activate` (or `.venv\\Scripts\\activate` on Windows PowerShell).
- Install runtime deps: `UV_CACHE_DIR=.uv-cache uv pip install opencv-python-headless numpy matplotlib`.
- Run the blue-car feasibility study: `UV_CACHE_DIR=.uv-cache uv run python scripts/analyze_blue_ncc.py` (generates overlays in `analysis/`).

## Coding Style & Naming Conventions
- Python code follows PEP 8 with 4-space indentation. Keep functions short and add comments only for non-obvious logic.
- Name scripts with verbs describing the action (e.g., `track_red_sequence.py`). Use snake_case for module-level variables and functions.
- Avoid hard-coded absolute paths; use `Path` objects rooted at the repo to stay portable.

## Testing Guidelines
- Place automated checks in `tests/` (create the folder if missing). Prefer `pytest`: install via `uv pip install pytest`, run with `uv run pytest`.
- For algorithm experiments, log numeric metrics (peak NCC, confidence ratios) and attach comparison plots to PRs when visual validation is key.

## Commit & Pull Request Guidelines
- Write commits in imperative mood (“Add template selector”, “Refine colour mask”). Group related edits; avoid mixing data dumps with code changes.
- Pull requests should summarize goals, list command outputs or artifacts produced, and reference assignment tasks. Include before/after imagery when altering tracking behaviour.

## Upcoming Implementation Plan
1. Fix red-sequence detection in the existing analysis script so the initial bounding box locks onto the red car instead of roadside distractors; keep the logic script-scoped until behaviour is validated.
2. ✅ Promote the improvements into the `tracker/` library so both sequences can reuse the shared implementation.
3. Build an evaluation script to iterate through each frame, log bounding boxes/scores, and export annotated videos or GIFs for the red and blue sequences (now bootstraps from tracker fusion).
4. Add configurable confidence thresholds and fallback widening logic to handle occlusions or lighting shifts, with metrics reporting for frames lost or re-acquired.
5. Integrate lightweight regression tests (e.g., assert high fused score on first frames) so future changes can be validated automatically.
6. Document usage in `README`-style notes so new contributors can reproduce tracking runs and interpret the output quickly.

## Agent-Specific Notes
- Prefer reproducible scripts over notebooks for auditability. When generating large outputs, add them to `.gitignore` and note how to reproduce.
- Respect the non–deep-learning constraint; document any classical vision techniques you add so reviewers can verify compliance.
- Commit the changes frequently so we can roll back if needed.
