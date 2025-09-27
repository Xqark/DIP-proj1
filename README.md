# NCC Colour Tracker Toolkit

This repository explores classical computer vision techniques for vehicle tracking in short colour video sequences. The tracker fuses template-based normalized cross-correlation (NCC) with HSV colour backprojection to stabilize detections even when illumination shifts or backgrounds are cluttered. The current focus is improving red-car lock-on while keeping the blue-car baseline intact.

## Detection Pipeline
- **Template extraction**: For each sequence we crop a reference template based on the ground-truth bounding box.
- **NCC response**: Frames are converted to grayscale and evaluated with NCC against the template to produce a similarity heatmap.
- **HSV backprojection**: The template's colour histogram (in HSV) is projected over the current frame to emphasize pixels with similar chromatic cues.
- **Fusion and bbox decoding**: The normalized NCC and colour probabilities are multiplied, and the peak fused score is decoded into a bounding box used to seed or update the tracker.
- **Tracking loop**: `tracker/NCCColourTracker` refines the fused response frame-to-frame, logging scores and optionally writing annotated videos for review.

## Repository Layout
- `scripts/` — runnable utilities for one-off analysis or batch evaluation.
- `tracker/` — reusable modules for sequence configs, fusion helpers, and the tracking loop.
- `sequences/` — raw frames for the blue and red vehicle sequences. (把老师提供的数据解压放这里)
- `analysis/` — generated overlays, score logs, and optional annotated videos (regenerate locally rather than committing binaries). (`scripts`里的程序会把产生的东西放这)
- `project1.md` — assignment brief; `AGENTS.md` — contributor guidelines (this file).

## Environment Setup

### Python Virtual Environment (Linux, macOS, Windows)
1. Ensure Python 3.9 or newer is installed.
2. Create a virtual environment in the project root:
   - Linux/macOS:
     ```bash
     python3 -m venv .venv
     ```
   - Windows (PowerShell):
     ```powershell
     py -3 -m venv .venv
     ```
3. Activate the environment:
   - Linux/macOS:
     ```bash
     source .venv/bin/activate
     ```
   - Windows (PowerShell):
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
4. Install runtime dependencies:
   ```bash
   pip install -r requirements.txt
   ```

Once the environment is active you can run the scripts exactly as shown below. If you prefer a different dependency manager (e.g., Conda, Poetry), mirror the packages listed in `requirements.txt`.


## Running The Utilities
- `scripts/analyze_blue_ncc.py`
  - Investigates a single frame/sequence, exporting overlays that show NCC, HSV backprojection, fused response, and the detected bounding box.
  - Usage (default blue sequence):
    ```bash
    python scripts/analyze_blue_ncc.py
    ```
    Key options: `--sequence {blue,red}`, `--frame-index <int>`, and `--use-padding` to visualize the raw NCC surface.
- `scripts/evaluate_sequences.py`
  - Runs the full tracker over the blue and/or red sequences, logs per-frame bounding boxes and scores, and optionally saves annotated MP4 videos.
  - Usage (both sequences, videos enabled):
    ```bash
    python scripts/evaluate_sequences.py
    ```
    Helpful flags: `--sequence {blue|red|both}`, `--no-video` to skip MP4 creation, and `--output-dir analysis/evaluation` to relocate artefacts.

Artifacts appear under `analysis/<sequence>/` for the analysis script and `analysis/evaluation/<sequence>/` for evaluation runs.

## Additional Notes
- Generated media (`analysis/` contents) should not be committed; keep runs reproducible by documenting command flags.
- `scripts/analyze_blue_ncc.py`会全局分析一张图片，ncc的template用的是都是第一帧中bounding box截出来的。所以如果处理第200帧，因为车靠近了，ncc效果就不好
- `scripts/evaluate_sequences.py`会处理全部图片然后输出视频，这个里面ncc的template是每次用前一帧的的bounding box截出来然后用`cv2.addWeighted`更新得到的。而且它每次只会分析前一帧bounding box附近的区域
- 其实只用ncc就够了，不管颜色的也可以
