
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import json

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------
ROOT = Path("ISL_CSLRT_Corpus")

MANIFEST_PATH = ROOT / "processed" / "sentence_video_manifest.csv"
OUTPUT_DIR = ROOT / "processed" / "frames_224"
OUTPUT_MANIFEST = ROOT / "processed" / "frame_sequence_manifest.csv"
ERROR_REPORT = ROOT / "processed" / "frame_extraction_errors.csv"

NUM_FRAMES = 32
IMAGE_SIZE = 224

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load manifest
# --------------------------------------------------
df = pd.read_csv(MANIFEST_PATH)

required_columns = {
    "sentence", "video_path", "gloss_sequence"
}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing manifest columns: {missing_columns}")

# --------------------------------------------------
# 3. Uniform frame sampling
# --------------------------------------------------
def extract_video_frames(video_path, output_folder):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Video contains no frames: {video_path}")

    # Sample evenly across the entire video
    indices = np.linspace(
        0, total_frames - 1, NUM_FRAMES
    ).round().astype(int)

    output_folder.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for i, frame_index in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = cap.read()

        if not success:
            cap.release()
            raise RuntimeError(
                f"Failed reading frame {frame_index} from {video_path}"
            )

        # OpenCV reads BGR; convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to model input dimensions
        frame = cv2.resize(
            frame,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA
        )

        # Save as JPEG (OpenCV writer expects BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        output_path = output_folder / f"frame_{i:03d}.jpg"

        if not cv2.imwrite(
            str(output_path),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        ):
            cap.release()
            raise RuntimeError(f"Could not save {output_path}")

        saved_paths.append(str(output_path))

    cap.release()

    return saved_paths

# --------------------------------------------------
# 4. Extract each video's frames
# --------------------------------------------------
records = []
errors = []

for idx, row in df.iterrows():
    video_path = Path(row["video_path"])

    # Stable folder name based on manifest row
    video_id = f"video_{idx:04d}"
    output_folder = OUTPUT_DIR / video_id

    try:
        frame_paths = extract_video_frames(
            video_path,
            output_folder
        )

        records.append({
            "video_id": video_id,
            "sentence": row["sentence"],
            "video_path": str(video_path),
            "gloss_sequence": row["gloss_sequence"],
            "num_frames": len(frame_paths),
            "frame_paths": json.dumps(frame_paths),
            "success": True
        })

    except Exception as exc:
        errors.append({
            "video_id": video_id,
            "sentence": row["sentence"],
            "video_path": str(video_path),
            "error": str(exc)
        })

    if (idx + 1) % 25 == 0 or idx + 1 == len(df):
        print(f"Processed {idx + 1}/{len(df)} videos")

# --------------------------------------------------
# 5. Save output manifests
# --------------------------------------------------
result_df = pd.DataFrame(records)
result_df.to_csv(OUTPUT_MANIFEST, index=False)

pd.DataFrame(
    errors,
    columns=["video_id", "sentence", "video_path", "error"]
).to_csv(ERROR_REPORT, index=False)

print("\n========== EXTRACTION SUMMARY ==========")
print("Videos in input manifest:", len(df))
print("Successfully processed:", len(records))
print("Failed videos:", len(errors))
print("Frames per successful video:", NUM_FRAMES)
print("Total extracted frames:", len(records) * NUM_FRAMES)
print("Frame size:", f"{IMAGE_SIZE} x {IMAGE_SIZE}")
print("Frame directory:", OUTPUT_DIR)
print("Sequence manifest:", OUTPUT_MANIFEST)
print("Error report:", ERROR_REPORT)
