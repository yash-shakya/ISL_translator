
from pathlib import Path
import cv2
import pandas as pd

ROOT = Path("ISL_CSLRT_Corpus")
MANIFEST = ROOT / "processed" / "sentence_video_manifest.csv"
OUTPUT = ROOT / "processed" / "video_properties.csv"

df = pd.read_csv(MANIFEST)

results = []

for index, row in df.iterrows():
    path = Path(row["video_path"])
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        results.append({
            "sentence": row["sentence"],
            "video_path": str(path),
            "readable": False,
            "frames": 0,
            "fps": 0,
            "duration_sec": 0,
            "width": 0,
            "height": 0,
        })
        cap.release()
        continue

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = frames / fps if fps > 0 else 0

    results.append({
        "sentence": row["sentence"],
        "video_path": str(path),
        "readable": frames > 0 and fps > 0,
        "frames": frames,
        "fps": fps,
        "duration_sec": duration,
        "width": width,
        "height": height,
    })

    cap.release()

props = pd.DataFrame(results)
props.to_csv(OUTPUT, index=False)

print("Total videos:", len(props))
print("Readable videos:", props["readable"].sum())
print("Unreadable videos:", (~props["readable"]).sum())

print("\nDuration statistics (seconds):")
print(props.loc[props["readable"], "duration_sec"].describe())

print("\nVideo resolution counts:")
print(
    props.loc[props["readable"]]
    .groupby(["width", "height"])
    .size()
    .sort_values(ascending=False)
)

print("\nSaved report:", OUTPUT)
