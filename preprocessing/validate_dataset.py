
from pathlib import Path
import pandas as pd

ROOT = Path(r"ISL_CSLRT_Corpus")

ANNOTATIONS = Path(r"ISL_CSLRT_Corpus/corpus_csv_files")

frame_df = pd.read_excel(
    ANNOTATIONS / "ISL_CSLRT_Corpus_frame_details.xlsx"
)

word_df = pd.read_excel(
    ANNOTATIONS / "ISL_CSLRT_Corpus_word_details.xlsx"
)

video_df = pd.read_excel(
    ANNOTATIONS / "ISL_CSLRT_Corpus details.xlsx"
)

gloss_df = pd.read_csv(
    ANNOTATIONS / "ISL Corpus sign glosses.csv"
)


def resolve_path(path):
    """Convert annotation paths into local paths."""
    path = str(path).replace("\\", "/")
    parts = Path(path).parts

    # Annotation paths may begin with ISL_CSLRT_Corpus/
    if parts and parts[0].lower() == "isl_cslrt_corpus":
        path = "/".join(parts[1:])

    return ROOT / path


def check_paths(df, column):
    paths = df[column].dropna().astype(str)
    exists = paths.map(
        lambda p: resolve_path(p).is_file()
    )

    print(f"\n{column}")
    print("Total paths:", len(paths))
    print("Existing:", int(exists.sum()))
    print("Missing:", int((~exists).sum()))

    if (~exists).any():
        print("Examples of missing paths:")
        print(paths[~exists].head(10).to_list())


print("=== DATASET SUMMARY ===")
print("Frame annotations:", len(frame_df))
print("Word image annotations:", len(word_df))
print("Video annotations:", len(video_df))
print("Gloss annotations:", len(gloss_df))

print("\nUnique sentence labels:")
print("Frames:", frame_df["Sentence"].nunique())
print("Videos:", video_df["Sentences"].nunique())
print("Glosses:", gloss_df["Sentence"].nunique())

print("\n=== PATH VALIDATION ===")
check_paths(frame_df, "Frames path")
check_paths(word_df, "Frames path")
check_paths(video_df, "File location")

print("\n=== FRAME COUNTS ===")
print(
    frame_df.groupby("Sentence")
    .size()
    .sort_values()
    .describe()
)

print("\n=== VIDEO COUNTS ===")
print(
    video_df.groupby("Sentences")
    .size()
    .sort_values()
    .describe()
)

print("\n=== GLOSS EXAMPLES ===")
print(gloss_df.head(10).to_string(index=False))



"""
=== DATASET SUMMARY ===
Frame annotations: 18863
Word image annotations: 1036
Video annotations: 493
Gloss annotations: 101

Unique sentence labels:
Frames: 97
Videos: 101
Glosses: 101

=== PATH VALIDATION ===

Frames path
Total paths: 18863
Existing: 18812
Missing: 51
Examples of missing paths:
['ISL_CSLRT_Corpus\\Frames_Sentence_Level\\i am hungry\\3\\ i am hungry (1).jpg', 'ISL_CSLRT_Corpus\\Frames_Sentence_Level\\i am hungry\\3\\ i am hungry (10).jpg', 'ISL_CSLRT_Corpus\\Frames_Sentence_Level\\i am hungry\\3\\ i am hungry (11).jpg', 'ISL_CSLRT_Corpus\\Frames_Sentence_Level\\iam hungry\\3\\ i am hungry (12).jpg', 'ISL_CSLRT_Corpus\\Frames_Sentence_Level\\i am hungry\\3\\ i am hungry (13).jpg', 'ISL_CSLRT_Corpus\\Frames_Sentence_Level\\i am hungry\\3\\ i am hungry (14).jpg', 'ISL_CSLRT_Corpus\\Frames_Sentence_Level\\i am hungry\\3\\ i am hungry (15).jpg', 'ISL_CSLRT_Corpus\\Frames_Sentence_Level\\i am hungry\\3\\ i am hungry (16).jpg', 'ISL_CSLRT_Corpus\\Frames_Sentence_Level\\i am hungry\\3\\ i am hungry (17).jpg', 'ISL_CSLRT_Corpus\\Frames_Sentence_Level\\i am hungry\\3\\ i am hungry (18).jpg']

Frames path
Total paths: 1036
Existing: 1036
Missing: 0

File location
Total paths: 493
Existing: 492
Missing: 1
Examples of missing paths:
['>>>\xa0']

=== FRAME COUNTS ===
count     97.000000
mean     194.463918
std       67.631363
min       46.000000
25%      145.000000
50%      178.000000
75%      238.000000
max      519.000000
dtype: float64

=== VIDEO COUNTS ===
count    101.000000
mean       4.871287
std        0.808250
min        1.000000
25%        5.000000
50%        5.000000
75%        5.000000
max        7.000000
dtype: float64

=== GLOSS EXAMPLES ===
                    Sentence           SIGN GLOSSES
          are you free today         YOU FREE TODAY
    are you hiding something     YOU HIDE SOMETHING
          bring water for me         BRING WATER ME
              can i help you             I HELP YOU
  can you repeat that please      YOU REPEAT PLEASE
              comb your hair          COMB YOU HAIR
             congratulations       CONGRATULATIIONS
could you please talk slower YOU PLEASE TALK SLOWER
              do me a favour           DO ME FAVOUR
            do not abuse him        DONOT ABUSE HIM
"""