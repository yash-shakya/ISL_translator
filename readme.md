 # ISL Subnets

Baseline Indian Sign Language (ISL) sentence-recognition project using a frozen
ResNet-18 image encoder and a bidirectional LSTM with CTC loss.

## Requirements

- Windows 10/11
- Python 3.12 or newer
- PowerShell
- Enough disk space for the extracted frames and PyTorch dependencies

The training scripts currently use the CPU. The first training run also needs
internet access so torchvision can download pretrained ResNet-18 weights.

## Create the environment

Open PowerShell in the repository root (`ISL_SUBNETS`) and run:

```powershell
python -m venv .venv
.venv\Scripts\Activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation for the current terminal, run this once and
activate the environment again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

To confirm that the environment is active:

```powershell
python --version
python -c "import torch, torchvision, cv2, pandas; print('Dependencies installed')"
```

## Dataset requirements

Keep the dataset in the repository as:

```text
ISL_CSLRT_Corpus\
	corpus_csv_files\
		ISL_CSLRT_Corpus_details.xlsx
		ISL_Corpus_sign_glosses.csv
	Videos_Sentence_Level\
	Frames_Sentence_Level\
```

The video annotation workbook must be named exactly
`ISL_CSLRT_Corpus_details.xlsx`, because that is the filename used by
`preprocessing/build_manifest.py`. The workbook must contain the `Sentences`
and `File location` columns. The gloss CSV must contain `Sentence` and
`SIGN GLOSSES` columns.

The repository currently contains the corpus folders and some processed files,
but the required annotation workbook is not included in every checkout. Obtain
the dataset and its annotations separately, then place them at the paths above.

## Prepare the dataset

Run these commands from the repository root, in order:

```powershell
python preprocessing/build_manifest.py
python preprocessing/extract_frames.py
python preprocessing/split_dataset.py
python preprocessing/check_glosses.py
python preprocessing/test_dataset.py
```

The commands create these files under `ISL_CSLRT_Corpus\processed`:

- `sentence_video_manifest.csv`
- `frame_sequence_manifest.csv`
- `train.csv`, `validation.csv`, and `test.csv`
- `gloss_vocabulary.txt`

Frame extraction samples 32 frames from each video and resizes them to
224 x 224 pixels. It can take a while and requires additional disk space.

## Train

```powershell
python train.py
```

Training creates a cached feature file at
`ISL_CSLRT_Corpus\processed\feature_cache.pt` and saves checkpoints under
`checkpoints\`. If the feature cache already exists, training reuses it.

## Evaluate and diagnose

After training has produced a checkpoint and feature cache:

```powershell
python evaluate.py
python diagnose.py
```

`evaluate.py` reports validation word error rate and decoded examples.
`diagnose.py` reports blank predictions, class frequencies, and sample decoded
sequences.

## Deactivate the environment

```powershell
deactivate
```
