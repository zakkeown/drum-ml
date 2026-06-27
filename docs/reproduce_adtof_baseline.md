# Reproducing the ADTOF baseline (step 1 floor)

ADTOF is our reproducible floor: a real-music, 5-class CRNN with released weights.
We run it in its **own** environment (heavy, version-pinned deps: TensorFlow/Keras
or the PyTorch port, plus `madmom`) and pipe its predictions into `drumml`'s eval
harness via `drumml.baselines.adtof_port`.

## 1. Install ADTOF (separate environment)

Pick one. The PyTorch port is closer to our stack and reports ~-0.2% F vs original.

```bash
# Option A — PyTorch port (recommended)
git clone https://github.com/xavriley/ADTOF-pytorch
# follow its README for weights + a `drumTranscriptor`-equivalent entry point

# Option B — original
git clone https://github.com/MZehren/ADTOF
pip install adtof        # Python 3.10; pulls TF + madmom
```

## 2. Get a test set with public audio

ADTOF/ENST audio is gated, but these are openly downloadable for cross-dataset eval:

- **MDB-Drums** — `git clone https://github.com/CarlSouthall/MDBDrums` (audio included)
- **E-GMD** — https://magenta.tensorflow.org/datasets/e-gmd (CC BY 4.0)
- **IDMT-SMT-Drums** — Zenodo record 7544164

## 3. Transcribe → canonical TSV

```python
from pathlib import Path
from drumml.baselines.adtof_port import run_adtof

audio_dir = Path("datasets/MDBDrums/audio/full_mix")
out_dir = Path("runs/adtof/mdb"); out_dir.mkdir(parents=True, exist_ok=True)
for wav in audio_dir.glob("*.wav"):
    ann = run_adtof(wav, out_dir)            # parses ADTOF output -> canonical
    ann.to_tsv(out_dir / f"{ann.track_id}.tsv")
```

> Adjust `command=`/`extra_args=` in `run_adtof` to match your install's CLI, and
> verify `ADTOF_PITCH_TO_CANONICAL` in `drumml/data/adtof.py` against your release.

## 4. Score (headline = cross-dataset macro-F at 5 classes)

```bash
uv run python scripts/run_eval.py \
    --dataset mdb --root datasets/MDBDrums \
    --pred-dir runs/adtof/mdb --scheme 5
```

Repeat per dataset; the design's headline number is the **mean macro-F across
held-out datasets** (`drumml.eval.cross_dataset_macro_f`). That ADTOF number is
the bar every later iteration (MT3-style seq2seq, then + MERT) must clear.
