# Handoff: submitting and running the RTE models

Everything is already trained. Nothing here needs training or a GPU.

## What you receive

| File | What it is |
|---|---|
| `rte_code_submission.zip` | All code, the report, the model card, dev predictions, executed Colab notebooks (with training logs) |
| `rte_models.zip` | The three trained models (A, B, C), about 0.75 GB; unzips to `models/` |
| `report/report.pdf` (inside the code zip) | The report to upload |
| `submission/model_card.pdf` (inside the code zip) | One-page description of the three models |

The three models are all DeBERTa-v3-xsmall:

| | Model | Params | Validation acc. | Dev acc. |
|---|---|---|---|---|
| A | masked knowledge distillation, seed 21 | 70.8M | 82.7 | 78.0 |
| B | A with a pruned vocabulary | 45.5M | 79.8 | 78.3 |
| C | plain fine-tuning, seed 21 | 70.8M | 81.0 | 80.9 |

## Before 15 Nov: submit

1. Fill in the name and roll number in `report/report.tex` and `submission/model_card.tex`, then rebuild both PDFs. Or ask for them to be rebuilt.
2. Upload `report/report.pdf` and the code (`rte_code_submission.zip`). Attach `submission/model_card.pdf` if there is room.

## Before 16 Nov: set up your laptop once (about 15 minutes, needs internet)

You need **Python 3.12** (3.13 also works). Get it from python.org. On Windows, tick "Add python.exe to PATH" during install.

Unzip `rte_code_submission.zip`, open a terminal in the `rte-challenge` folder, then:

**Windows (PowerShell)**
```
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install --no-deps -e .
```

**macOS / Linux**
```
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install --no-deps -e .
```

The install downloads about 1 GB (PyTorch is most of it).

Then unzip `rte_models.zip` **inside** the `rte-challenge` folder, so that you have `rte-challenge/models/A_...`, `B_...`, `C_...`.

**Dry run.** Do this once, ideally with Wi-Fi turned off to prove it works offline:
```
python scripts/predict_test.py --input submission/dev_predictions/C_plain_full_70.8M_predictions.jsonl --out dryrun
```
Expected: three lines, one per model, with accuracy **0.7798 (A), 0.7834 (B), 0.8087 (C)**, each taking a few seconds.

## 16 Nov, in class: run the test set

1. Activate the environment again (`.venv\Scripts\activate` on Windows, or `source .venv/bin/activate`).
2. Download the test file into the `rte-challenge` folder.
3. Run:
   ```
   python scripts/predict_test.py --input <test file name>
   ```
4. The predictions are in `predictions/`: one CSV and one JSONL per model (`A_...`, `B_...`, `C_...`).
   - CSV columns: `idx, prediction, label_id, p_entailment`.
   - `label_id` uses the GLUE convention: 0 = entailment, 1 = not_entailment.
   - Hand in the format the TAs ask for.

The script accepts `.jsonl`, `.json`, `.csv` or `.tsv`. It looks for `premise`/`hypothesis` columns, or `sentence1`/`sentence2`. If it says a column is missing, open the file, check its column names, and rename them to `premise` and `hypothesis`.

**If the laptop fails:** open `notebooks/colab_inference.ipynb` in Colab. Upload the code zip, the models zip and the test file, then run all cells; it downloads `predictions.zip`. Keep both zips in your Google Drive beforehand, because uploading 0.75 GB in class is slow. The notebook can copy them from Drive instead (set `DRIVE_DIR`).

## For the viva

- Read the report (3 pages). The key decisions are in Section 3: why these three models, and whether more parameters are worth it.
- `notebooks/executed/` holds the original Colab runs with all their outputs, as proof of training.
