# RTE Challenge: a vocabulary-pruned DeBERTa-v3-xsmall

Binary textual entailment (`entailment` / `not_entailment`) on RTE.

- **Baseline / teacher:** DeBERTa-v3-base, fine-tuned on Colab.
- **Submission:** DeBERTa-v3-xsmall fine-tuned on RTE (checkpoint chosen by validation), then vocabulary-pruned to **45.5M** parameters.
- **Also studied:** knowledge distillation from the teacher (out-of-fold soft labels), DeBERTa-v3-small, a factorised embedding.
- **Evaluation:** GLUE RTE validation set (277 rows), used only for reporting.

| Deliverable | File |
|---|---|
| Report (PDF) | [`report/report.pdf`](report/report.pdf) (source `report/report.tex`) |
| Model card (1 page) | [`submission/model_card.pdf`](submission/model_card.pdf) |
| Dev predictions | [`submission/dev_predictions.csv`](submission/dev_predictions.csv) (`idx, prediction, p_entailment`), same in `.jsonl` |
| Hand-annotated errors | [`report/error_annotation.csv`](report/error_annotation.csv) |

## Results (dev accuracy %, 5 seeds unless noted)

| Model | Total params | Non-embedding | Dev acc | Val acc |
|---|---|---|---|---|
| DeBERTa-v3-base (baseline, teacher) | 184.4M | 86.0M | 84.8 ± 1.1 | 86.2 |
| DeBERTa-v3-small | 141.9M | 43.5M | 76.5 ± 1.4 | 75.1 |
| DeBERTa-v3-xsmall, plain fine-tuning | 70.8M | 21.6M | 77.7 ± 2.1 | 78.5 |
| xsmall + KD (α 0.5, T 2) | 70.8M | 21.6M | 76.8 ± 1.6 | 78.4 |
| xsmall, rank-128 factorised embedding | 38.1M | 21.6M | 51.0 ± 5.0 | 57.6 |
| **Submission**: best-validation xsmall checkpoint (seed 21), vocab-pruned | **45.5M** | 21.6M | **80.1** (1 checkpoint) | 81.0 |

Masked KD and KD (α 0.8, T 4) are pending (`notebooks/colab_sweep_a_distill.ipynb`).

## Environment

Everything runs inside a project-local venv. The system Python is never used.

| Piece | Where |
|---|---|
| Interpreter | `.python-version` → CPython **3.12**, downloaded and managed by [uv](https://docs.astral.sh/uv/) |
| Top-level deps | `requirements.in`, `requirements-dev.in` |
| Pinned lock (universal: macOS + Linux/Colab) | `requirements.txt`, `requirements-dev.txt` |
| Package | `src/rte`, installed editable into `.venv` |

```bash
make venv        # create .venv (Python 3.12) and install the pinned lock + the rte package
make lock        # only after editing requirements*.in: re-resolve the pins
```

On Colab, the notebook installs the same lock but filters out `torch`, `nvidia-*` and `triton`, so Colab keeps its CUDA build of torch.

## Reproduce

```bash
make data        # data/train.jsonl from the challenge zip (2,490), data/dev.jsonl = GLUE RTE validation (277)
# no zip? `python scripts/prepare_data.py --out data` rebuilds train.jsonl from public GLUE RTE,
# byte-identical to the challenge file (fingerprint-checked)
make test        # unit tests (splits, padding, KD loss, param counting, configs)
make baselines   # majority / word-overlap / TF-IDF+LR sanity baselines
```

1. **Baseline and teacher (Colab GPU):** open `notebooks/colab_teacher.ipynb` and run all cells. It
   - trains DeBERTa-v3-base over 5 seeds → `outputs/baseline_deberta_v3_base/`
   - exports out-of-fold teacher logits → `outputs/baseline_deberta_v3_base_oof/teacher_logits_train.npy`

   Download `teacher_artifacts.zip` and unzip it in the project root.
   The student runs (steps 2–3) can also run on Colab via `notebooks/colab_student.ipynb`.
2. **Plain student (the submission model):** `make train CONFIG=configs/student_deberta_v3_xsmall.yaml`
3. **Distilled student (comparison):** `make distill CONFIG=configs/distill_deberta_v3_xsmall.yaml`
4. **Submission (prune the plain student's best-validation checkpoint, export predictions, figures):**
   ```bash
   make submission      # -> outputs/student_deberta_v3_xsmall_pruned/, submission/dev_predictions.{csv,jsonl}
   make figures         # -> report/figures/acc_vs_params.pdf, report/results.json
   make analyse RUN=outputs/student_deberta_v3_xsmall_pruned
   cd report && tectonic report.tex   # report PDF; submission/model_card.tex likewise
   ```
5. **Predictions and error analysis for any run:**
   ```bash
   make predict RUN=outputs/distill_deberta_v3_xsmall
   make analyse RUN=outputs/distill_deberta_v3_xsmall
   ```

### Sweeps (Colab, unattended)

`notebooks/colab_sweep_a_distill.ipynb` (masked KD; alpha 0.8 / T 4) and `notebooks/colab_sweep_b_size.ipynb`
(xsmall with a rank-128 factorised embedding, ~38M params; DeBERTa-v3-small) need no uploads or Drive: Runtime → Run all.
Each run's outputs download as a zip when it finishes.
Result: the rank-128 embedding collapses (dev 51.0, near chance: it discards too much of the
pretrained embedding), and DeBERTa-v3-small (141.9M) is no better than xsmall (76.5 vs 77.7).

### Vocabulary pruning (post-hoc, no retraining)

```bash
python -m rte.prune --checkpoint outputs/distill_deberta_v3_xsmall --out outputs/distill_deberta_v3_xsmall_pruned
python -m rte.predict --checkpoint outputs/distill_deberta_v3_xsmall_pruned --input data/dev.jsonl
```
Keeps the embedding rows of the training-set tokens plus the top-N tokens by the tokenizer's own
unigram frequency; every other token maps to `[UNK]` (the tokenizer is unchanged). N is picked on
train alone (98% token coverage of one half of train by the other half plus top-N). On rows made
only of kept tokens the outputs are identical to the full model. `prune.json` records coverage,
dev agreement with the full model and the params-vs-dev-accuracy curve over N.

### Interrupted runs

- **Resume:** each finished seed (`outputs/<run>/seeds/`) and OOF fold (`outputs/<run>_oof/folds/`) is saved as soon as it completes. Re-running the same command skips them. On Colab, training writes to local disk and copies each finished seed/fold to `MyDrive/rte-challenge/outputs` (set via `RTE_MIRROR_DIR`), so an expired runtime keeps everything finished.
- **Recover metrics from a log:** if the outputs are gone but the cell output survives, paste the `rte.train` log into a file and run
  `python -m rte.recover --config configs/baseline_deberta_v3_base.yaml --log teacher_log.txt`.
  It rebuilds `metrics.json` (per-seed val/dev accuracy, history, parameter counts) and flags it `recovered_from_log`. Dev probabilities, checkpoints and teacher logits cannot be recovered from a log.

## Protocol (what keeps the numbers honest)

- **Dev is never used for selection.** Each seed early-stops on a 10% hold-out of train, grouped by premise. 99 premises appear with several hypotheses, so random splits would leak.
- **5 seeds per configuration, reported as mean ± std.** RTE fine-tuning is seed-sensitive. With n=277, one dev example is 0.36 points and the standard error is about ±2.4 points.
- **Teacher soft labels are out-of-fold.** Each training row is scored by a teacher that never saw it (5 folds, grouped by premise), which avoids overconfident in-sample targets.
- **KD loss:** `alpha * CE(y) + (1 - alpha) * T² * KL(teacher_T || student_T)`, with alpha=0.5 and T=2. The student and distill configs share every other hyperparameter, so the effect of distillation is isolated.
- **Every run writes `outputs/<run>/`:** resolved `config.yaml`, `environment.json` (exact package versions), `params.json` (total / trainable / embedding / non-embedding), `metrics.json` (per-seed history), `dev_predictions.jsonl`, and the best-validation `model/`.

## Layout

```
configs/    baseline_deberta_v3_base.yaml · student_deberta_v3_xsmall.yaml · distill_deberta_v3_xsmall.yaml
scripts/    prepare_data.py · make_figures.py · export_predictions.py
report/     report.tex/.pdf · figures/ · results.json · error_annotation.csv
submission/ model_card.tex/.pdf · dev_predictions.csv/.jsonl
docs/       seminar.html (talk walkthrough)
src/rte/    data · config · device · seed · params · modeling · baselines · train · distill · predict · prune · analysis · recover
notebooks/  colab_teacher.ipynb · colab_student.ipynb · colab_sweep_a_distill.ipynb · colab_sweep_b_size.ipynb
artifacts/  teacher/teacher_logits_train.npy (OOF teacher soft labels, so distillation needs no teacher rerun)
tests/      test_data.py · test_modeling.py · test_recover.py
```

## Sanity baselines (dev)

| Method | Dev acc |
|---|---|
| Majority class | 52.7 |
| Word-overlap threshold (tuned on train) | 57.4 |
| TF-IDF + logistic regression | 57.0 |

## Rules compliance

- No external NLI data.
- No pre-fine-tuned RTE/NLI checkpoints. Only the pretrained `microsoft/deberta-v3-*` encoders are used.
- The GLUE RTE validation split is used for evaluation only.
