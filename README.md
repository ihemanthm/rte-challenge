# RTE Challenge: DeBERTa-v3 baseline and distilled student

Binary textual entailment (`entailment` / `not_entailment`) on RTE.

- **Baseline / teacher:** DeBERTa-v3-base, fine-tuned on Colab.
- **Submission:** DeBERTa-v3-xsmall, trained with knowledge distillation from that teacher.
- **Evaluation:** both models are scored on the GLUE RTE validation set (277 rows).

The aim is to keep most of the baseline's accuracy at a fraction of its parameter count.

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
make test        # unit tests (splits, padding, KD loss, param counting, configs)
make baselines   # majority / word-overlap / TF-IDF+LR sanity baselines
```

1. **Baseline and teacher (Colab GPU):** open `notebooks/colab_teacher.ipynb` and run all cells. It
   - trains DeBERTa-v3-base over 5 seeds → `outputs/baseline_deberta_v3_base/`
   - exports out-of-fold teacher logits → `outputs/baseline_deberta_v3_base_oof/teacher_logits_train.npy`

   Download `teacher_artifacts.zip` and unzip it in the project root.
2. **Student without distillation (ablation, local MPS):** `make train CONFIG=configs/student_deberta_v3_xsmall.yaml`
3. **Distilled student (submission):** `make distill CONFIG=configs/distill_deberta_v3_xsmall.yaml`
4. **Predictions and error analysis:**
   ```bash
   make predict RUN=outputs/distill_deberta_v3_xsmall
   make analyse RUN=outputs/distill_deberta_v3_xsmall
   ```

## Protocol (what keeps the numbers honest)

- **Dev is never used for selection.** Each seed early-stops on a 10% hold-out of train, grouped by premise. 99 premises appear with several hypotheses, so random splits would leak.
- **5 seeds per configuration, reported as mean ± std.** RTE fine-tuning is seed-sensitive. With n=277, one dev example is 0.36 points and the standard error is about ±2.4 points.
- **Teacher soft labels are out-of-fold.** Each training row is scored by a teacher that never saw it (5 folds, grouped by premise), which avoids overconfident in-sample targets.
- **KD loss:** `alpha * CE(y) + (1 - alpha) * T² * KL(teacher_T || student_T)`, with alpha=0.5 and T=2. The student and distill configs share every other hyperparameter, so the effect of distillation is isolated.
- **Every run writes `outputs/<run>/`:** resolved `config.yaml`, `environment.json` (exact package versions), `params.json` (total / trainable / embedding / non-embedding), `metrics.json` (per-seed history), `dev_predictions.jsonl`, and the best-validation `model/`.

## Layout

```
configs/    baseline_deberta_v3_base.yaml · student_deberta_v3_xsmall.yaml · distill_deberta_v3_xsmall.yaml
scripts/    prepare_data.py
src/rte/    data · config · device · seed · params · baselines · train · distill · predict · analysis
notebooks/  colab_teacher.ipynb
tests/      test_data.py
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
