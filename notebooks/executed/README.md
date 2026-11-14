# Executed Colab notebooks (with cell outputs)

These are the notebooks exactly as they ran on Colab (Tesla T4), with every cell output kept:
training logs, per-epoch validation accuracy, per-seed results and timings. They are a record,
not something to re-run: the trained models ship separately, and inference needs no training.
Only change: the code repository URL is anonymised (`github.com/ANONYMIZED/...`).

| Notebook | What it ran | Key result in the outputs |
|---|---|---|
| `1_teacher_baseline_and_oof_logits.ipynb` | DeBERTa-v3-base, 5 seeds; then 5-fold out-of-fold teacher logits | dev 84.8 ± 1.1; OOF accuracy 82.85% |
| `1b_teacher_oof_logits_rerun.ipynb` | Out-of-fold teacher logits only (repeat run) | OOF 82.85%, logits byte-identical to the first run |
| `2_students_plain_and_kd.ipynb` | xsmall plain fine-tuning and plain KD, 5 seeds each | 77.7 ± 2.1 and 76.8 ± 1.6 |
| `3_sweep_a_kd_variants_and_submission_rerun.ipynb` | masked KD and KD α 0.8 / T 4, 5 seeds each; last cell retrains masked KD seed 21, prunes it, exports predictions | 78.3 ± 1.4, 77.3 ± 0.8; seed 21 val 0.8266 / dev 0.7798, pruned 45.5M dev 0.7834 |
| `4_sweep_b_size.ipynb` | xsmall with rank-128 factorised embedding; DeBERTa-v3-small, 5 seeds each | 51.0 ± 5.0 and 76.5 ± 1.4 |

The clean, runnable versions of the same notebooks are one folder up (`notebooks/*.ipynb`).
