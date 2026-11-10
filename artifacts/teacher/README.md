# Teacher soft labels

`teacher_logits_train.npy` (shape 2490 x 2, float32) holds the out-of-fold logits of the
DeBERTa-v3-base teacher on `data/train.jsonl`, row-aligned with it. Each row was scored by a
teacher fold model that never saw it (5 folds grouped by premise, seed 13). OOF accuracy: 82.85%.

Reproduce: `python -m rte.train --config configs/baseline_deberta_v3_base.yaml --oof`, which writes
`outputs/baseline_deberta_v3_base_oof/teacher_logits_train.npy`. Two independent Colab T4 runs
produced byte-identical files. The file is committed so the distillation runs need no teacher
retraining (about 3.5 GPU-hours) and no upload.
