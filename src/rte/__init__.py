"""RTE challenge: DeBERTa-v3 baseline/teacher and distilled student."""

LABELS = ("entailment", "not_entailment")
LABEL2ID = {name: i for i, name in enumerate(LABELS)}
ID2LABEL = dict(enumerate(LABELS))
