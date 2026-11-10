from rte.recover import parse_log

LOG = """run baseline | model microsoft/deberta-v3-base | device cuda
  seed 13 epoch  1 loss 0.6931 val_acc 0.5000 (40s)
  seed 13 epoch  1 loss 0.6900 val_acc 0.5500 (40s)
  seed 13 epoch  2 loss 0.5000 val_acc 0.8000 (40s)
  seed 13 epoch  3 loss 0.3000 val_acc 0.7800 (40s)
seed 13: val 0.8000 dev 0.8123
  seed 21 epoch  1 loss 0.6000 val_acc 0.7000 (40s)
seed 21: val 0.7000 dev 0.7509
"""


def test_parse_log_reads_seeds_and_restarted_history():
    s13, s21 = parse_log(LOG)
    assert (s13["seed"], s13["best_val_acc"], s13["dev_acc"], s13["best_epoch"]) == (
        13,
        0.8,
        0.8123,
        2,
    )
    assert [h["epoch"] for h in s13["history"]] == [1, 2, 3]  # restart kept the last attempt
    assert s13["history"][0]["val_acc"] == 0.55
    assert (s21["seed"], s21["dev_acc"]) == (21, 0.7509)
