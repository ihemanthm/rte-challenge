"""Report figures and results table from outputs/*/metrics.json and the pruning curve.

python scripts/make_figures.py --pruned outputs/student_deberta_v3_xsmall_pruned --out report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = [  # run dir, short label, role
    ("baseline_deberta_v3_base", "teacher\nv3-base", "teacher"),
    ("student_deberta_v3_small", "v3-small", "other"),
    ("student_deberta_v3_xsmall", "v3-xsmall", "student"),
    ("distill_deberta_v3_xsmall", "xsmall + KD", "student"),
    ("distill_masked_deberta_v3_xsmall", "xsmall + masked KD", "student"),
    ("distill_a08_t4_deberta_v3_xsmall", "xsmall + KD a0.8 T4", "student"),
    ("student_deberta_v3_xsmall_emb128", "xsmall, rank-128 emb", "failed"),
]
# the four xsmall variants share one x position: stack their labels to the right with leader lines
OFFSETS = {
    "student_deberta_v3_xsmall": (18, 28),
    "distill_masked_deberta_v3_xsmall": (18, 18),
    "distill_a08_t4_deberta_v3_xsmall": (18, 8),
    "distill_deberta_v3_xsmall": (18, -2),
    "student_deberta_v3_small": (6, -14),
}
LEADER = {"arrowstyle": "-", "lw": 0.4, "color": "grey"}
COLORS = {"teacher": "#a86400", "student": "#3a47c9", "other": "#596173", "failed": "#b73b26"}


def load(outputs: Path) -> list[dict]:
    rows = []
    for run, label, role in RUNS:
        f = outputs / run / "metrics.json"
        if not f.exists():
            continue
        m = json.loads(f.read_text())
        seeds = m["seeds"]
        rows.append(
            {
                "run": run,
                "label": label,
                "role": role,
                "params": m["params"]["total"],
                "non_emb": m["params"]["non_embedding"],
                "dev_mean": m["dev_acc_mean"] * 100,
                "dev_std": m["dev_acc_std"] * 100,
                "val_mean": sum(s["best_val_acc"] for s in seeds) / len(seeds) * 100,
                "selected_seed": m.get("selected_seed"),
                "selected_dev": m.get("selected_seed_dev_acc", 0) * 100,
            }
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outputs", default="outputs")
    p.add_argument("--pruned", required=True, help="run dir written by rte.prune")
    p.add_argument("--out", default="report")
    args = p.parse_args()
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    rows = load(Path(args.outputs))
    prune = json.loads((Path(args.pruned) / "prune.json").read_text())

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for r in rows:
        a.errorbar(
            r["params"] / 1e6,
            r["dev_mean"],
            yerr=r["dev_std"],
            fmt="o",
            ms=5,
            color=COLORS[r["role"]],
            capsize=2,
        )
        dx, dy = OFFSETS.get(r["run"], (5, 4))
        a.annotate(
            r["label"],
            (r["params"] / 1e6, r["dev_mean"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=7,
            arrowprops={"arrowstyle": "-", "lw": 0.4, "color": "grey"}
            if r["run"] in OFFSETS
            else None,
        )
    a.scatter(
        [prune["params"]["total"] / 1e6],
        [prune["dev_acc_pruned"] * 100],
        marker="*",
        s=110,
        color="#1d7f55",
        zorder=3,
    )
    a.annotate(
        "submission (pruned,\nsingle ckpt)",
        (prune["params"]["total"] / 1e6, prune["dev_acc_pruned"] * 100),
        textcoords="offset points",
        xytext=(-38, -30),
        fontsize=7,
        color="#1d7f55",
    )
    a.axhline(52.7, ls=":", color="grey", lw=0.8)
    a.text(195, 53.5, "majority", fontsize=7, color="grey", ha="right")
    a.set_xlabel("total parameters (M)")
    a.set_ylabel("dev accuracy % (mean ± sd, 5 seeds)")
    a.set_xlim(0, 215)
    a.set_ylim(44, 93)
    a.set_title("(a) accuracy vs. size", fontsize=9)

    c = prune["curve_over_top_n"]
    b.plot(
        [x["total_params"] / 1e6 for x in c],
        [x["dev_acc"] * 100 for x in c],
        "o-",
        color="#3a47c9",
        ms=3,
    )
    b.scatter(
        [prune["params"]["total"] / 1e6],
        [prune["dev_acc_pruned"] * 100],
        marker="*",
        s=110,
        color="#1d7f55",
        zorder=3,
    )
    b.set_xlabel("total parameters (M)")
    b.set_ylabel("dev accuracy % (selected ckpt)")
    b.set_title("(b) vocabulary pruning (submitted ckpt)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "figures" / "acc_vs_params.pdf")
    fig.savefig(out / "figures" / "acc_vs_params.png", dpi=200)

    (out / "results.json").write_text(
        json.dumps(
            {
                "runs": rows,
                "pruned": {
                    k: prune[k]
                    for k in (
                        "top_n",
                        "kept_rows",
                        "dev_acc_full",
                        "dev_acc_pruned",
                        "dev_prediction_agreement",
                        "dev_coverage",
                        "params",
                    )
                },
            },
            indent=2,
        )
    )
    for r in rows:
        print(
            f"{r['label']:24s} {r['params'] / 1e6:6.1f}M dev {r['dev_mean']:.1f}±{r['dev_std']:.1f}"
            f" val {r['val_mean']:.1f}"
        )


if __name__ == "__main__":
    main()
