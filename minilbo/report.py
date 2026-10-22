"""Chart for the LBO model."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .model import LBOInputs, run_lbo, sensitivity_entry_exit


def plot_overview(i: LBOInputs, path: str | Path) -> None:
    r = run_lbo(i)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    # Sources & uses
    ax = axes[0, 0]
    su = r.sources_uses
    for k, side in enumerate(["Source", "Use"]):
        part = su[su["side"] == side]
        bottom = 0
        for (_, row), colr in zip(part.iterrows(), plt.cm.Blues(np.linspace(0.85, 0.3, len(part)))):
            ax.bar(k, row["amount"], bottom=bottom, color=colr, edgecolor="white")
            if row["amount"] > 40:
                ax.text(k, bottom + row["amount"] / 2, f"{row['item']}\n{row['amount']:,.0f}", ha="center", va="center",
                        fontsize=8, color="white" if row["amount"] > 150 else "black")
            bottom += row["amount"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Sources", "Uses"]); ax.set_title("Sources and uses (EUR million)")

    # Debt paydown
    ax = axes[0, 1]
    p = r.projections
    ax.bar(p.index, p["senior_debt"], label="Senior", color="#1f4e79")
    ax.bar(p.index, p["subordinated_debt"], bottom=p["senior_debt"], label="Subordinated", color="#8fb4d9")
    ax.set_title("Debt paydown and leverage"); ax.set_xlabel("Year"); ax.set_ylabel("Debt (EUR million)")
    ax2 = ax.twinx()
    ax2.plot(p.index, p["net_debt_to_ebitda"], color="#c0392b", marker="o", label="Net debt / EBITDA")
    ax2.set_ylabel("x EBITDA"); ax2.set_ylim(0, max(5, p["net_debt_to_ebitda"].max() + 0.5))
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)

    # Value creation bridge
    ax = axes[1, 0]
    b = r.bridge
    labels = ["Sponsor\nequity"] + [n.replace(" ", "\n", 1) for n in b.index] + ["Exit\nequity"]
    start = r.sponsor_equity
    cum = start
    ax.bar(0, start, color="#7f8c8d")
    ax.text(0, start, f"{start:,.0f}", ha="center", va="bottom", fontsize=8)
    for k, (name, v) in enumerate(b.items(), start=1):
        ax.bar(k, v, bottom=cum if v >= 0 else cum + v, color="#4f9d69" if v >= 0 else "#c0392b")
        ax.text(k, max(cum, cum + v), f"{v:+,.0f}", ha="center", va="bottom", fontsize=8)
        cum += v
    ax.bar(len(b) + 1, r.exit_equity, color="#1f4e79")
    ax.text(len(b) + 1, r.exit_equity, f"{r.exit_equity:,.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(f"Value creation bridge (MOIC {r.moic:.2f}x, IRR {r.irr:.1%})")

    # IRR sensitivity
    ax = axes[1, 1]
    entries = [i.entry_multiple + d for d in (-1.0, -0.5, 0.0, 0.5, 1.0)]
    exits = [i.exit_multiple + d for d in (-1.0, 0.0, 1.0, 2.0)]
    s = sensitivity_entry_exit(i, entries, exits)
    im = ax.imshow(s.to_numpy(), cmap="RdYlGn", aspect="auto", vmin=0.05, vmax=0.30)
    ax.set_xticks(range(len(exits))); ax.set_xticklabels([f"{x:.1f}x" for x in exits])
    ax.set_yticks(range(len(entries))); ax.set_yticklabels([f"{x:.1f}x" for x in entries])
    for a in range(len(entries)):
        for c in range(len(exits)):
            ax.text(c, a, f"{s.iat[a, c]:.1%}", ha="center", va="center", fontsize=9)
    ax.set_xlabel("Exit EV/EBITDA"); ax.set_ylabel("Entry EV/EBITDA"); ax.set_title("Sponsor IRR sensitivity")

    for k, a in enumerate(axes.flat):
        a.spines["top"].set_visible(False)
        if k != 3: a.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
