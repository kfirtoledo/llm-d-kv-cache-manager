"""Plot CPU + Storage offloading: HMA vs no-HMA for 20b and 120b at concurrent=10."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# (model, tier, hma) -> (last_10_s, throughput_GBs)
DATA = {
    ("20b", "cpu", "no-HMA"):  (0.750, 3.94),
    ("20b", "cpu", "HMA"):     (0.400, 7.39),
    ("20b", "storage", "no-HMA"): (1.330, 2.20),
    ("20b", "storage", "HMA"):    (0.750, 3.93),
    ("120b", "cpu", "no-HMA"):  (1.050, 4.19),
    ("120b", "cpu", "HMA"):     (0.310, 14.35),
    ("120b", "storage", "no-HMA"): (1.900, 2.32),
    ("120b", "storage", "HMA"):    (1.010, 4.35),
}

CATEGORIES = [
    ("20b", "cpu"),
    ("20b", "storage"),
    ("120b", "cpu"),
    ("120b", "storage"),
]
MODEL_NAMES = {"20b": "GPT-OSS-20B", "120b": "GPT-OSS-120B"}
TIER_NAMES = {"cpu": "CPU", "storage": "Storage"}
LABELS = [f"{MODEL_NAMES[m]}\n{TIER_NAMES[t]}" for m, t in CATEGORIES]


def make_plot(metric_idx: int, ylabel: str, title: str, fname: str, fmt: str = "{:.2f}",
              higher_better: bool = False):
    no_hma = [DATA[(m, t, "no-HMA")][metric_idx] for m, t in CATEGORIES]
    hma = [DATA[(m, t, "HMA")][metric_idx] for m, t in CATEGORIES]
    x = np.arange(len(CATEGORIES))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 6))
    bars_n = ax.bar(x - w / 2, no_hma, w, label="no-HMA", color="#d97a5b")
    bars_h = ax.bar(x + w / 2, hma, w, label="HMA", color="#5b8fd9")
    for bars in (bars_n, bars_h):
        for b in bars:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                    ha="center", va="bottom", fontsize=10)
    # Improvement of HMA vs no-HMA, written in white inside each blue bar.
    for b, no, ye in zip(bars_h, no_hma, hma):
        imp = ye / no if higher_better else no / ye
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() / 2,
                f"{imp:.2f}×", ha="center", va="center", fontsize=11,
                fontweight="bold", color="white")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    plt.setp(ax.get_yticklabels(), fontweight="bold")
    ax.set_title(title, fontsize=13)
    leg = ax.legend(fontsize=11)
    for txt in leg.get_texts():
        txt.set_fontweight("bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, max(no_hma + hma) * 1.20)
    plt.tight_layout()
    out = Path(__file__).parent / fname
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def make_speedup_plot():
    speedup = []
    for m, t in CATEGORIES:
        s_no = DATA[(m, t, "no-HMA")][0]
        s_h = DATA[(m, t, "HMA")][0]
        speedup.append(s_no / s_h)
    x = np.arange(len(CATEGORIES))
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, speedup, color="#5b8fd9", width=0.55)
    for b in bars:
        v = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}×",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=11, fontweight="bold")
    ax.set_ylabel("HMA speedup vs no-HMA (×)", fontsize=12, fontweight="bold")
    plt.setp(ax.get_yticklabels(), fontweight="bold")
    ax.set_title("HMA Speedup over no-HMA (last_10 latency ratio)\nconcurrent=10, 128k tokens, block=256", fontsize=13)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, max(speedup) * 1.25)
    plt.tight_layout()
    out = Path(__file__).parent / "speedup.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    make_plot(
        metric_idx=0,
        ylabel="Latency [s]",
        title="CPU + Storage offloading — last_10 latency, HMA vs no-HMA\nconcurrent=10, 128k tokens, block=256",
        fname="latency.png",
        fmt="{:.3f}",
    )
    make_plot(
        metric_idx=1,
        ylabel="Throughput (GB/s, higher is better)",
        title="CPU + Storage offloading — throughput, HMA vs no-HMA\nconcurrent=10, 128k tokens, block=256",
        fname="throughput.png",
        fmt="{:.2f}",
        higher_better=True,
    )
    make_speedup_plot()
