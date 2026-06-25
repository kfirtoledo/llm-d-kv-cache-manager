"""Plot CPU + Storage offloading averages with error bars (10 reps per config)."""

import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PATTERN = re.compile(
    r"cold:\s+([\d.]+)s\s+hot_avg\([^)]+\):\s+([\d.]+)s\s+last_10_avg:\s+([\d.]+)s\s+throughput:\s+([\d.]+)\s+GB/s"
)
ROOT = Path(__file__).parent

CATEGORIES = [
    ("20b", "cpu"),
    ("120b", "cpu"),
    ("20b", "storage"),
    ("120b", "storage"),
]
MODEL_NAMES = {"20b": "GPT-OSS-20B", "120b": "GPT-OSS-120B"}
TIER_NAMES = {"cpu": "CPU", "storage": "Storage"}
LABELS = [f"{MODEL_NAMES[m]}\n{TIER_NAMES[t]}" for m, t in CATEGORIES]
# Visual separator x-position between cpu group and storage group.
GROUP_SEPARATOR_X = 1.5


def collect():
    data = {}
    for cfg in ROOT.iterdir():
        if not cfg.is_dir():
            continue
        l10 = []
        tput = []
        for log in sorted(cfg.glob("run*.log")):
            text = log.read_text(errors="ignore")
            m = PATTERN.search(text)
            if not m:
                continue
            _, _, x, y = m.groups()
            l10.append(float(x))
            tput.append(float(y))
        if not l10:
            continue
        data[cfg.name] = {"l10": l10, "tput": tput}
    return data


def get(data, model, tier, hma):
    suffix = "hma" if hma else "nohma"
    key = f"gpt-oss-{model}_{suffix}_{tier}_c10"
    return data[key]


def make_bar(data, metric, ylabel, title, fname, fmt="{:.3f}", higher_better=False):
    no_hma_mean = []
    hma_mean = []
    no_hma_err = []
    hma_err = []
    for m, t in CATEGORIES:
        no = get(data, m, t, False)[metric]
        ye = get(data, m, t, True)[metric]
        no_hma_mean.append(statistics.mean(no))
        hma_mean.append(statistics.mean(ye))
        no_hma_err.append(statistics.pstdev(no))
        hma_err.append(statistics.pstdev(ye))
    x = np.arange(len(CATEGORIES))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 6))
    b_n = ax.bar(x - w / 2, no_hma_mean, w, yerr=no_hma_err, capsize=4,
                 label="no-HMA", color="#d97a5b")
    b_h = ax.bar(x + w / 2, hma_mean, w, yerr=hma_err, capsize=4,
                 label="HMA", color="#5b8fd9")
    for bars, vals in ((b_n, no_hma_mean), (b_h, hma_mean)):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                    ha="center", va="bottom", fontsize=10)
    # Improvement of HMA vs no-HMA, written in white inside each blue bar.
    for b, no, ye in zip(b_h, no_hma_mean, hma_mean):
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
    ax.axvline(GROUP_SEPARATOR_X, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ymax = max(no_hma_mean + hma_mean) * 1.20
    ax.set_ylim(0, ymax)
    ax.text(0.5, ymax * 0.97, "CPU offloading", ha="center", fontsize=12,
            fontweight="bold", color="#666")
    ax.text(2.5, ymax * 0.97, "Storage offloading", ha="center", fontsize=12,
            fontweight="bold", color="#666")
    plt.tight_layout()
    out = ROOT / fname
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def make_speedup(data):
    speedup = []
    for m, t in CATEGORIES:
        no = statistics.mean(get(data, m, t, False)["l10"])
        ye = statistics.mean(get(data, m, t, True)["l10"])
        speedup.append(no / ye)
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
    ax.set_title("HMA Speedup (10-rep mean last_10 ratio)\nconcurrent=10, 128k tokens, block=256", fontsize=13)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.axvline(GROUP_SEPARATOR_X, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ymax = max(speedup) * 1.25
    ax.set_ylim(0, ymax)
    ax.text(0.5, ymax * 0.97, "CPU offloading", ha="center", fontsize=12,
            fontweight="bold", color="#666")
    ax.text(2.5, ymax * 0.97, "Storage offloading", ha="center", fontsize=12,
            fontweight="bold", color="#666")
    plt.tight_layout()
    out = ROOT / "speedup.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    data = collect()
    make_bar(
        data, "l10",
        ylabel="Batch Latency [s]",
        #title="CPU + Storage — 10-rep mean ± stdev\nconcurrent=10, 128k tokens, block=256",
        fname="latency.png",
        fmt="{:.3f}",
    )
    make_bar(
        data, "tput",
        ylabel="Throughput (GB/s, higher better)",
        title="CPU + Storage — 10-rep mean ± stdev\nconcurrent=10, 128k tokens, block=256",
        fname="throughput.png",
        fmt="{:.2f}",
        higher_better=True,
    )
    make_speedup(data)
