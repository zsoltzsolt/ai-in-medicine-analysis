import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.ndimage import gaussian_filter1d


def plot_yearly_bar_chart(
    years, 
    counts, 
    title="",
    xlabel="",
    ylabel="",
    log_scale=False, 
    percentage=False,
    save_file=None
    ):
    years, counts = np.asarray(years), np.asarray(counts, dtype=float)

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.bar(years, counts, width=0.75, color="#dbeafe", edgecolor="#2563eb", linewidth=1.2, zorder=2)

    trend = gaussian_filter1d(counts, sigma=1.4)

    ax.plot(years, trend, color="#1d4ed8", linewidth=2.4, zorder=4)

    ax.fill_between(years, trend, color="#1d4ed8", alpha=0.08, zorder=1)

    ax.set_title(title, fontsize=16, fontweight="bold", pad=18)

    ax.set_xlabel(xlabel, fontsize=15, labelpad=10)
    ax.set_ylabel(f"{ylabel} (log scale)" if log_scale else ylabel, fontsize=15, labelpad=10)
    plt.tick_params(axis='both', labelsize=12)

    if log_scale:
        ax.set_yscale("log")

    if percentage:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    else:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K" if x >= 1000 else f"{x:.0f}"))

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    # ax.legend(frameon=False)

    plt.tight_layout()

    if save_file:
        plt.savefig(save_file, dpi=300, bbox_inches="tight")
    
    plt.show()