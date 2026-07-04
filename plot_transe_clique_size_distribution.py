
import glob

import json

import os

from collections import Counter

import matplotlib.pyplot as plt

DATA_DIR = "./Data"

OUTPUT_DIRS = ["./figures", "./Data/figures"]

def load_clique_sizes():

    sizes = []

    pattern = os.path.join(DATA_DIR, "cliques_tau_*.json")

    for path in sorted(glob.glob(pattern)):

        with open(path, "r", encoding="utf-8") as handle:

            cliques = json.load(handle)

        sizes.extend(len(clique) for clique in cliques)

    return sizes

def plot_clique_size_distribution(sizes):

    counts = Counter(sizes)

    clique_sizes = sorted(counts)

    frequencies = [counts[size] for size in clique_sizes]

    fig, ax = plt.subplots(figsize=(8, 4.8))

    ax.bar(clique_sizes, frequencies, width=0.65, color="#2E86AB", edgecolor="black", alpha=0.85)

    ax.set_xlabel("Clique Size", fontsize=12)

    ax.set_ylabel("Count", fontsize=12)

    ax.set_xticks(clique_sizes)

    ax.grid(True, alpha=0.3)

    fig.tight_layout(pad=0.3)

    for output_dir in OUTPUT_DIRS:

        os.makedirs(output_dir, exist_ok=True)

        fig.savefig(os.path.join(output_dir, "clique_size_distribution.pdf"), dpi=300, bbox_inches="tight")

    plt.close(fig)

if __name__ == "__main__":

    clique_sizes = load_clique_sizes()

    if not clique_sizes:

        raise RuntimeError("No clique sizes found. Check Data/cliques_tau_*.json files.")

    plot_clique_size_distribution(clique_sizes)

    print("Saved clique_size_distribution.pdf")
