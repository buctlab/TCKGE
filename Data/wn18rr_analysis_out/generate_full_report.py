import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)


def load_all_louvain():
    files = glob.glob("louvain_tau_*.csv")
    dfs = []
    for f in files:
        tau = f.split("_")[-1].replace(".csv", "")
        df = pd.read_csv(f)
        df["tau"] = tau
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def load_all_leiden():
    files = glob.glob("leiden_tau_*.csv")
    dfs = []
    for f in files:
        tau = f.split("_")[-1].replace(".csv", "")
        df = pd.read_csv(f)
        df["tau"] = tau
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def load_cliques():
    files = glob.glob("cliques_tau_*.json")
    sizes = []
    taus = []
    for f in files:
        tau = f.split("_")[-1].replace(".json", "")
        with open(f, "r", encoding="utf-8") as fp:
            cliques = json.load(fp)
        for c in cliques:
            sizes.append(len(c))
            taus.append(tau)
    return pd.DataFrame({"tau": taus, "clique_size": sizes})


def plot_community_size_distribution(df, method="louvain"):
    plt.figure(figsize=(8, 4.2))
    sns.histplot(df["size"], bins=40, kde=True)
    plt.xlabel("Community Size")
    plt.ylabel("Count")
    out = f"{OUT_DIR}/{method}_community_size_distribution.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def plot_clique_size_distribution(df):
    plt.figure(figsize=(8, 4.8))
    sns.countplot(x="clique_size", hue="tau", data=df)
    plt.xlabel("Clique Size")
    plt.ylabel("Frequency")
    plt.tight_layout(pad=0.3)
    out = f"{OUT_DIR}/clique_size_distribution.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def plot_robustness():
    df = pd.read_csv("summary_robustness.csv")

    # Convert tau to numeric
    df["tau"] = pd.to_numeric(df["tau"])

    metrics = [
        ("n_components", "Number of Connected Components"),
        ("largest_component", "Size of Largest Component"),
        ("n_louvain_communities", "Number of Louvain Communities"),
        ("n_leiden_communities", "Number of Leiden Communities"),
        ("n_cliques_extracted", "Number of Maximal Cliques")
    ]

    outputs = []

    for col, title in metrics:
        plt.figure(figsize=(8,6))
        sns.lineplot(data=df, x="tau", y=col, marker="o")
        plt.xlabel("τ")
        plt.ylabel(title)
        plt.title(f"{title} vs τ")
        out = f"{OUT_DIR}/{col}_vs_tau.pdf"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        outputs.append(out)

    return outputs


def generate_conclusion_text():
    df = pd.read_csv("summary_robustness.csv")
    text = []

    text.append("This section summarizes the main experimental results using tolerance-relation-based clustering on WN18RR.\n")

    # Trend observations
    for tau in df["tau"].unique():
        sub = df[df["tau"] == tau].iloc[0]
        text.append(
            f"For τ={tau}: components={sub['n_components']}, "
            f"largest component={sub['largest_component']}, "
            f"Louvain communities={sub['n_louvain_communities']}, "
            f"Leiden communities={sub['n_leiden_communities']}, "
            f"cliques={sub['n_cliques_extracted']}."
        )

    text.append("\nOverall, decreasing τ increases graph connectivity, resulting in fewer components and larger community structures.\n")
    text.append("Louvain and Leiden clustering both show stable behavior across τ, while maximal cliques tend to vanish at higher τ levels.\n")

    return "\n".join(text)


def generate_latex_file(figures):
    tex = r"""
\documentclass[conference]{IEEEtran}
\usepackage{graphicx}
\usepackage{booktabs}
\begin{document}

\section{Experimental Analysis}

This section presents the experimental analysis for tolerance-relation-based clustering on the WN18RR knowledge graph.

\subsection{Clique Size Distribution}

Fig.~\ref{fig:clique} shows the maximal clique size distribution across different threshold~$\tau$.

\begin{figure}[h]
\centering
\includegraphics[width=0.45\textwidth]{""" + figures["clique"] + r"""}
\caption{Clique Size Distribution Across $\tau$.}
\label{fig:clique}
\end{figure}

\subsection{Community Size Distributions}

\begin{figure}[h]
\centering
\includegraphics[width=0.45\textwidth]{""" + figures["louvain"] + r"""}
\caption{Louvain Community Size Distribution}
\label{fig:louvain}
\end{figure}

\begin{figure}[h]
\centering
\includegraphics[width=0.45\textwidth]{""" + figures["leiden"] + r"""}
\caption{Leiden Community Size Distribution}
\label{fig:leiden}
\end{figure}

\subsection{Robustness Analysis}

The robustness analysis over various $\tau$ is shown in Fig.~\ref{fig:robustness}.

\begin{figure}[h]
\centering
\includegraphics[width=0.45\textwidth]{""" + figures["robustness"] + r"""}
\caption{Robustness Metrics vs $\tau$.}
\label{fig:robustness}
\end{figure}

\subsection{Discussion}

""" + figures["conclusion"] + r"""

\end{document}
"""

    with open("experiment_report.tex", "w", encoding="utf-8") as fp:
        fp.write(tex)

    print("LaTeX written to experiment_report.tex")


def main():
    print("Loading cluster data...")

    louvain = load_all_louvain()
    leiden = load_all_leiden()
    cliques = load_cliques()

    print("Generating plots...")

    louvain_fig = plot_community_size_distribution(louvain, "louvain")
    leiden_fig = plot_community_size_distribution(leiden, "leiden")
    clique_fig = plot_clique_size_distribution(cliques)
    robustness_figs = plot_robustness()

    print("Generating conclusion text...")
    conclusion = generate_conclusion_text()

    # Create a text file for conclusions
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(conclusion)

    figures = {
        "clique": clique_fig,
        "louvain": louvain_fig,
        "leiden": leiden_fig,
        "robustness": robustness_figs[0],  # representative
        "conclusion": conclusion
    }

    print("Generating LaTeX file...")
    generate_latex_file(figures)

    print("Done! All outputs generated.")


if __name__ == "__main__":
    main()
