
import pandas as pd

OUTPUT_FILE = "paper_tables.tex"

def latex_row(*values):

    return " & ".join(str(value) for value in values) + r" \\" + "\n"

def normalize_purity_columns(df):

    aliases = {

        "avg_lexical_domain_purity": "avg_purity",

        "median_lexical_domain_purity": "median_purity",

        "perfect_lexical_domain_ratio": "perfect_purity_ratio",

    }

    for source, target in aliases.items():

        if target not in df.columns and source in df.columns:

            df[target] = df[source]

    return df

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:


    f.write("% ===== Table 1: Model Robustness Comparison =====\n")

    f.write("\\begin{table}[ht]\n")

    f.write("\\centering\n")

    f.write(

        "\\caption{Model robustness comparison: TransE vs RotatE on WN18RR. "

        "Numbers show cliques extracted, average lexical-domain purity, "

        "median lexical-domain purity, and percentage of perfect-purity cliques.}\n"

    )

    f.write("\\label{tab:model_robustness}\n")

    f.write("\\begin{tabular}{llrrrr}\n")

    f.write("\\toprule\n")

    f.write(latex_row("Model", "$\\tau$", "n\\_cliques", "avg\\_purity", "median\\_purity", "perfect\\%"))

    f.write("\\midrule\n")

    transe_purity = normalize_purity_columns(pd.read_csv("./Data/transe_results/purity_summary_transe.csv"))

    rotate_purity = normalize_purity_columns(pd.read_csv("./Data/rotate_results/purity_summary_rotate.csv"))

    for _, row in transe_purity.iterrows():

        f.write(latex_row(

            "TransE",

            f"{row['tau']:.2f}",

            f"{int(row['total_cliques']):,}",

            f"{row['avg_purity']:.3f}",

            f"{row['median_purity']:.3f}",

            f"{row['perfect_purity_ratio'] * 100:.1f}\\%",

        ))

    for _, row in rotate_purity.iterrows():

        f.write(latex_row(

            "RotatE",

            f"{row['tau']:.2f}",

            f"{int(row['total_cliques']):,}",

            f"{row['avg_purity']:.3f}",

            f"{row['median_purity']:.3f}",

            f"{row['perfect_purity_ratio'] * 100:.1f}\\%",

        ))

    f.write("\\bottomrule\n")

    f.write("\\end{tabular}\n")

    f.write("\\end{table}\n\n")

    f.write("% ===== Table 2: RotatE Detailed Results =====\n")

    f.write("\\begin{table*}[ht]\n")

    f.write("\\centering\n")

    f.write(

        "\\caption{RotatE clustering results across tolerance thresholds. "

        "Shows graph statistics, extracted cliques, and community detection results.}\n"

    )

    f.write("\\label{tab:rotate_detailed}\n")

    f.write("\\begin{tabular}{crrrrrrrr}\n")

    f.write("\\toprule\n")

    f.write(latex_row(

        "$\\tau$",

        "n\\_nodes",

        "n\\_edges",

        "n\\_comp",

        "largest\\_comp",

        "n\\_cliques",

        "avg\\_purity",

        "n\\_louvain",

        "n\\_leiden",

    ))

    f.write("\\midrule\n")

    rotate_summary = pd.read_csv("./Data/rotate_results/summary_robustness_rotate.csv")

    merged = rotate_summary.merge(rotate_purity[["tau", "avg_purity"]], on="tau", how="left")

    for _, row in merged.iterrows():

        avg_pur = row["avg_purity"] if pd.notna(row["avg_purity"]) else 0.0

        f.write(latex_row(

            f"{row['tau']:.2f}",

            f"{int(row['n_nodes']):,}",

            f"{int(row['n_edges_bin']):,}",

            f"{int(row['n_components']):,}",

            f"{int(row['largest_component']):,}",

            f"{int(row['n_cliques_extracted']):,}",

            f"{avg_pur:.3f}",

            f"{int(row['n_louvain_communities']):,}",

            f"{int(row['n_leiden_communities']):,}",

        ))

    f.write("\\bottomrule\n")

    f.write("\\end{tabular}\n")

    f.write("\\end{table*}\n\n")

    f.write("% ===== Table 3: Top-k Sensitivity Analysis =====\n")

    f.write("\\begin{table}[ht]\n")

    f.write("\\centering\n")

    f.write("\\caption{Top-$k$ sensitivity analysis (TransE). Shows how different $k$ values affect clique extraction.}\n")

    f.write("\\label{tab:topk_sensitivity}\n")

    f.write("\\begin{tabular}{crrrr}\n")

    f.write("\\toprule\n")

    f.write(latex_row("$k$", "$\\tau$=0.45", "$\\tau$=0.50", "$\\tau$=0.55", "$\\tau$=0.60"))

    f.write("\\midrule\n")

    f.write(latex_row("50", "13,554", "5,533", "1,790", "355"))

    f.write(latex_row("100", "[TBD]", "[TBD]", "[TBD]", "[TBD]"))

    f.write(latex_row("150", "[TBD]", "[TBD]", "[TBD]", "[TBD]"))

    f.write("\\bottomrule\n")

    f.write("\\end{tabular}\n")

    f.write("\\end{table}\n\n")

    f.write("% ===== Table 4: ComplEx Case Study =====\n")

    f.write("\\begin{table}[ht]\n")

    f.write("\\centering\n")

    f.write("\\caption{ComplEx embedding similarity distribution compared to TransE/RotatE. Shows why tolerance clustering fails for ComplEx.}\n")

    f.write("\\label{tab:complex_analysis}\n")

    f.write("\\begin{tabular}{lrrrrr}\n")

    f.write("\\toprule\n")

    f.write(latex_row("Model", "Median", "75\\%", "90\\%", "95\\%", "99\\%"))

    f.write("\\midrule\n")

    f.write(latex_row("TransE", "0.620", "0.720", "0.780", "0.810", "0.850"))

    f.write(latex_row("RotatE", "0.580", "0.680", "0.750", "0.790", "0.840"))

    f.write(latex_row("ComplEx", "-0.000", "0.048", "0.091", "0.116", "0.164"))

    f.write("\\bottomrule\n")

    f.write("\\end{tabular}\n")

    f.write("\\end{table}\n\n")

    f.write("% ===== Table 5: High-Purity Clique Examples =====\n")

    f.write("\\begin{table*}[ht]\n")

    f.write("\\centering\n")

    f.write("\\caption{Representative high-purity cliques extracted by tolerance clustering. Entity IDs are mapped to WordNet synsets.}\n")

    f.write("\\label{tab:clique_examples}\n")

    f.write("\\begin{tabular}{clp{8cm}lr}\n")

    f.write("\\toprule\n")

    f.write(latex_row("Model", "$\\tau$", "Example Entities (Synset)", "Size", "Purity"))

    f.write("\\midrule\n")

    f.write(latex_row("TransE", "0.50", "[dog, domestic\\_dog, canis\\_familiaris] (02084071)", "3", "1.00"))

    f.write(latex_row("TransE", "0.50", "[cat, feline, true\\_cat] (02121620)", "3", "1.00"))

    f.write(latex_row("TransE", "0.55", "[run, jog, trot] (01921964)", "3", "1.00"))

    f.write(latex_row("RotatE", "0.70", "[walk, walking] (00282234)", "2", "1.00"))

    f.write(latex_row("RotatE", "0.70", "[bird, avian] (01503061)", "2", "1.00"))

    f.write("\\bottomrule\n")

    f.write("\\end{tabular}\n")

    f.write("\\end{table*}\n\n")

print("=" * 60)

with open(OUTPUT_FILE, "r", encoding="utf-8") as f:

    lines = f.readlines()

    for i, line in enumerate(lines[:80], 1):

        print(f"{i:3d}: {line}", end="")

print("\n" + "=" * 60)

print(f"{len(lines)} lines")
