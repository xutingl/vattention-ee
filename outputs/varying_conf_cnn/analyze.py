import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

# Base directory for experiment results
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Color palette for policies
POLICY_COLORS = {
    "eager": "#1f77b4",
    "lazy": "#ff7f0e",
    "latency-only": "#2ca02c",
    "median": "#d62728",
    "rebatching": "#9467bd",
}


def load_csv(model: str, batch_size: int, ee_config: str, policy: str) -> pd.DataFrame:
    """Load a single experiment CSV file."""
    pattern = os.path.join(BASE_DIR, model, f"batch_{batch_size}", policy, f"req_*_batch_{batch_size}_{ee_config}_{policy}_copy.csv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        print(f"Warning: no file matching: {pattern}")
        return None
    return pd.read_csv(matches[0])


# Derived columns: name -> lambda(df) computing the value from existing columns
DERIVED_COLUMNS = {
    "decode_throughput": lambda df: df["num_output_tokens"] / df["decode_time"],
}


def get_column_value(df: pd.DataFrame, col: str):
    """Get first-row value for a column, supporting both raw and derived columns."""
    if col in df.columns:
        return df[col].iloc[0]
    if col in DERIVED_COLUMNS:
        return DERIVED_COLUMNS[col](df).iloc[0]
    return None


def plot_line_graph(
    y: str,
    x: str,
    ee_configs: List[str],
    batch_size: int,
    policies: List[str],
    model: str = "llama-2-13b",
    save: bool = True,
):
    """Plot a line graph with ee_configs on the x-axis progression.

    Args:
        y: Column name for the y-axis (value from first row of CSV).
        x: Column name for the x-axis (value from first row of CSV).
        ee_configs: List of EE config strings, e.g. ["layer_20_conf_0.0", "layer_20_conf_0.5"].
        batch_size: Batch size (e.g. 8).
        policies: List of policy names, e.g. ["eager", "lazy", "rebatching"].
        model: Model name subfolder (default "llama-2-13b").
        save: Whether to save the plot to a file.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw Non-EE baseline if "off" directory exists
    off_dir = os.path.join(BASE_DIR, model, f"batch_{batch_size}", "off")
    if os.path.isdir(off_dir):
        off_files = sorted(os.listdir(off_dir))
        if off_files:
            off_df = pd.read_csv(os.path.join(off_dir, off_files[0]))
            off_df.columns = off_df.columns.str.strip()
            off_throughput = get_column_value(off_df, y)
            if off_throughput is not None:
                ax.axhline(y=off_throughput, color="black", linestyle="--", linewidth=1.5, label="Non-EE")

    for policy in policies:
        x_vals = []
        y_vals = []
        for ee_config in ee_configs:
            df = load_csv(model, batch_size, ee_config, policy)
            if df is None:
                continue
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            x_val = get_column_value(df, x)
            y_val = get_column_value(df, y)
            if x_val is None or y_val is None:
                print(f"Warning: column '{x}' or '{y}' not found in CSV for {policy}/{ee_config}")
                continue
            # Skip zero-valued conf_score_ee points on x-axis
            if "conf_score_ee" in x and x_val == 0:
                continue
            x_vals.append(x_val)
            y_vals.append(y_val)

        color = POLICY_COLORS.get(policy, None)
        ax.plot(x_vals, y_vals, marker="o", label=policy, color=color)

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{y} vs {x} (batch_size={batch_size}, model={model})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        out_name = f"{y}_vs_{x}_batch{batch_size}_{model}.png"
        out_path = os.path.join(BASE_DIR, out_name)
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot to {out_path}")

    plt.show()

qwen_ee_configs = [
    "layer_20_conf_0.01",
    "layer_20_conf_0.02",
    "layer_20_conf_0.05",
    "layer_20_conf_0.1",
    "layer_20_conf_0.25",
    "layer_20_conf_0.5",
]

llama_ee_configs = [

    "layer_20_conf_0.03",
    "layer_20_conf_0.05",
    "layer_20_conf_0.1",
    "layer_20_conf_0.2",
    "layer_20_conf_0.5",
]

llama_70b_configs = [
    "layer_40_conf_0.005",
    "layer_40_conf_0.01",
    "layer_40_conf_0.02",
    "layer_40_conf_0.03",
    "layer_40_conf_0.04",
    "layer_40_conf_0.05",
]


if __name__ == "__main__":

    policies = ["eager", "lazy", "median", "rebatching", "latency-only"]

    llama13b = "llama-2-13b"
    qwen = "qwen-14b-chat"

    # plot_line_graph(
    #     y="decode_throughput",
    #     x="avg_conf_score_ee",
    #     ee_configs=qwen_ee_configs,
    #     batch_size=8,
    #     policies=policies,
    #     model=qwen,
    # )

    # plot_line_graph(
    #     y="decode_throughput",
    #     x="avg_conf_score_ee",
    #     ee_configs=llama_ee_configs,
    #     batch_size=8,
    #     policies=policies,
    #     model=llama13b,
    # )

    plot_line_graph(
        y="decode_throughput",
        x="avg_conf_score_ee",
        ee_configs=llama_70b_configs,
        batch_size=8,
        policies=policies,
        model="llama-2-70b",
    )
