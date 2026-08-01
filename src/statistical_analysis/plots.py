import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt

from preprocessing.dedup import template_key


def plot_class_distribution(df, path):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = df["label"].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots()
    counts.plot.barh(ax=ax)
    ax.invert_yaxis()
    ax.set_title("Class distribution")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_template_frequency(df, path):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = df["text"].map(template_key).value_counts()
    fig, ax = plt.subplots()
    ax.hist(counts.values)
    ax.set_xlabel("rows per template")
    ax.set_ylabel("number of templates")
    ax.set_title("Template duplication")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
