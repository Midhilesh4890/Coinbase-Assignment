import pandas as pd
from pathlib import Path

from models.diagnostics import learning_curve, mcnemar, novel_message_check
from preprocessing.data import load_dataset


def table_from_df(df):
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def table_from_dict(data):
    return table_from_df(pd.DataFrame([data]))


def main():
    data = load_dataset()
    curve = learning_curve(data, "tfidf_logreg_balanced")
    test = mcnemar(data, "tfidf_logreg", "tfidf_logreg_balanced")
    novel = novel_message_check("tfidf_logreg_balanced")
    report_path = Path("reports/diagnostics.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# Learning curve\n\n"
        + table_from_df(curve)
        + "\n\n# McNemar test\n\n"
        + table_from_dict(test)
        + "\n\n# Novel message check\n\n"
        + table_from_df(novel),
        encoding="utf-8",
    )
    print(curve)
    print(test)
    print(novel)


main()
