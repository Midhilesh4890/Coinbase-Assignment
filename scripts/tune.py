from pathlib import Path

from models.tuning import grid_search, nested_cv
from preprocessing.data import load_dataset


def _markdown_table(df):
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        values = []
        for value in row.tolist():
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main():
    data = load_dataset()
    results = grid_search(data)
    nested = nested_cv(data)
    print(results)
    print(nested)
    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "tuning.md"
    report = ["# Tuning", "", _markdown_table(results), "", "## Nested cross-validation", "", str(nested)]
    report_path.write_text("\n".join(report), encoding="utf-8")


main()
