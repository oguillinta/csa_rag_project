import pandas as pd

def csv_to_documents(df: pd.DataFrame, source_name: str):
    """Turn each row into a simple pipe-separated string, plus metadata and ids."""
    docs, metas, ids = [], [], []
    df = df.fillna("")
    for i, row in df.iterrows():
        parts = []
        for col in df.columns:
            val = str(row[col])
            parts.append(f"{col}: {val}")
        text = " | ".join(parts)

        text = text.lower()

        docs.append(text)
        metas.append({"source": source_name, "row_index": int(i)})
        ids.append(f"{source_name}-{i}")
    return docs, metas, ids

