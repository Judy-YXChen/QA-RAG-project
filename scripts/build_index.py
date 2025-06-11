# scripts/build_index.py
import pandas as pd, numpy as np, faiss, joblib, os
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from sklearn.cluster import KMeans

DATA = {
    "bge": dict(
        embed_name="BAAI/bge-base-zh",
        csv_emb="bge_embeddings.csv",
        csv_meta="kmeans_clustered_result.csv",
        out_dir="faiss_index_bge",
        kmeans_out="kmeans_bge.pkl",
    ),
    "bert": dict(
        embed_name="bert-base-chinese",
        csv_emb="bert_embeddings.csv",
        csv_meta="bert_kmeans_clustered_result.csv",
        out_dir="faiss_index_bert",
        kmeans_out="kmeans_bert.pkl",
    ),
}

def build(model_key: str, n_clusters: int = 15):
    cfg = DATA[model_key]
    emb_df   = pd.read_csv(cfg["csv_emb"])      # shape (N, dim)
    meta_df  = pd.read_csv(cfg["csv_meta"])     # 需含欄位：cluster, title, content, …
    vectors  = emb_df.values.astype("float32")

    # ❶ 建 LangChain Document list（可加任意 metadata）
    docs = [
        Document(
            page_content=row["processed_content(ckip)"],
            metadata={
                "title":  row["輿情標題"],
                "cluster_id": int(row["cluster"]),
                "source": f"news_{idx}"
            },
        )
        for idx, row in meta_df.iterrows()
    ]

    # ❷ 建 FAISS 向量庫
    embedder = HuggingFaceEmbeddings(model_name=cfg["embed_name"])
    vs = FAISS.from_embeddings(embeddings=vectors, texts=[d.page_content for d in docs],
                               embedding=embedder, metadatas=[d.metadata for d in docs])

    vs.save_local(cfg["out_dir"])

    # ❸ KMeans（給 Mode 2, 3 用）
    km = KMeans(n_clusters=n_clusters, random_state=42).fit(vectors)
    joblib.dump(km, cfg["kmeans_out"])
    print(f"[{model_key}] index & KMeans saved ✔")

if __name__ == "__main__":
    for key in DATA:
        build(key)
