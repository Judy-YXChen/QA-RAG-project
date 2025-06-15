# scripts/build_index.py
import os, sys, joblib, pandas as pd, numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
# --- 新匯入路徑 (v0.2) ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# -------------------------------------------------------
# 1. 路徑輔助：讓腳本可在 scripts/ 內執行
# -------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = lambda fname: os.path.join(ROOT, "data", fname)
index_path = lambda fname: os.path.join(ROOT, "indexes", fname)

# -------------------------------------------------------
# 2. 你的四個檔案設定
# -------------------------------------------------------
DATA = {
    "bge": dict(
        embed_name="BAAI/bge-base-zh",
        vec_file=data_path("bge_base_vec_ckip.csv"),
        meta_file=data_path("bge_kmeans_clustered_result_406.csv"),
        out_dir=index_path("faiss_index_bge"),
        kmeans_out=index_path("kmeans_bge.pkl"),
    ),
    "bert": dict(
        embed_name="bert-base-chinese",
        vec_file=data_path("bert_base_vec_ckip.csv"),
        meta_file=data_path("bert_kmeans_clustered_result_406.csv"),
        out_dir=index_path("faiss_index_bert"),
        kmeans_out=index_path("kmeans_bert.pkl"),
    ),
}

# -------------------------------------------------------
# 3. 建索引 + KMeans
# -------------------------------------------------------
def build(model_key: str):
    cfg = DATA[model_key]

    # 3-1 讀嵌入向量
    vectors = pd.read_csv(cfg["vec_file"], header=None).values.astype("float32")

    # 3-2 讀 meta
    meta_df = pd.read_csv(cfg["meta_file"])
    if len(vectors) != len(meta_df):
        raise ValueError(
            f"[{model_key}] 向量筆數 {len(vectors)} ≠ meta 筆數 {len(meta_df)}"
        )

    # 3-3 構建 Document list（欄位名稱可自行調整）
    docs = [
        Document(
            page_content=str(row.get("processed_content(ckip)", row.get("content", ""))),
            metadata={
                "title": row.get("title", ""),
                "cluster_id": int(row["cluster"]),
                "source": row.get("source", ""),  # 使用原始連結網址
                "doc_id": idx + 1,  # 從 1 開始
            },
        )
        for idx, row in meta_df.iterrows()
    ]

    # 3-4 建 FAISS VectorStore
    embedder = HuggingFaceEmbeddings(model_name=cfg["embed_name"])
    text_embeddings = list(zip([d.page_content for d in docs], vectors))
    vs = FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=embedder,
        metadatas=[d.metadata for d in docs],
    )
    os.makedirs(cfg["out_dir"], exist_ok=True)
    vs.save_local(cfg["out_dir"])

    # 3-5 KMeans（供 Mode 2/3 預測 cluster）
    km = KMeans(n_clusters=meta_df["cluster"].nunique(), random_state=42).fit(vectors)
    joblib.dump(km, cfg["kmeans_out"])

    print(f"[{model_key}] ✅ Index → {cfg['out_dir']} | KMeans → {cfg['kmeans_out']}")

# -------------------------------------------------------
# 4. CLI 入口
# -------------------------------------------------------
if __name__ == "__main__":
    for key in DATA:
        build(key)
