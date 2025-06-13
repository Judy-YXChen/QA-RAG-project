import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none" # 關閉 Streamlit 的「檔案監聽 ‧ 熱重載」機制，避免 Streamlit 監聽 torch.classes 目錄時觸發的已知 bug
import streamlit as st
from typing import List, Dict, Any, Optional, cast
from enum import Enum
import requests
import joblib
from pydantic import SecretStr
from dotenv import load_dotenv
load_dotenv() # 自 .env 載入 GOOGLE_API_KEY，供 ChatGoogleGenerativeAI 使用

# ============================ Page Config ============================
st.set_page_config(page_title="RAG QA Chatbot", page_icon="🧠", layout="centered")

# -------------------------------------------------------------------
# 0. ENUMS & CONSTANTS
# -------------------------------------------------------------------
class RetrievalMode(str, Enum):
    """The three retrieval pipelines a user can choose from."""
    ALL = "All News"                         # 🔍 全庫檢索
    CLUSTER = "Within Cluster"              # 🔍 限縮到同群
    TEMPLATE = "Cluster + QA Template"      # 🔍 同群 + LangGraph（簡/詳答）

class VectorModel(str, Enum):
    """Which embedding / clustering space to use."""
    BGE = "BGE‑base‑zh"            # 中文語料最佳，適合新聞語句
    BERT = "bert‑base‑chinese"     # 經典中文 BERT

# 對應路徑 & 模型名稱
MODEL_CONFIG: Dict[VectorModel, Dict[str, str]] = {
    VectorModel.BGE: {
        "embedding_name": "BAAI/bge-base-zh",
        "faiss_path": "indexes/faiss_index_bge",
        "cluster_path": "indexes/kmeans_bge.pkl",
    },
    VectorModel.BERT: {
        "embedding_name": "bert-base-chinese",
        "faiss_path": "indexes/faiss_index_bert",
        "cluster_path": "indexes/kmeans_bert.pkl",
    },
}

# TODO: 與負責 LangGraph 的同學確認真實 URL
LG_ENDPOINT = "http://localhost:8080/rag/cluster_template"  # POST JSON → {answer, sources}

# -------------------------------------------------------------------
# 1. SIDEBAR (UI) -----------------------------------------------------
# -------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    vec_model: VectorModel = st.selectbox(
        "Vector Model", list(VectorModel), format_func=lambda m: m.value)
    mode: RetrievalMode = st.selectbox(
        "Retrieval Mode", list(RetrievalMode), format_func=lambda m: m.value)
    top_k = st.slider("Top‑K Documents", 1, 20, 1)
    temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.7)
    st.markdown("---")
    st.caption("Mode 1 = 全庫 | Mode 2 = 依分群 | Mode 3 = 依分群 + QA")

# -------------------------------------------------------------------
# 2. RESOURCE LOADERS (CACHED) ---------------------------------------
# -------------------------------------------------------------------
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent

@st.cache_resource(show_spinner=False)
def load_resources_per_model(cfg: Dict[str, str]) -> Dict[str, Any]:
    """Load embeddings, FAISS index, cluster model, base LLM for one vector model."""
    embeddings = HuggingFaceEmbeddings(model_name=cfg["embedding_name"])
    faiss_dir = ROOT / cfg["faiss_path"] # 絕對路徑，避免因工作目錄不同而找不到
    vector_store = FAISS.load_local(str(faiss_dir), embeddings, allow_dangerous_deserialization=True)
    # clusterer is optional ## debug
    # === Load cluster model ===
    clusterer = None
    clusterer_path = os.path.join(ROOT, cfg["cluster_path"])
    print("[DEBUG] cluster_path =", clusterer_path)
    print("[DEBUG] Exists?", os.path.exists(clusterer_path))

    try:
        clusterer = joblib.load(clusterer_path)
        print("[INFO] Cluster model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load clusterer from {clusterer_path}: {e}")
        clusterer = None

    gemini_api_key = SecretStr(os.getenv("GOOGLE_API_KEY") or "")
    gemini_api_base = "https://generativelanguage.googleapis.com/v1beta/openai/"
    base_llm = ChatOpenAI(
        api_key=gemini_api_key,
        base_url=gemini_api_base,
        model="gemini-1.5-flash",
        temperature=0.0
    )
    return {
        "embeddings": embeddings,
        "vector_store": vector_store,
        "clusterer": clusterer,
        "base_llm": base_llm,
    }

@st.cache_resource(show_spinner=False)
def load_all_resources() -> Dict[VectorModel, Dict[str, Any]]:
    """Cache resources for all configured vector models."""
    return {m: load_resources_per_model(MODEL_CONFIG[m]) for m in VectorModel}

_RESOURCES = load_all_resources()

# -------------------------------------------------------------------
# 3. HELPER FUNCTIONS -------------------------------------------------
# -------------------------------------------------------------------

def _predict_cluster(query: str, model: VectorModel) -> Optional[int]:
    """Predict cluster ID using vector model's clusterer."""
    clusterer = _RESOURCES[model]["clusterer"]
    if clusterer is None:
        print("[WARN] Cluster model not loaded.")
        return None

    try:
        vec_raw = _RESOURCES[model]["embeddings"].embed_query(query)
        vec = np.asarray(vec_raw, dtype=np.float32).reshape(1, -1)
        cid = clusterer.predict(vec)[0]
        print(f"[DEBUG] Predicted cluster ID: {cid}")
        return int(cid)
    except Exception as e:
        print(f"[ERROR] Failed to predict cluster: {e}")
        return None

def _build_retriever(model: VectorModel, k: int, cluster_id: Optional[int] = None):
    """Return retriever bound to chosen vector space, optionally filtered by cluster."""
    vs = _RESOURCES[model]["vector_store"]
    kwargs: Dict[str, Any] = {"k": k}

    if cluster_id is not None:
        kwargs["filter"] = {"cluster_id": cluster_id}
        print(f"[DEBUG] Applying cluster filter: cluster_id = {cluster_id}")
    
    retriever = vs.as_retriever(search_kwargs=kwargs)
    return retriever

def get_rag_answer(query: str, model: VectorModel, mode: RetrievalMode, k: int, temp: float):
    """Route query through the selected pipeline & vector space."""
    gemini_api_key = SecretStr(os.getenv("GOOGLE_API_KEY") or "")
    gemini_api_base = "https://generativelanguage.googleapis.com/v1beta/openai/"

    llm = ChatOpenAI(
        api_key=gemini_api_key,
        base_url=gemini_api_base,
        model="gemini-1.5-flash",
        temperature=temp
    )

    # Mode 1 ── 全庫檢索
    if mode == RetrievalMode.ALL:
        retriever = _build_retriever(model, k)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Use the following context to answer the question. If you don’t know, say so.\n\n{context}"),
            ("human", "{input}")
        ])
        combine_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
        chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=combine_chain)
        result = chain.invoke({"input": query})
        return result["answer"], result.get("source_documents", [])

    # Mode 2 ── 群內檢索
    if mode == RetrievalMode.CLUSTER:
        cid = _predict_cluster(query, model)
        if cid is None:
            return "⚠️ 無法辨識此問題屬於哪個主題群，將跳過回答。", []
        retriever = _build_retriever(model, k, cluster_id=cid)
        docs = retriever.invoke(query)
        if not docs:
            return f"⚠️ 雖然預測為第 {cid} 群，但查無相關資料。", []
        print(f"[DEBUG] Retrieved {len(docs)} docs from cluster {cid}")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Use the following context to answer the question. If you don’t know, say so.\n\n{context}"),
            ("human", "{input}")
        ])
        combine_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
        chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=combine_chain)
        try:
            result = chain.invoke({"input": query})
            answer = f"(Model {model.value} | Cluster {cid})\n" + result["answer"]
            return answer, result.get("source_documents", [])
        except Exception as e:
            return f"❌ 分群問答流程失敗：{e}", []

    # Mode 3 ── LangGraph 模板（外部 API 呼叫）
    cid = _predict_cluster(query, model)
    payload = {
        "query": query,
        "cluster_id": cid,
        "vector_model": model.value,
        "top_k": k,
        "temperature": temp,
        "answer_style": "auto",
    }
    try:
        rsp = requests.post(LG_ENDPOINT, json=payload, timeout=90)
        rsp.raise_for_status()
        data = rsp.json()
        return data.get("answer", "⚠️ 尚未產生回答。"), data.get("sources", [])
    except Exception as e:
        return f"❌ LangGraph 呼叫失敗：{e}", []

# -------------------------------------------------------------------
# 4. SESSION STATE (CHAT) --------------------------------------------
# -------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = cast(List[Dict[str, str]], [])

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# -------------------------------------------------------------------
# 5. CHAT INPUT LOOP --------------------------------------------------
# -------------------------------------------------------------------
user_query = st.chat_input("Ask about your knowledge base…")

if user_query:
    # 5.1 ── Echo user
    st.session_state["messages"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 5.2 ── Generate answer
    with st.chat_message("assistant"):
        with st.spinner("🔎 Retrieving & Generating…"):
            answer, docs = get_rag_answer(user_query, vec_model, mode, top_k, temperature)
            st.markdown(answer)
            if docs:
                with st.expander("🔗 Sources"):
                    for i, d in enumerate(docs, 1):
                        title = d.metadata.get("title", f"Doc {i}")
                        source = d.metadata.get("source", "")
                        st.markdown(f"{i}. **{title}** — {source}")

    # -- 5.3 Save assistant turn
    st.session_state["messages"].append({"role": "assistant", "content": answer})

# -------------------------------------------------------------------
# 6. FOOTER -----------------------------------------------------------
# -------------------------------------------------------------------
st.markdown(
    """<hr style='margin-top:2rem;margin-bottom:1rem;'>\n    <center><small>Built with 🦜 LangChain + 🧠 Streamlit </small></center>""",
    unsafe_allow_html=True,
)
