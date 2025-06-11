import streamlit as st
from typing import List, Dict, Any, Optional, cast
from enum import Enum
import requests
import joblib

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

# 對應路徑 & 模型名稱（請依實際檔案調整）
MODEL_CONFIG: Dict[VectorModel, Dict[str, str]] = {
    VectorModel.BGE: {
        "embedding_name": "BAAI/bge-base-zh",
        "faiss_path": "faiss_index_bge",
        "cluster_path": "kmeans_bge.pkl",
    },
    VectorModel.BERT: {
        "embedding_name": "bert-base-chinese",
        "faiss_path": "faiss_index_bert",
        "cluster_path": "kmeans_bert.pkl",
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
    top_k = st.slider("Top‑K Documents", 1, 20, 5)
    temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.7)
    st.markdown("---")
    st.caption("Mode 1 = 全庫 | Mode 2 = 依分群 | Mode 3 = 交由 LangGraph（簡/詳答）")

# -------------------------------------------------------------------
# 2. RESOURCE LOADERS (CACHED) ---------------------------------------
# -------------------------------------------------------------------
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

@st.cache_resource(show_spinner=False)
def load_resources_per_model(cfg: Dict[str, str]) -> Dict[str, Any]:
    """Load embeddings, FAISS index, cluster model, base LLM for one vector model."""
    embeddings = HuggingFaceEmbeddings(model_name=cfg["embedding_name"])
    vector_store = FAISS.load_local(cfg["faiss_path"], embeddings)
    # clusterer is optional
    try:
        clusterer = joblib.load(cfg["cluster_path"])
    except FileNotFoundError:
        clusterer = None
    base_llm = OpenAI(model_name="gpt-4o-mini", temperature=0.0)
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
    """Predict cluster_id for the query using the chosen model."""
    clusterer = _RESOURCES[model]["clusterer"]
    if clusterer is None:
        return None
    try:
        vec = _RESOURCES[model]["embeddings"].embed_query(query)
        return int(clusterer.predict([vec])[0])
    except Exception:
        return None


def _build_retriever(model: VectorModel, k: int, cluster_id: Optional[int] = None):
    """Return retriever bound to chosen vector space, optionally filtered by cluster."""
    vs = _RESOURCES[model]["vector_store"]
    kwargs: Dict[str, Any] = {"k": k}
    if cluster_id is not None:
        kwargs["filter"] = {"cluster_id": cluster_id}
    return vs.as_retriever(search_kwargs=kwargs)


def get_rag_answer(query: str, model: VectorModel, mode: RetrievalMode, k: int, temp: float):
    """Route query through the selected pipeline & vector space."""

    base_llm = _RESOURCES[model]["base_llm"]

    # Mode 1 ── 全庫檢索
    if mode == RetrievalMode.ALL:
        retriever = _build_retriever(model, k)
        chain = RetrievalQA.from_chain_type(base_llm, retriever=retriever, chain_type="stuff")
        chain.combine_documents_chain.llm.temperature = temp
        result: Dict[str, Any] = chain({"query": query})
        return result["result"], result.get("source_documents", [])

    # Mode 2 ── 群內檢索
    if mode == RetrievalMode.CLUSTER:
        cid = _predict_cluster(query, model)
        if cid is None:
            return "🚧 Cluster model not loaded; using full‑corpus retrieval.", []
        retriever = _build_retriever(model, k, cluster_id=cid)
        chain = RetrievalQA.from_chain_type(base_llm, retriever=retriever, chain_type="stuff")
        chain.combine_documents_chain.llm.temperature = temp
        result: Dict[str, Any] = chain({"query": query})
        answer = f"(Model {model.value} | Cluster {cid})\n" + result["result"]
        return answer, result.get("source_documents", [])

    # Mode 3 ── LangGraph（多步模板）
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
        return data.get("answer", "[No answer]"), data.get("sources", [])
    except Exception as e:
        return f"❌ LangGraph call failed: {e}", []

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
    """<hr style='margin-top:2rem;margin-bottom:1rem;'>\n    <center><small>Built with 🦜 LangChain + LangGraph (mode 3) + 🧠 Streamlit </small></center>""",
    unsafe_allow_html=True,
)
