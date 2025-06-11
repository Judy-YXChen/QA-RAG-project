import streamlit as st
import csv
import os

st.set_page_config(page_title="RAG 問答系統", layout="centered")

st.title("🧠 Retrieval-Augmented Generation (RAG) Demo")

# ===== Query Input =====
st.subheader("請輸入您的問題")
user_query = st.text_input("例如：ChatGPT 是如何運作的？")

# ===== Advanced Settings =====
with st.expander("⚙️ 高級選項", expanded=False):
    embedding_model = st.selectbox("Embedding 模型", ["bge-base-zh", "bert-base-chinese"])
    top_k = st.slider("Top-K 檢索數量", min_value=1, max_value=10, value=5)
    temperature = st.slider("生成溫度（temperature）", min_value=0.0, max_value=1.0, value=0.7)
    enable_summary = st.checkbox("是否摘要來源內容", value=True)

# ===== Submit Query =====
if st.button("🚀 開始查詢") and user_query:
    # 模擬 RAG 輸出
    st.markdown("### 🧾 回答結果")
    st.success("ChatGPT 是一種大型語言模型，訓練自 OpenAI 的 GPT 架構...")

    st.markdown("### 📚 檢索到的來源文件")
    for i in range(top_k):
        st.info(f"來源 {i+1}：這是一段關於 ChatGPT 技術細節的說明...\n\n[來源連結](https://example.com)")

    st.markdown("### 🔍 LangGraph 執行流程（模擬）")
    st.code("節點順序：Retriever -> Reranker -> Generator -> Return")

# ===== Feedback =====
st.markdown("---")
st.subheader("🗣️ 回饋區")
feedback = st.radio("這個回答對你有幫助嗎？", ["👍 有幫助", "👎 沒幫助"])
comment = st.text_area("補充建議或錯誤指出（可選）")
if st.button("📩 提交回饋"):
    if not user_query:
        st.warning("請先輸入一個問題並執行查詢後再提供回饋。")
    else:
        # 記錄回答內容（這裡對應前面寫死的回覆，之後還要更新）
        response_text = "ChatGPT 是一種大型語言模型，訓練自 OpenAI 的 GPT 架構..."

        # 檢查檔案是否存在
        file_exists = os.path.isfile("feedback.csv")

        with open("feedback.csv", "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["query", "response", "feedback", "comment"])
            writer.writerow([user_query, response_text, feedback, comment])

        st.success("✅ 感謝您的回饋，我們已成功記錄。")
