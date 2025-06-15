import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from ckip_transformers.nlp import CkipWordSegmenter
from pandas import DataFrame
from stopwordsiso import stopwords
from time import sleep

# 初始化 CKIP 斷詞器（只要初始化一次，不要每次函數呼叫都跑）
ws_driver = CkipWordSegmenter(device=-1)  # -1 表示使用 CPU
STOPWORDS = stopwords(["zh", "zh-tw"])

def extract_keywords_ckip(text: str) -> list[str]:
    """使用 CKIP 將輸入文字斷詞，並去除繁中 & 簡中停用詞"""
    words = ws_driver([text])[0]
    keywords = [w for w in words if w not in STOPWORDS and len(w.strip()) > 1]
    return list(set(keywords))

# ====== 載入 QA 資料與向量 ======
QA_DF = pd.read_csv("data/Groq_news_with_questions_Final.csv")
QA_VEC_SIMPLE = np.load("data/QA_simple_vectors.npy")
QA_VEC_COMPLEX = np.load("data/QA_complex_vectors.npy")

def query_type_check(query: str) -> str:
    """判斷輸入問題為簡問還是詳問。"""
    return "Q_simple" if len(query.strip()) < 40 else "Q_complex"

def get_top_simulated_questions(
    query: str,
    query_vec: np.ndarray,
    cluster_id: int,
    question_type: str,
    top_n: int = 20
) -> DataFrame:
    """語意找 top-N，再加上關鍵詞強匹配 QA 補充進去"""
    qa_subset = QA_DF[QA_DF["cluster"] == cluster_id].reset_index(drop=True)
    vec_subset = QA_VEC_SIMPLE[qa_subset.index] if question_type == "Q_simple" else QA_VEC_COMPLEX[qa_subset.index]

    query_vec = np.array(query_vec, dtype=np.float32).reshape(1, -1)
    similarities = cosine_similarity(query_vec, vec_subset)[0]

    top_indices = similarities.argsort()[::-1][:top_n]
    top_df = qa_subset.iloc[top_indices].copy()
    top_df["similarity"] = similarities[top_indices]

    print(f"[DEBUG] Cluster {cluster_id} QA 數量：{len(qa_subset)}")
    print(f"[DEBUG] Top {top_n} QA by similarity:")
    for i, row in top_df.iterrows():
        print(f"- {row[question_type]} (similarity: {row['similarity']:.4f})")

    # 加入 CKIP 強相關詞補強機制
    query_terms = extract_keywords_ckip(query)
    print(f"[DEBUG] Query CKIP 關鍵詞: {query_terms}")

    def is_overlap(text):
        return any(term in text for term in query_terms)

    matched = qa_subset[qa_subset[question_type].apply(is_overlap)]
    print(f"[DEBUG] 額外找到 {len(matched)} 筆含 query 詞的 QA")

    # 合併（避免重複）並加入 priority 欄位排序
    merged_df = pd.concat([top_df, matched]).drop_duplicates().reset_index(drop=True)

    # 優先排含 query 關鍵詞的 QA，再依語意相似度排序
    merged_df["priority"] = merged_df[question_type].apply(
        lambda q: any(term in q for term in query_terms)
    )

    merged_df = merged_df.sort_values(by=["priority", "similarity"], ascending=[False, False])
    merged_df = merged_df.drop(columns=["priority"])  # 可選：刪除 debug 欄位

    # 最後限制最多 top_n（可改為 20 或 30）
    return merged_df.head(top_n)




def judge_relevance_batch(llm: ChatOpenAI, query: str, qa_list: list[str]) -> list[bool]:
    """一次判斷多個 QA 是否與 query 有關，回傳 boolean list"""
    qa_block = "\n".join(
        [f"{i+1}. {q}" for i, q in enumerate(qa_list)]
    )
    prompt = (
        f"使用者的問題是：{query}\n"
        f"以下是可能的模擬問題列表，請你判斷每一題是否與使用者的問題有關，僅回覆 '是' 或 '否'，以換行分隔：\n{qa_block}"
    )
    try:
        response = llm.invoke(prompt)

        # 取得文字內容
        text = getattr(response, "content", response)

        # 若仍是 BaseMessage（非 str），再取 content
        if not isinstance(text, str):
            text = getattr(text, "content", "")

        # 若為 list，處理成單一 string
        if isinstance(text, list):
            text = text[0] if isinstance(text[0], str) else text[0].get("text", "")

        text = text.strip()
        flags = text.splitlines()
        return [("是" in f.strip()) for f in flags]

    except Exception as e:
        print(f"[ERROR] relevance batch check failed: {e}")
        return [False] * len(qa_list)


def select_relevant_qa(llm, query, qa_df: pd.DataFrame, question_col: str, max_rounds: int = 3) -> list[dict]:
    """使用 Gemini 檢查問題是否與 query 相關，最多進行三輪檢查，最後不足再用 cosine similarity 補滿。"""
    if len(qa_df) == 0:
        print("[DEBUG] QA 資料為空，無法進行相關性判斷")
        return []

    # 準備 QA pool
    qa_df = qa_df[[question_col, "title", "source", "similarity"]].dropna(subset=[question_col]).drop_duplicates()
    qa_list = qa_df.to_dict(orient="records")

    # prompt chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位多領域的問答系統助手，請判斷下列問題是否與使用者問題主題相符。如果你覺得它有助於回答用戶的問題，請回答「是」，否則回答「否」。\n\n用戶問題：{query}\n\n問題：{question}"),
        ("human", "{question}")
    ])
    relevance_chain = LLMChain(llm=llm, prompt=prompt)

    selected = []
    batch_size = 10
    total = len(qa_list)
    qa_list = qa_list[:max_rounds * batch_size]  # 限制最多檢查數量

    print(f"[DEBUG] 開始 relevance 檢查（question type: {question_col}）")

    for i in range(max_rounds):
        start = i * batch_size
        end = start + batch_size
        batch = qa_list[start:end]
        print(f"[DEBUG] === Relevance Check Round {i+1} ===")
        print(f"[DEBUG] Round {i+1} QA 數量: {len(batch)}")

        if not batch:
            break

        for q in batch:
            question = q[question_col]
            print(f"[Q] {question}")
            try:
                ans = relevance_chain.invoke({"query": query, "question": question})["text"]
                print(f"[A] {ans}")
                if "是" in ans:
                    selected.append(q)
            except Exception as e:
                print(f"[ERROR] relevance check fail: {e}")

        # 避免超過 API 限制
        if i < max_rounds - 1:
            print("[DEBUG] 等待 10 秒以避免 Gemini API 達到速率上限...")
            sleep(10)

        if len(selected) >= 10:
            print(f"[DEBUG] 已累積 {len(selected)} 筆相關 QA，提前結束")
            break

    # === 補足不滿 10 筆的部分 ===
    if len(selected) < 10:
        print("[DEBUG] 補齊相關 QA 至 10 筆...")
        remaining = [q for q in qa_list if q not in selected]
        remaining_sorted = sorted(remaining, key=lambda q: q.get("similarity", 0), reverse=True)
        to_add = remaining_sorted[:10 - len(selected)]
        print(f"[DEBUG] 補充 {len(to_add)} 筆 QA：")
        for q in to_add:
            print("-", q[question_col], f"(similarity: {q.get('similarity', 0):.4f})")
        selected += to_add

    print(f"[DEBUG] 最終選中的 QA 數量: {len(selected)}")
    return selected

def generate_final_answer(llm: ChatOpenAI, query: str, selected_qa: list) -> str:
    """根據篩選後的 QA 列表，組合 context 並請 Gemini 回答。"""
    context = "\n\n".join(
        [f"標題：{q['title']}\n新聞內容：{q['content']}" for q in selected_qa if "content" in q]
    )
    prompt = (
        f"使用者問題：{query}\n"
        f"請根據以下新聞內容，彙整出有條理的答案：\n\n{context}"
    )
    try:
        result = llm.invoke(prompt)

        print("[DEBUG] LLM invoke 回傳型別：", type(result))
        print("[DEBUG] LLM invoke 回傳內容：", result)

        if isinstance(result, str):
            return result.strip()

        if hasattr(result, "content"):
            content = result.content
        else:
            content = result

        if isinstance(content, list):
            string_items = [c for c in content if isinstance(c, str)]
            dict_items = [c["text"] for c in content if isinstance(c, dict) and "text" in c]
            merged = string_items + dict_items
            if merged:
                return "\n".join(merged).strip()
            return str(content)

        return str(content).strip()

    except Exception as e:
        print(f"[ERROR] LLM 回答失敗：{e}")
        return "⚠️ 無法產生答案，請稍後再試。"


