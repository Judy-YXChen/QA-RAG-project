import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import ChatOpenAI
from ckip_transformers.nlp import CkipWordSegmenter
from pandas import DataFrame
from stopwordsiso import stopwords

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
        lambda q: any(term in q for term in extract_keywords_ckip(query))
    )

    merged_df = merged_df.sort_values(by=["priority", "similarity"], ascending=[False, False])
    merged_df = merged_df.drop(columns=["priority"])  # 可選：刪除 debug 欄位

    # 最後限制最多 top_n（可改為 20 或 30）
    return merged_df.head(top_n)




def judge_relevance(llm: ChatOpenAI, query: str, qa_question: str) -> bool:
    """使用 Gemini 判斷該 QA 是否與 user query 相關。"""
    prompt = (
        f"使用者的問題是：{query}\n"
        f"以下是一個可能的模擬問題：{qa_question}\n"
        f"請判斷這個模擬問題是否有助於回答使用者的問題，僅回覆 是 或 否。"
    )
    try:
        result = llm.invoke(prompt)
        return "是" in result.content
    except Exception:
        return False

def select_relevant_qa(llm: ChatOpenAI, query: str, top_20_df: pd.DataFrame, question_col: str, max_rounds=3):
    """從前20筆 QA 中進行 relevance check，最多保留10筆相關項目（含 debug 訊息）。"""
    selected = []
    reserve = top_20_df.iloc[10:].to_dict("records")
    candidates = top_20_df.iloc[:10].to_dict("records")

    print(f"[DEBUG] 開始 relevance 檢查（question type: {question_col}）")
    print(f"[DEBUG] Top 20 QA preview:")
    for i, q in enumerate(top_20_df[question_col].tolist(), 1):
        print(f"{i:02d}. {q}")

    rounds = 0
    while rounds < max_rounds:
        print(f"\n[DEBUG] === Relevance Check Round {rounds+1} ===")
        still_needed = 10 - len(selected)
        if still_needed <= 0:
            break

        new_selected = []
        for item in candidates:
            qtext = item[question_col]
            prompt = (
                f"使用者的問題是：{query}\n"
                f"以下是一個模擬問題：{qtext}\n"
                f"請只回覆 是 或 否，這個模擬問題是否有助於回答使用者的問題？"
            )
            try:
                result = llm.invoke(prompt)
                reply = str(getattr(result, "content", result)).strip()
                print(f"[Q] {qtext}\n[A] {reply}\n")
                if reply.startswith("是"):
                    new_selected.append(item)
            except Exception as e:
                print(f"[ERROR] relevance check fail: {e}")

        selected.extend(new_selected)

        # 從 reserve 補足
        candidates = []
        while reserve and len(selected) + len(candidates) < 10:
            candidates.append(reserve.pop(0))

        rounds += 1

    print(f"[DEBUG] 最終選中的 QA 數量: {len(selected)}")
    return selected[:10]


def generate_final_answer(llm: ChatOpenAI, query: str, selected_qa: list) -> str:
    """根據篩選後的 QA 列表，組合 context 並請 Gemini 回答。"""
    context = "\n\n".join(
        [f"標題：{q['title']}\n新聞內容：{q['content']}" for q in selected_qa]
    )
    prompt = (
        f"使用者問題：{query}\n"
        f"請根據以下新聞內容，彙整出有條理的答案：\n\n{context}"
    )
    result = llm.invoke(prompt)
    return str(getattr(result, "content", result))

