# QA-RAG-Project 🧠🔎

一個 **Retrieval-Augmented Generation (RAG)** 原型，  
採用 **Streamlit** 前端 + **LangChain 0.3** / **LangGraph** 後端，  
支援向量檢索（FAISS）、多模型選擇、分群 Mode、以及來源可追溯的 AI 答覆。

---

## ✨ 特色

| 功能 | 說明 |
|------|------|
| **三種檢索模式** | ① 全庫檢索 (All News) ② 群內檢索 (Within Cluster) ③ 群內 + 簡/詳答模板 (LangGraph) |
| **多向量空間** | `BAAI/bge-base-zh` & `bert-base-chinese` 皆可切換 |
| **快取與熱重載** | `@st.cache_resource` 使索引載入僅一次；Streamlit 自動熱重載 |
| **安全反序列化** | 使用 `allow_dangerous_deserialization=True`，僅對本地索引開啟 |
| **.env 機制** | API Key 不入 Git，採 `python-dotenv` 載入 |

---

## 🗂️ 專案結構

```
QA-RAG-project/
│
├─ app.py                  # Streamlit 入口
├─ data/                   # 原始 / 清洗後 CSV & 嵌入向量
│   ├─ bge_base_vec_ckip.csv
│   ├─ bert_base_vec_ckip.csv
│   ├─ bge_kmeans_clustered_result_406.csv
│   └─ bert_kmeans_clustered_result_406.csv
│
├─ indexes/                # FAISS / KMeans 產物 (gitignored)
│   ├─ faiss_index_bge/
│   └─ faiss_index_bert/
│
├─ scripts/
│   └─ build_index.py      # 一鍵產生索引
│
├─ .env.example            # 環境變數範例 (請自行複製為 .env)
├─ requirements.txt
└─ README.md
```

---

## ⚙️ 安裝

```bash
git clone https://github.com/<your-org>/QA-RAG-project.git
cd QA-RAG-project
python -m venv .venv && source .venv/bin/activate   # Windows 用 .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env          # 填入 OPENAI_API_KEY=sk-xxx
```

---

## 🏗️ 建立向量索引與分群模型

```bash
python scripts/build_index.py
```

> **備註**：
>
> * 若想更換嵌入模型或資料集，編輯 `scripts/build_index.py` 的 `DATA` 字典。
> * 產物將存於 `indexes/`，不會推上 Git。

---

## 🚀 執行

```bash
streamlit run app.py
```

打開瀏覽器 [http://localhost:8501](http://localhost:8501)，  
在 Sidebar 選擇 *Vector Model* 與 *Retrieval Mode*，即可開始提問！

---

## 🔑 環境變數

| 變數               | 說明                                    |
| ------------------ | ---------------------------------------- |
| `OPENAI_API_KEY`   | 你的 OpenAI Key；寫在 `.env` 或部署平台的 Secret |

---

## 📝 TODO

* [ ] LangGraph Mode 3：簡 / 詳答模板實作
* [ ] Hybrid Search (BM25 + dense)
* [ ] 評測腳本：EM / Rouge-L 指標
* [ ] Dockerfile & CI