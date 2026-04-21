from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
import json, math, re
from collections import Counter

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

deepseek = OpenAI(
    api_key="sk-47caf8ee0df84ba8a712ae60af574e4a",
    base_url="https://api.deepseek.com"
)

# 内存存储
doc_store: dict = {}


def split_text(text: str, chunk_size=500, overlap=50):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def tokenize(text: str) -> list:
    """简单分词：按字符和标点切分，适配中英文"""
    # 英文单词 + 中文单字
    tokens = re.findall(r'[a-zA-Z]+|[\u4e00-\u9fff]', text.lower())
    return tokens


def tf(tokens: list) -> dict:
    count = Counter(tokens)
    total = len(tokens) + 1
    return {w: c / total for w, c in count.items()}


def bm25_score(query_tokens: list, chunk_tokens: list, avg_len: float) -> float:
    """BM25 相似度打分"""
    k1, b = 1.5, 0.75
    chunk_tf = Counter(chunk_tokens)
    chunk_len = len(chunk_tokens)
    score = 0.0
    for token in query_tokens:
        if token in chunk_tf:
            freq = chunk_tf[token]
            idf = math.log(2)  # 简化 IDF
            num = freq * (k1 + 1)
            den = freq + k1 * (1 - b + b * chunk_len / (avg_len + 1))
            score += idf * num / den
    return score


class IndexRequest(BaseModel):
    paper_id: str
    paper_name: str
    content: str
    username: str


class QARequest(BaseModel):
    question: str
    paper_id: str
    username: str
    paper_name: str = ""
    top_k: int = 5


@app.get("/")
def health():
    return {"status": "ScholarMind RAG API running"}


@app.options("/api/index")
async def options_index():
    return {}


@app.options("/api/qa")
async def options_qa():
    return {}


@app.post("/api/index")
async def index_paper(req: IndexRequest):
    key = f"{req.username}_{req.paper_id}"
    chunks = split_text(req.content)
    if not chunks:
        raise HTTPException(400, "内容为空")

    doc_store[key] = [
        {
            "text": chunk,
            "tokens": tokenize(chunk),
            "paper_id": req.paper_id,
            "paper_name": req.paper_name,
            "chunk_index": i
        }
        for i, chunk in enumerate(chunks)
    ]
    return {"status": "ok", "chunks": len(chunks)}


@app.post("/api/qa")
async def rag_qa(req: QARequest):
    # 收集候选片段
    candidates = []
    for key, chunks in doc_store.items():
        if not key.startswith(req.username):
            continue
        if req.paper_id and f"_{req.paper_id}" not in key:
            continue
        candidates.extend(chunks)

    if not candidates:
        raise HTTPException(404, "请先索引论文")

    # BM25 检索
    query_tokens = tokenize(req.question)
    avg_len = sum(len(c["tokens"]) for c in candidates) / len(candidates)
    scored = sorted(
        candidates,
        key=lambda c: bm25_score(query_tokens, c["tokens"], avg_len),
        reverse=True
    )[:req.top_k]

    context = "\n\n".join(
        f"【片段{i+1}·{c['paper_name']}·第{c['chunk_index']+1}段】\n{c['text']}"
        for i, c in enumerate(scored)
    )

    system_prompt = f"""你是专业学术论文分析助手。
基于以下检索到的论文片段准确回答问题，回答时用【片段N】标注出处，用中文回答。

{context}

要求：只引用上述片段中的信息，不要杜撰。"""

    def generate():
        sources = [
            {"name": c["paper_name"], "chunk": c["chunk_index"] + 1}
            for c in scored
        ]
        yield f"data: {json.dumps({'sources': sources}, ensure_ascii=False)}\n\n"

        stream = deepseek.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.question}
            ],
            stream=True
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield f"data: {json.dumps({'text': delta}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )