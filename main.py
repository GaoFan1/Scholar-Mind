from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
import json, math

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

vector_store: dict = {}


def split_text(text: str, chunk_size=600, overlap=80):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def embed(texts: list) -> list:
    resp = deepseek.embeddings.create(
        model="deepseek-embedding",
        input=texts
    )
    return [r.embedding for r in resp.data]


def cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)


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
    embeddings = embed(chunks)
    vector_store[key] = [
        {
            "text": chunk,
            "embedding": emb,
            "paper_id": req.paper_id,
            "paper_name": req.paper_name,
            "chunk_index": i
        }
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]
    return {"status": "ok", "chunks": len(chunks)}


@app.post("/api/qa")
async def rag_qa(req: QARequest):
    candidates = []
    for key, chunks in vector_store.items():
        if not key.startswith(req.username):
            continue
        if req.paper_id and f"_{req.paper_id}" not in key:
            continue
        candidates.extend(chunks)

    if not candidates:
        raise HTTPException(404, "请先索引论文")

    q_vec = embed([req.question])[0]
    scored = sorted(
        candidates,
        key=lambda c: cosine_sim(q_vec, c["embedding"]),
        reverse=True
    )[:req.top_k]

    context = "\n\n".join(
        f"【片段{i+1}·{c['paper_name']}·第{c['chunk_index']+1}段】\n{c['text']}"
        for i, c in enumerate(scored)
    )

    system_prompt = f"""你是专业学术论文分析助手。
基于以下检索到的论文片段回答问题，回答时用【片段N】标注出处，用中文回答。

{context}"""

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