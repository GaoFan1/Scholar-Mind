from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import chromadb
from openai import OpenAI
import hashlib, json

app = FastAPI()

# ── CORS：允许你的前端域名 ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 上线后改为你的 GitHub Pages 域名
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 客户端 ──
deepseek = OpenAI(
    api_key="sk-47caf8ee0df84ba8a712ae60af574e4a",
    base_url="https://api.deepseek.com"
)
chroma = chromadb.PersistentClient(path="./chroma_data")


# ── 工具函数 ──
def split_text(text: str, chunk_size=512, overlap=64):
    """按字符切片，带重叠"""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def embed(texts: list[str]) -> list[list[float]]:
    """DeepSeek embedding（不额外收费）"""
    resp = deepseek.embeddings.create(
        model="deepseek-embedding",
        input=texts
    )
    return [r.embedding for r in resp.data]


# ── 接口 1：上传并向量化论文 ──
class IndexRequest(BaseModel):
    paper_id: str
    paper_name: str
    content: str
    username: str

@app.post("/api/index")
async def index_paper(req: IndexRequest):
    col_name = f"user_{req.username[:20]}"
    col = chroma.get_or_create_collection(col_name)

    # 先删旧版本（重复上传）
    try:
        existing = col.get(where={"paper_id": req.paper_id})
        if existing["ids"]:
            col.delete(ids=existing["ids"])
    except:
        pass

    chunks = split_text(req.content)
    if not chunks:
        raise HTTPException(400, "内容为空")

    embeddings = embed(chunks)
    ids = [f"{req.paper_id}_{i}" for i in range(len(chunks))]

    col.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=[{
            "paper_id": req.paper_id,
            "paper_name": req.paper_name,
            "chunk_index": i
        } for i in range(len(chunks))]
    )
    return {"status": "ok", "chunks": len(chunks)}


# ── 接口 2：RAG 问答（流式）──
class QARequest(BaseModel):
    question: str
    paper_id: str       # 单篇模式填 paper_id，跨库模式填 ""
    username: str
    paper_name: str = ""
    top_k: int = 5

@app.post("/api/qa")
async def rag_qa(req: QARequest):
    col_name = f"user_{req.username[:20]}"
    try:
        col = chroma.get_or_create_collection(col_name)
    except:
        raise HTTPException(500, "集合初始化失败")

    q_vec = embed([req.question])[0]

    # 检索：单篇 or 跨文库
    where = {"paper_id": req.paper_id} if req.paper_id else None
    results = col.query(
        query_embeddings=[q_vec],
        n_results=req.top_k,
        where=where,
        include=["documents", "metadatas", "distances"]
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    if not docs:
        raise HTTPException(404, "未找到相关内容，请先索引论文")

    # 拼装上下文，带来源标注
    context_parts = []
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        context_parts.append(
            f"【片段{i+1}·{meta['paper_name']}·第{meta['chunk_index']+1}段】\n{doc}"
        )
    context = "\n\n".join(context_parts)

    system_prompt = f"""你是专业学术论文分析助手。
以下是从论文中检索到的最相关片段，请严格基于这些内容回答问题，回答时注明引用自哪个片段。

{context}

要求：
1. 只引用上述片段中的信息，不要杜撰
2. 回答时用【片段N】标注出处
3. 用中文回答，专业准确"""

    def generate():
        stream = deepseek.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.question}
            ],
            stream=True
        )
        # 先发送来源信息
        sources = [{"name": m["paper_name"], "chunk": m["chunk_index"]+1} for m in metas]
        yield f"data: {json.dumps({'sources': sources})}\n\n"
        # 再流式发文字
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield f"data: {json.dumps({'text': delta})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── 接口 3：健康检查 ──
@app.get("/")
def health():
    return {"status": "ScholarMind RAG API running"}