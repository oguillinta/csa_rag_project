from openai import OpenAI
from typing import List, Tuple
from config import OPENAI_MODEL_CHAT, OPENAI_MODEL_EMBED, ROLE_NAME, ROLE_SCOPE, SIMILARITY_THRESHOLD, TOP_K
import httpx 

def get_client(api_key: str):
    if not api_key:
        raise ValueError("OpenAI API key is missing.")
    http_client = httpx.Client(trust_env=False, timeout=60)
    return OpenAI(api_key=api_key, http_client=http_client)

def embed_texts(client: OpenAI, texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(model=OPENAI_MODEL_EMBED, input=texts)
    return [d.embedding for d in resp.data]

def best_hits(results) -> List[tuple]:
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    sims = [1 - d for d in dists]
    return list(zip(docs, metas, sims))

def build_system_prompt():
    text = (
        "You are a **{role}**.\n"
        "Your scope: {scope}.\n\n"
        "Rules:\n"
        "1) Only use information from the retrieved snippets.\n"
        "2) If the question is outside your scope or not supported by the snippets, politely refuse.\n"
        "3) Be concise. If useful, include (source: filename).\n"
        "4) Do not invent facts.\n"
    )
    return text.format(role=ROLE_NAME, scope=ROLE_SCOPE)

def compose_context(hits: List[tuple]) -> str:
    lines = []
    for doc, meta, sim in hits:
        src = meta.get("source", "unknown")
        lines.append(f"[{src}] {doc}")
    return "\n".join(lines)

def is_confident(hits: List[tuple], threshold: float) -> bool:
    if not hits:
        return False
    return max(h[2] for h in hits) >= threshold

def answer_with_rag(client: OpenAI, user_question: str, hits: List[tuple], threshold: float) -> str:
    if not is_confident(hits, threshold):
        return ("I don't have enough data to answer that as a Customer Support Agent. "
                "Try asking about order status, shipping, cancellations, or returns.")

    context = compose_context(hits)
    system_msg = build_system_prompt()
    user_msg = (
        "User question: " + user_question + "\n\n"
        "Retrieved snippets:\n" + context + "\n\n"
        "Answer using only the snippets above."
    )

    resp = client.chat.completions.create(
        model=OPENAI_MODEL_CHAT,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2
    )
    return resp.choices[0].message.content.strip()
