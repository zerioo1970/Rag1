# -*- coding: utf-8 -*-
"""
最小 RAG 示例(纯 Python,无需任何 API Key / 联网 / 第三方库)
======================================================

RAG = 检索增强生成,核心四步:
  1) 切分文档 (Chunking)        —— 把知识库切成小块
  2) 向量化   (Embedding)       —— 把每块文本变成向量(这里用 TF-IDF 演示)
  3) 检索     (Retrieval)       —— 用户提问也向量化,找出最相关的块
  4) 生成     (Generation)      —— 把检索到的内容塞给"大模型"生成回答

为了让你能直接跑通,这里:
  - 用 TF-IDF + 余弦相似度 代替真正的语义 embedding(无需下载模型)
  - 用一个"模拟生成器"代替真正的大模型(无需 API Key)
文件最后有注释,告诉你怎么换成真正的工业级组件。
"""

import re
import math
from collections import Counter


# ============================================================
# 0. 知识库(假设这是你公司/项目的私有资料)
# ============================================================
KNOWLEDGE_BASE = [
    "RAG 的全称是检索增强生成(Retrieval-Augmented Generation),"
    "由 Meta 的研究者在 2020 年的论文中首次提出。"
    "它的核心思想是:在向大模型提问前,先从外部知识库检索相关资料,再让模型基于资料作答。",

    "RAG 最大的好处是减少幻觉。因为答案有真实资料作为依据,而不是让模型凭记忆瞎编。"
    "同时它还能让模型用上训练时没见过的私有数据和最新数据。",

    "向量数据库是 RAG 系统的关键组件,用来存储文本的向量并支持快速相似度检索。"
    "常见的向量数据库有 Pinecone、Milvus、Chroma,以及 PostgreSQL 的 pgvector 扩展。",

    "Embedding(嵌入)是把文本转换成一组数字(向量)的过程。"
    "语义相近的文本,它们的向量在空间中也更接近,因此可以用向量距离来衡量语义相似度。",

    "切分(Chunking)是指把长文档拆成较小的文本块。"
    "块太大检索不精准,块太小又会丢失上下文,所以块的大小需要权衡,常见做法是按句子或固定长度切分。",

    "做 RAG 开发最常用的语言是 Python,常用框架包括 LangChain、LlamaIndex 和 Haystack。",
]


# ============================================================
# 1. 切分文档 (Chunking)
#    这里演示:把每条资料按句子切分,再合并到不超过 max_len 的小块
# ============================================================
def split_into_chunks(documents, max_len=60):
    chunks = []
    for doc in documents:
        # 按中文句号/分号等切句
        sentences = re.split(r"(?<=[。;!?])", doc)
        current = ""
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if len(current) + len(s) <= max_len:
                current += s
            else:
                if current:
                    chunks.append(current)
                current = s
        if current:
            chunks.append(current)
    return chunks


# ============================================================
# 2. 向量化 (Embedding) —— 用 TF-IDF 演示
#    真实场景会用神经网络 embedding 模型,语义更准
# ============================================================
def tokenize(text):
    """简单分词:英文按单词,中文按相邻两字(bigram),够演示用。"""
    text = text.lower()
    english = re.findall(r"[a-z0-9]+", text)
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    bigrams = [chinese_chars[i] + chinese_chars[i + 1]
               for i in range(len(chinese_chars) - 1)]
    return english + chinese_chars + bigrams


def build_tfidf(chunks):
    """根据所有 chunk 计算 IDF,并把每个 chunk 转成 TF-IDF 向量。"""
    doc_tokens = [tokenize(c) for c in chunks]
    n_docs = len(chunks)

    # 计算每个词在多少篇文档中出现过 -> IDF
    df = Counter()
    for tokens in doc_tokens:
        for t in set(tokens):
            df[t] += 1
    idf = {t: math.log((n_docs + 1) / (cnt + 1)) + 1 for t, cnt in df.items()}

    # 每个 chunk -> {词: tf-idf 权重}
    vectors = []
    for tokens in doc_tokens:
        tf = Counter(tokens)
        total = len(tokens)
        vec = {t: (cnt / total) * idf.get(t, 0.0) for t, cnt in tf.items()}
        vectors.append(vec)
    return vectors, idf


def embed_query(query, idf):
    """用同样的方式把用户问题转成向量。"""
    tokens = tokenize(query)
    tf = Counter(tokens)
    total = len(tokens) or 1
    return {t: (cnt / total) * idf.get(t, 0.0) for t, cnt in tf.items()}


# ============================================================
# 3. 检索 (Retrieval) —— 余弦相似度找最相关的块
# ============================================================
def cosine_similarity(v1, v2):
    common = set(v1) & set(v2)
    dot = sum(v1[t] * v2[t] for t in common)
    norm1 = math.sqrt(sum(w * w for w in v1.values()))
    norm2 = math.sqrt(sum(w * w for w in v2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def retrieve(query, chunks, vectors, idf, top_k=2):
    q_vec = embed_query(query, idf)
    scored = [(cosine_similarity(q_vec, vectors[i]), chunks[i])
              for i in range(len(chunks))]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


# ============================================================
# 4. 生成 (Generation) —— 这里用"模拟生成器"
#    真实场景:把 context + question 拼成 prompt 交给大模型
# ============================================================
def generate_answer(query, retrieved):
    context = "\n".join(f"  - {chunk}" for _, chunk in retrieved)

    # 这就是喂给真实大模型的 prompt 模板
    prompt = (
        "请只根据以下资料回答问题,不要编造。\n"
        f"【资料】\n{context}\n\n"
        f"【问题】{query}\n【回答】"
    )

    # ↓↓↓ 演示用的"假"生成:直接返回最相关的那条资料作为答案 ↓↓↓
    best_chunk = retrieved[0][1] if retrieved else "(没有找到相关资料)"
    fake_answer = f"根据资料,{best_chunk}"
    return prompt, fake_answer


# ============================================================
# 主流程:把四步串起来跑一遍
# ============================================================
def main():
    print("=" * 60)
    print("步骤 1:切分文档")
    chunks = split_into_chunks(KNOWLEDGE_BASE)
    print(f"知识库被切成了 {len(chunks)} 个块:")
    for i, c in enumerate(chunks):
        print(f"  [{i}] {c}")

    print("=" * 60)
    print("步骤 2:把每个块向量化 (TF-IDF)")
    vectors, idf = build_tfidf(chunks)
    print(f"完成,共 {len(vectors)} 个向量,词表大小 {len(idf)}")

    questions = [
        "RAG 为什么能减少幻觉?",
        "常见的向量数据库有哪些?",
        "做 RAG 用什么编程语言?",
    ]

    for q in questions:
        print("=" * 60)
        print(f"用户提问:{q}")
        print("-" * 60)
        print("步骤 3:检索最相关的块")
        retrieved = retrieve(q, chunks, vectors, idf, top_k=2)
        for score, chunk in retrieved:
            print(f"  相似度 {score:.3f} | {chunk}")

        print("-" * 60)
        print("步骤 4:生成回答")
        prompt, answer = generate_answer(q, retrieved)
        print("【最终回答】", answer)


if __name__ == "__main__":
    main()


# ============================================================
# 如何升级成"真正的"工业级 RAG?
# ============================================================
# 1) Embedding:把 build_tfidf / embed_query 换成真正的语义模型,例如:
#       from sentence_transformers import SentenceTransformer
#       model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
#       vec = model.encode(text)
#    或调用 OpenAI 的 text-embedding-3-small 接口。
#
# 2) 向量检索:数据量大时,把线性扫描换成向量数据库
#    (Chroma / Milvus / pgvector),支持百万级快速检索。
#
# 3) 生成:把 generate_answer 里的"假"回答换成真实大模型调用,例如:
#       from openai import OpenAI
#       client = OpenAI(api_key="你的key")
#       resp = client.chat.completions.create(
#           model="gpt-4o-mini",
#           messages=[{"role": "user", "content": prompt}],
#       )
#       answer = resp.choices[0].message.content
#
# 框架(LangChain / LlamaIndex)其实就是把上面这四步封装好了而已。
