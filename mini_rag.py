# -*- coding: utf-8 -*-
"""
极简 RAG(检索增强生成)—— 从零实现,不依赖任何框架
=====================================================
目标:用最少的代码讲清楚 RAG 的每一个环节。

RAG 的完整流程分两个阶段:
  【索引阶段-离线】 文档 -> 切分(chunk) -> 向量化(embedding) -> 存起来
  【查询阶段-在线】 问题 -> 向量化 -> 检索最相似片段 -> 拼 prompt -> 交给大模型生成

为了让你零配置就能跑起来看效果:
  - 向量化(embedding)这一步,我们用纯 Python 手写的 TF-IDF 实现,
    不调用任何外部 API,方便你看清"文本怎么变成数字、怎么算相似度"。
  - 生成这一步,如果你配置了 OpenAI 兼容的 API(DeepSeek/通义/智谱等),
    就真的调大模型;没配置就把"将要发给模型的 prompt"打印出来。
"""

import math
import os
import re
from collections import Counter

# =============================================================
# 第 0 步:准备知识库(真实项目里这些来自 PDF/Word/数据库/网页)
# =============================================================
DOCUMENTS = [
    "RAG 全称是检索增强生成。它先从知识库检索相关资料,再让大模型参考这些资料回答问题,从而减少幻觉。",
    "向量化(Embedding)是把一段文本转换成一串数字(向量)。语义相近的文本,它们的向量在空间里也更接近。",
    "向量数据库用于存储海量向量,并能快速找出与查询向量最相似的若干条,常用相似度度量是余弦相似度。",
    "Chunking 指把长文档切分成小片段。切太大会引入噪音、超出上下文;切太小又会丢失上下文,需要权衡。",
    "大模型(LLM)例如 GPT、Claude、DeepSeek,擅长根据给定上下文生成流畅的自然语言回答。",
    "Python 是一种解释型、动态类型的高级编程语言,因语法简洁、生态丰富,广泛用于 AI 与数据科学。",
]


# =============================================================
# 第 1 步:切分(Chunking)
# 这里文档本身已经很短,每条就是一个 chunk。真实场景要按长度/句子切。
# =============================================================
def split_into_chunks(documents):
    # 简单起见:一条文档 = 一个片段。返回 [(chunk_id, 文本), ...]
    return [(i, doc) for i, doc in enumerate(documents)]


# =============================================================
# 第 2 步:分词(把中英文文本拆成词/字的列表)
# 中文没有空格,这里做最朴素的处理:英文按单词、中文按单字切。
# 真实项目会用 jieba 等分词器,或直接用神经网络 embedding 模型。
# =============================================================
def tokenize(text):
    text = text.lower()
    tokens = []
    # 先抓出连续的英文字母/数字作为一个词
    for token in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text):
        tokens.append(token)
    return tokens


# =============================================================
# 第 3 步:向量化(Embedding)—— 手写 TF-IDF
# 核心思想:
#   TF (词频):某个词在一段文本里出现得越多,越能代表这段文本。
#   IDF(逆文档频率):一个词在越少的文档里出现,它区分度越高越重要。
#   一段文本的向量 = 每个词的 (TF * IDF) 组成的一串数字。
# 这就是最经典的、可解释的"把文本变成向量"的方法。
# =============================================================
def build_vocab_and_idf(chunks):
    # 统计每个词出现在多少篇文档里(document frequency)
    doc_freq = Counter()
    tokenized = []
    for _, text in chunks:
        toks = tokenize(text)
        tokenized.append(toks)
        for word in set(toks):          # 用 set:一篇文档里同一个词只算一次
            doc_freq[word] += 1

    n_docs = len(chunks)
    # 计算每个词的 IDF 值:文档总数 / 含该词的文档数,取对数平滑
    idf = {w: math.log((1 + n_docs) / (1 + df)) + 1 for w, df in doc_freq.items()}
    vocab = list(idf.keys())            # 词表:决定了向量每一维代表哪个词
    return vocab, idf, tokenized


def embed(tokens, vocab, idf):
    # 把一串分词后的 tokens 变成 TF-IDF 向量(一个 {词: 权重} 的稀疏字典)
    tf = Counter(tokens)
    total = len(tokens) or 1
    vector = {}
    for word, count in tf.items():
        if word in idf:                 # 只保留词表里有的词
            vector[word] = (count / total) * idf[word]   # TF * IDF
    return vector


# =============================================================
# 第 4 步:余弦相似度 —— 衡量两个向量方向有多接近
# 值域 [0,1](本例向量非负),越接近 1 越相似。
#   cos = (A·B) / (|A| * |B|)
# =============================================================
def cosine_similarity(vec_a, vec_b):
    # 点积:只需遍历较短的那个向量里出现的词
    dot = sum(weight * vec_b.get(word, 0.0) for word, weight in vec_a.items())
    norm_a = math.sqrt(sum(w * w for w in vec_a.values()))
    norm_b = math.sqrt(sum(w * w for w in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# =============================================================
# 第 5 步:检索(Retrieval)—— 找出与问题最相似的 top_k 个片段
# =============================================================
def retrieve(question, chunks, chunk_vectors, vocab, idf, top_k=2):
    q_vector = embed(tokenize(question), vocab, idf)   # 问题也向量化
    scored = []
    for (chunk_id, text), vec in zip(chunks, chunk_vectors):
        score = cosine_similarity(q_vector, vec)
        scored.append((score, chunk_id, text))
    scored.sort(reverse=True)                          # 按相似度从高到低排
    return scored[:top_k]                              # 取前 top_k 条


# =============================================================
# 第 6 步:组装 Prompt —— 把检索到的资料 + 问题拼成给模型的输入
# 这一步是 RAG 的"增强"所在:把私有知识塞进上下文。
# =============================================================
def build_prompt(question, retrieved):
    context = "\n".join(f"[资料{i+1}] {text}" for i, (_, _, text) in enumerate(retrieved))
    prompt = (
        "你是一个严谨的助手。请只根据下面提供的资料回答用户问题,"
        "如果资料中没有相关信息,就回答\"资料中未提及\"。\n\n"
        f"=== 资料 ===\n{context}\n\n"
        f"=== 用户问题 ===\n{question}\n\n"
        "=== 你的回答 ===\n"
    )
    return prompt


# =============================================================
# 第 7 步:生成(Generation)—— 交给大模型
# 若配置了环境变量 LLM_API_KEY 就真的调用 OpenAI 兼容接口;
# 否则只打印 prompt,让你看清 RAG 最终"喂"给模型的是什么。
# =============================================================
def generate(prompt):
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        return None  # 未配置,交给调用方打印 prompt
    # 用标准库发请求,连 openai 的 SDK 都不依赖
    import json
    import urllib.request
    base_url = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/chat/completions")
    model = os.environ.get("LLM_MODEL", "deepseek-chat")
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }).encode("utf-8")
    req = urllib.request.Request(
        base_url, data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


# =============================================================
# 主流程:把上面 7 步串起来
# =============================================================
def main():
    # ---- 索引阶段(离线,只需做一次)----
    chunks = split_into_chunks(DOCUMENTS)
    vocab, idf, tokenized = build_vocab_and_idf(chunks)
    chunk_vectors = [embed(toks, vocab, idf) for toks in tokenized]  # 每个片段的向量

    # ---- 查询阶段(在线,每次提问都走一遍)----
    question = "什么是向量化?为什么相似文本向量也接近?"
    print(f"用户问题: {question}\n")

    retrieved = retrieve(question, chunks, chunk_vectors, vocab, idf, top_k=2)
    print("检索到的最相关片段(按相似度排序):")
    for score, cid, text in retrieved:
        print(f"  相似度={score:.3f} | 片段#{cid}: {text}")
    print()

    prompt = build_prompt(question, retrieved)
    answer = generate(prompt)

    if answer is None:
        print("=" * 60)
        print("未检测到 LLM_API_KEY,以下是【将要发给大模型的完整 Prompt】:")
        print("=" * 60)
        print(prompt)
        print("提示:设置环境变量后即可获得真实回答,例如:")
        print('  export LLM_API_KEY="你的key"')
        print('  export LLM_BASE_URL="https://api.deepseek.com/chat/completions"')
        print('  export LLM_MODEL="deepseek-chat"')
    else:
        print("=" * 60)
        print("大模型基于检索资料生成的回答:")
        print("=" * 60)
        print(answer)


if __name__ == "__main__":
    main()
