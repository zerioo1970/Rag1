# -*- coding: utf-8 -*-
"""
用 LlamaIndex 实现的 RAG 示例(本地免费模型,无需 API Key)
============================================================

对比 rag_demo.py(手写版,~220 行),这个版本用框架把
"切分 / 向量化 / 建索引 / 检索" 全部封装,核心只需几行。

说明:
  - Embedding(向量化)用本地开源模型 BAAI/bge-small-zh-v1.5(首次运行会自动下载约 100MB)
  - 不接大模型(LLM),所以"生成"这一步用检索结果直接组装;
    文件末尾注释了如何换成真正的大模型生成。

运行前先安装依赖:
    pip install llama-index-core llama-index-embeddings-huggingface
"""

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# ============================================================
# 0. 知识库(和手写版一致,方便对比)
# ============================================================
TEXTS = [
    "RAG 的全称是检索增强生成(Retrieval-Augmented Generation),"
    "由 Meta 的研究者在 2020 年的论文中首次提出。"
    "核心思想是:在向大模型提问前,先从外部知识库检索相关资料,再让模型基于资料作答。",

    "RAG 最大的好处是减少幻觉。因为答案有真实资料作为依据,而不是让模型凭记忆瞎编。"
    "同时它还能让模型用上训练时没见过的私有数据和最新数据。",

    "向量数据库是 RAG 系统的关键组件,用来存储文本的向量并支持快速相似度检索。"
    "常见的向量数据库有 Pinecone、Milvus、Chroma,以及 PostgreSQL 的 pgvector 扩展。",

    "Embedding(嵌入)是把文本转换成一组数字(向量)的过程。"
    "语义相近的文本,它们的向量在空间中也更接近,因此可以用向量距离来衡量语义相似度。",

    "切分(Chunking)是指把长文档拆成较小的文本块。"
    "块太大检索不精准,块太小又会丢失上下文,所以块的大小需要权衡。",

    "做 RAG 开发最常用的语言是 Python,常用框架包括 LangChain、LlamaIndex 和 Haystack。",
]


def main():
    # --------------------------------------------------------
    # 关键设置:用本地免费的 embedding 模型,并关闭 LLM
    # 这样整个流程不需要任何 API Key
    # --------------------------------------------------------
    print("加载本地 embedding 模型(首次会下载约 100MB)...")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")
    Settings.llm = None  # 不接大模型,只做检索

    # --------------------------------------------------------
    # 核心:一行完成 切分 + 向量化 + 建索引
    # (对比手写版的 split_into_chunks + build_tfidf 几十行)
    # --------------------------------------------------------
    documents = [Document(text=t) for t in TEXTS]
    index = VectorStoreIndex.from_documents(documents)
    print("索引构建完成。\n")

    # 创建检索器:检索语义最相关的 top_k 个块
    retriever = index.as_retriever(similarity_top_k=2)

    questions = [
        "RAG 为什么能减少幻觉?",
        "常见的向量数据库有哪些?",
        "做 RAG 用什么编程语言?",
        # 故意不含"幻觉"二字,考验语义检索能力(TF-IDF 版可能答不准)
        "怎样防止模型胡编乱造?",
    ]

    for q in questions:
        print("=" * 60)
        print(f"用户提问:{q}")

        # 检索(LlamaIndex 帮你把问题向量化 + 算相似度 + 排序)
        nodes = retriever.retrieve(q)

        print("检索到最相关的内容:")
        for n in nodes:
            print(f"  相似度 {n.score:.3f} | {n.node.get_content()[:50]}...")

        # 生成(这里用检索结果直接组装;真实场景交给大模型)
        answer = f"根据资料,{nodes[0].node.get_content()}" if nodes else "(没有找到相关资料)"
        print("【最终回答】", answer)


if __name__ == "__main__":
    main()


# ============================================================
# 如何换成"真正的大模型生成"?
# ============================================================
# 只需把上面的"检索 + 手动组装"换成 LlamaIndex 的查询引擎:
#
#   方式 A:用 OpenAI(需要 API Key)
#       import os
#       os.environ["OPENAI_API_KEY"] = "你的key"
#       # 删掉 Settings.llm = None 这一行(默认就会用 OpenAI 的 GPT)
#       query_engine = index.as_query_engine()
#       print(query_engine.query("常见的向量数据库有哪些?"))
#
#   方式 B:用本地开源大模型(如 Ollama,需先在本机启动 ollama)
#       from llama_index.llms.ollama import Ollama
#       Settings.llm = Ollama(model="qwen2.5")
#       query_engine = index.as_query_engine()
#       print(query_engine.query("常见的向量数据库有哪些?"))
#
# 对比 rag_demo.py 手写版,你会发现:框架把切分、向量化、检索、
# 生成全封装好了,代价是它是"黑盒",且依赖外部模型/服务。
