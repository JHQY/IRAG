'''Retriever placeholder'''
"""
RAG 查询接口模块
为问答同学提供：简单调用即可使用向量知识检索。
"""

from embedding.embedder import Embedder
from storage.milvus_store import MilvusVectorStore
import numpy as np


class RAGInterface:
    """面向问答模块的 RAG 接口封装"""

    def __init__(self):
        print("🔗 初始化 RAG 接口组件...")
        self.embedder = Embedder()
        self.store = MilvusVectorStore()

    # ------------------------------------------------------
    # 基础搜索接口
    # ------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 5, filters: dict = None):
        """
        输入查询语句 -> 输出最相似文本块及元信息。
        参数：
          query: str —— 问题文本
          top_k: int —— 返回前多少条相似内容
          filters: dict —— 可选过滤条件，如 {"company": "AIA"}
        返回：
          List[{"text": str, "score": float, "metadata": dict}]
        """
        # 1️⃣ 嵌入 query
        try:
            q_emb = self.embedder.embed_query(query)
        except Exception as e:
            print(f"❌ 查询嵌入失败: {e}")  
            raw_emb = self.embedder.model.encode([query], convert_to_numpy=True, show_progress_bar=False)   
            q_emb = np.array(raw_emb, dtype=np.float32)[0]
        
        if isinstance(q_emb, np.ndarray) and q_emb.ndim >1:
            q_emb = q_emb[0]
       
        # 2️⃣ 相似检索
        hits = self.store.similarity_search(q_emb, top_k=top_k, filters=filters)

        # 3️⃣ 结构化输出
        results = []
        for chunk, score in hits:
            results.append({
                "text": chunk.text,
                "score": round(score, 4),
                "metadata": chunk.metadata
            })

        return results

    # ------------------------------------------------------
    # 高级接口（预留给 LLM 使用）
    # ------------------------------------------------------
    def retrieve_context(self, query: str, top_k: int = 5):
        """
        返回一个合并后的上下文字符串，可直接送入 LLM。
        """
        hits = self.retrieve(query, top_k=top_k)
        context = "\n---\n".join([f"{h['text']}" for h in hits])
        return context


# -------------------------
# 调试入口（可独立运行）
# -------------------------
if __name__ == "__main__":
    rag = RAGInterface()
    query = "怕出意外应该买哪个保险？"
    results = rag.retrieve(query, top_k=3)
    print("\n🔍 Top-3 结果:")
    for i, r in enumerate(results, 1):
        print(f"\n{i}. [score={r['score']}]")
        print(r["text"][:400], "...")
        print("metadata:", r["metadata"])
