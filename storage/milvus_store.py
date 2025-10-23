# storage/milvus_store.py
from pymilvus import (
    connections, FieldSchema, CollectionSchema,
    DataType, Collection, utility
)
from config.settings import settings
import numpy as np

class Chunk:
    """一个文本或表格块"""
    def __init__(self, text, metadata):
        self.text = text
        self.metadata = metadata


class MilvusVectorStore:
    """
    Milvus 向量存储与检索类
    - 自动连接 Milvus
    - 自动创建 collection
    - 提供 add / search 功能
    """

    def __init__(self):
        self.collection_name = settings.MILVUS_COLLECTION
        self.dim = settings.MILVUS_DIM

        # 连接 Milvus
        connections.connect(
            alias="default",
            host=settings.MILVUS_HOST,
            port=str(settings.MILVUS_PORT)
        )

        # 检查 collection 是否存在
        if not utility.has_collection(self.collection_name):
            self._create_collection()

        # 加载 collection
        self.collection = Collection(self.collection_name)
        self.collection.load()

    # ------------------------------------------------------
    # 创建 collection
    # ------------------------------------------------------
    def _create_collection(self):
        print(f"[Milvus] Creating collection: {self.collection_name}")

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]

        schema = CollectionSchema(
            fields=fields,
            description="Insurance Knowledge Base"
        )

        collection = Collection(name=self.collection_name, schema=schema)

        # 创建索引
        index_params = {
            "index_type": settings.MILVUS_INDEX_TYPE,
            "metric_type": settings.MILVUS_METRIC_TYPE,
            "params": {"M": 8, "efConstruction": 64}
        }

        collection.create_index(field_name="vector", index_params=index_params)
        print(f"[Milvus] Collection `{self.collection_name}` created with index.")
        return collection

    # ------------------------------------------------------
    # 插入数据
    # ------------------------------------------------------
    def add(self, embeddings, chunks):
        """
        向 Milvus 插入一批数据
        参数：
          embeddings: List[np.ndarray]  向量
          chunks: List[Chunk]           对应文本块
        """
        if len(embeddings) == 0:
            return

        texts = [c.text for c in chunks]
        metas = [c.metadata for c in chunks]

        # 插入顺序必须与 collection 定义匹配
        insert_data = [
            #[None] * len(embeddings),  # auto_id 主键
            embeddings,
            texts,
            metas
        ]

        self.collection.insert(insert_data)
        self.collection.flush()
        print(f"[Milvus] ✅ Inserted {len(embeddings)} records.")

    # ------------------------------------------------------
    # 向量检索
    # ------------------------------------------------------
    def similarity_search(self, query_embedding, top_k=5, filters=None):
        """
        执行相似度搜索
        参数：
          query_embedding: np.ndarray
          top_k: 检索结果数
          filters: dict，可按 metadata 过滤
        返回：
          [(Chunk, distance), ...]
        """
        search_params = {
            "metric_type": settings.MILVUS_METRIC_TYPE,
            "params": {"ef": 50}
        }

        expr = None
        if filters:
            expr = " and ".join([
                f'metadata["{k}"] == "{v}"' for k, v in filters.items()
            ])

        results = self.collection.search(
            data=[query_embedding],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["text", "metadata"]
        )

        hits = []
        for hit in results[0]:
            text = hit.entity.get("text")
            meta = hit.entity.get("metadata")
            hits.append((Chunk(text, meta), float(hit.distance)))

        return hits
