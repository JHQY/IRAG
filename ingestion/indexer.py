# # ingestion/indexer.py
# from ingestion.loader import scan_documents
# from ingestion.parser import parse_pdf
# from ingestion.chunker import chunk_blocks
# from embedding.embedder import Embedder
# from storage.milvus_store import MilvusVectorStore, Chunk
# from config.settings import settings
# from tqdm import tqdm

# def build_index(source_dir="sourcepdf"):
#     """
#     构建保险知识库索引：
#     1. 扫描所有文件
#     2. 抽取文字 + 表格（带上下文）
#     3. 对文字内容分块
#     4. 嵌入 + 写入 Milvus
#     """
#     print("🚀 开始构建索引 ...")
#     docs = scan_documents(source_dir)
#     if not docs:
#         print("⚠️ 没有找到可索引的文件。")
#         return

#     embedder = Embedder()
#     store = MilvusVectorStore()
#     total_chunks = 0

#     for doc in tqdm(docs, desc="索引进度"):
#         try:
#             parsed_blocks = parse_pdf(doc["path"])
#             # print(f"📄 解析完成：{doc['path']}，提取到 {len(parsed_blocks)} 个内容块。")
#             # print(f"预览内容块：{parsed_blocks[:2]}")  # 打印前两个内容块以供调试
#             if not parsed_blocks:
#                 print(f"⚠️ 文件无有效内容：{doc['path']}")
#                 continue

#             for block in parsed_blocks:
#                 # content = block.get("text", "").strip()
#                 # modality = block.get("modality", "text")
#                 # page = block.get("page", 0)
                

#                 # 跳过空块
#                 if not content:
#                     continue

#                 # 仅文本进行分块；表格保持整块
#                 if modality == "text":
#                     chunks = chunk_blocks(parsed_blocks)
#                 else:
#                     chunks = [content]
                
#                 for c in chunks:
#                     emb = embedder.embed_text([c])[0]
#                     meta = {
#                         **doc["metadata"],
#                         "page": page,
#                         "modality": modality
#                     }
#                     store.add([emb], [Chunk(c, meta)])
#                     total_chunks += 1

#         except Exception as e:
#             print(f"❌ 文件处理失败: {doc['path']} ({e})")

#     print(f"✅ 索引完成，共写入 {total_chunks} 个文本块。")

from ingestion.loader import scan_documents
from ingestion.parser import parse_pdf
from ingestion.chunker import chunk_blocks
from embedding.embedder import Embedder
from storage.milvus_store import MilvusVectorStore, Chunk
from tqdm import tqdm
import time

def build_index(source_dir="sourcepdf"):
    """
    构建保险知识库索引：
    1. 扫描所有文件
    2. 抽取文字 + 表格（带上下文）
    3. 对文字内容分块
    4. 嵌入 + 写入 Milvus
    """
    print("🚀 开始构建索引 ...")
    docs = scan_documents(source_dir)
    if not docs:
        print("⚠️ 没有找到可索引的文件。")
        return

    embedder = Embedder()
    store = MilvusVectorStore()
    total_chunks = 0
    batch_chunks = []
    batch_texts = []
    batch_embs = []
    batch_size = 500  # 每批处理的文本块数量

    for doc in tqdm(docs, desc="索引进度"):
        try:
            parsed_blocks = parse_pdf(doc["path"])
            if not parsed_blocks:
                print(f"⚠️ 文件无有效内容：{doc['path']}")
                continue

            # 给每个 block 添加 metadata
            for b in parsed_blocks:
                b.setdefault("metadata", {})
                b["metadata"].update({
                    "source": doc.get("path", ""),
                    "company": doc.get("company", ""),
                    "category": doc.get("category", ""),
                    "page_number": b.get("page_number", None),
                    "modality": b.get("modality", "text")
                })

            chunks = chunk_blocks(parsed_blocks, max_len=500, overlap=50)

            # ✅ 嵌入并写入 Milvus
            for c in chunks:
                text = c.get("text", "").strip()
                if not text:
                    continue
                meta = c.get("metadata", {})
                # emb = embedder.embed_text([text])[0]
                # store.add([emb], [Chunk(text, meta)])
                # total_chunks += 1
                batch_chunks.append(Chunk(text, meta))
                batch_texts.append(text)

                if len(batch_chunks) >= batch_size:
                    _flush_batch(store, embedder, batch_chunks, batch_texts)
                    total_chunks += len(batch_chunks)
                    batch_chunks = []
                    batch_texts = []
        except Exception as e:
            print(f"❌ 文件处理失败: {doc['path']} ({e})")
    # 处理剩余的批次
    if batch_chunks:
        _flush_batch(store, embedder, batch_chunks, batch_texts)
        total_chunks += len(batch_chunks)

    print(f"✅ 索引完成，共写入 {total_chunks} 个文本块。")

def _flush_batch(store, embedder, batch_chunks, batch_texts):
    try:
        start = time.time()
        embeddings = embedder.embed_text(batch_texts)
        store.add(embeddings, batch_chunks)
        cost = time.time() - start
        print(f"[批次写入] ✅ 写入 {len(batch_chunks)} 个文本块，耗时 {cost:.2f} 秒。")
    except Exception as e:
        print(f"❌ 批次写入失败: {e}")  