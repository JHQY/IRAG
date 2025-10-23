'''Chunker for splitting parsed text blocks into smaller segments'''
from typing import List, Dict

def chunk_blocks(blocks: List[Dict], max_len: int = 500, overlap: int = 50):
    """
    Split text blocks into smaller chunks with overlap.
    Each output chunk preserves metadata such as company/category/page number.
    """

    chunks = []

    for block in blocks:
        text = block.get("text", "").strip()
        if not text:
            continue

        metadata = block.get("metadata", {})
        # 自动从上层路径结构补齐公司与险种（如果存在）
        if not metadata:
            source = block.get("source", "")
            if "/" in source or "\\" in source:
                parts = source.replace("\\", "/").split("/")
                if len(parts) >= 3:
                    metadata = {
                        "source": source,
                        "company": parts[-3],
                        "category": parts[-2],
                        "page_number": block.get("page_number", None)
                    }
                else:
                    metadata = {"source": source}
            else:
                metadata = {"source": source}

        # 分词逻辑：优先按空格切分（英语）；若文本无空格，则按字符切
        words = text.split() if " " in text else list(text)
        if len(words) <= max_len:
            chunks.append({"text": text, "metadata": metadata})
            continue

        # 滑动窗口切分
        start = 0
        while start < len(words):
            end = min(start + max_len, len(words))
            piece = " ".join(words[start:end]) if " " in text else "".join(words[start:end])
            chunks.append({
                "text": piece,
                "metadata": metadata
            })
            start += max_len - overlap

    return chunks

def chunk_text(text: str, max_len: int = 500, overlap: int = 50) -> List[str]:
    """
    Split a single text string into smaller chunks with overlap.
    """
    chunks = []
    words = text.split() if " " in text else list(text)
    if len(words) <= max_len:
        return [text]

    start = 0
    while start < len(words):
        end = min(start + max_len, len(words))
        piece = " ".join(words[start:end]) if " " in text else "".join(words[start:end])
        chunks.append(piece)
        start += max_len - overlap

    return chunks
