"""
文本分块器 - 智能分割文本以适应向量化
"""

import re
from typing import List, Dict, Optional, TYPE_CHECKING
from dataclasses import dataclass

# 避免循环导入
if TYPE_CHECKING:
    pass


@dataclass
class TextChunk:
    """文本块"""
    text: str
    chunk_id: int
    source_file: str
    page_number: Optional[int] = None
    metadata: Optional[Dict] = None


class TextChunker:
    """文本分块器"""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separator: str = "\n\n"
    ):
        """
        初始化分块器

        Args:
            chunk_size: 每块的最大字符数
            chunk_overlap: 块之间的重叠字符数
            separator: 分隔符
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def chunk(
        self,
        text: str,
        source_file: str,
        pages: Optional[List[str]] = None
    ) -> List[TextChunk]:
        """
        将文本分割成块

        Args:
            text: 完整文本
            source_file: 源文件路径
            pages: 按页分割的文本列表（可选）

        Returns:
            TextChunk列表
        """
        if pages:
            return self._chunk_by_pages(text, pages, source_file)
        else:
            return self._chunk_by_size(text, source_file)

    def _chunk_by_pages(
        self,
        text: str,
        pages: List[str],
        source_file: str
    ) -> List[TextChunk]:
        """按页/段落分块"""
        chunks = []
        chunk_id = 0

        for page_num, page_text in enumerate(pages, 1):
            if not page_text.strip():
                continue

            # 如果单页过长，进一步分割
            if len(page_text) > self.chunk_size:
                sub_chunks = self._split_text(page_text)
                for i, sub_chunk in enumerate(sub_chunks):
                    chunks.append(TextChunk(
                        text=sub_chunk,
                        chunk_id=chunk_id,
                        source_file=source_file,
                        page_number=page_num,
                        metadata={"sub_chunk": i, "total_sub_chunks": len(sub_chunks)}
                    ))
                    chunk_id += 1
            else:
                chunks.append(TextChunk(
                    text=page_text,
                    chunk_id=chunk_id,
                    source_file=source_file,
                    page_number=page_num
                ))
                chunk_id += 1

        return chunks

    def _chunk_by_size(self, text: str, source_file: str) -> List[TextChunk]:
        """按固定大小分块"""
        chunks = []
        chunk_id = 0

        sub_chunks = self._split_text(text)

        for i, chunk_text in enumerate(sub_chunks):
            chunks.append(TextChunk(
                text=chunk_text,
                chunk_id=chunk_id,
                source_file=source_file,
                metadata={"sub_chunk": i, "total_sub_chunks": len(sub_chunks)}
            ))
            chunk_id += 1

        return chunks

    def _split_text(self, text: str) -> List[str]:
        """
        分割文本，考虑重叠

        策略:
        1. 先按分隔符（段落）分割
        2. 合并段落直到达到chunk_size
        3. 保持overlap
        """
        # 按分隔符分割
        splits = text.split(self.separator)
        splits = [s.strip() for s in splits if s.strip()]

        if not splits:
            return [text]

        chunks = []
        current_chunk = ""
        current_size = 0

        for i, split in enumerate(splits):
            split_size = len(split)

            # 如果单个片段就超过chunk_size，强制分割
            if split_size > self.chunk_size:
                # 先保存当前chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # 分割大片段
                sub_splits = self._force_split(split)
                chunks.extend(sub_splits[:-1])  # 除了最后一段

                # 最后一段作为新chunk的开始
                current_chunk = sub_splits[-1]
                current_size = len(sub_splits[-1])
            else:
                # 检查加上这个split是否会超过chunk_size
                if current_size + split_size + len(self.separator) > self.chunk_size:
                    # 保存当前chunk
                    if current_chunk:
                        chunks.append(current_chunk.strip())

                    # 创建新chunk，考虑overlap
                    if self.chunk_overlap > 0 and current_chunk:
                        overlap_text = self._get_overlap_text(current_chunk)
                        current_chunk = overlap_text + self.separator + split
                        current_size = len(current_chunk)
                    else:
                        current_chunk = split
                        current_size = split_size
                else:
                    # 添加到当前chunk
                    if current_chunk:
                        current_chunk += self.separator + split
                    else:
                        current_chunk = split
                    current_size = current_size + split_size + len(self.separator)

        # 添加最后一个chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _force_split(self, text: str) -> List[str]:
        """强制分割长文本"""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            if end >= len(text):
                chunks.append(text[start:].strip())
                break

            # 尝试在句子边界分割
            split_pos = self._find_sentence_boundary(text, start, end)

            chunks.append(text[start:split_pos].strip())
            start = split_pos

        return chunks

    def _find_sentence_boundary(self, text: str, start: int, end: int) -> int:
        """在chunk_size附近寻找句子边界"""
        # 常见的句子结束符号
        sentence_ends = ['。', '！', '？', '.', '!', '?', '\n']

        # 从end向前搜索最近的句子结束符
        search_start = max(start, end - 100)

        for i in range(end, search_start, -1):
            if i < len(text) and text[i] in sentence_ends:
                return i + 1

        # 如果找不到，就在end处分割
        return end

    def _get_overlap_text(self, text: str) -> str:
        """获取文本末尾的overlap部分"""
        if len(text) <= self.chunk_overlap:
            return text

        # 尝试在句子边界分割
        overlap_start = len(text) - self.chunk_overlap

        # 寻找句子边界
        for i in range(overlap_start, len(text)):
            if i > 0 and text[i-1:i] in ['。', '！', '？', '. ', '! ', '? ', '\n']:
                return text[i:].strip()

        return text[overlap_start:]

    def chunk_documents(self, documents: List[Dict]) -> List[TextChunk]:
        """
        批量分块多个文档

        Args:
            documents: 文档解析结果列表

        Returns:
            所有文本块列表
        """
        all_chunks = []

        for doc in documents:
            chunks = self.chunk(
                text=doc['text'],
                source_file=doc['metadata']['filepath'],
                pages=doc.get('pages')
            )
            all_chunks.extend(chunks)

        return all_chunks


if __name__ == "__main__":
    # 测试
    chunker = TextChunker(chunk_size=200, chunk_overlap=50)

    text = """
    这是第一段内容，用来测试文本分块功能。
    文本分块是向量化之前的重要步骤。

    这是第二段内容。我们需要确保分块后的文本保持语义完整性。

    这是第三段内容，应该会被分割到不同的chunk中。
    同时也要考虑chunk之间的overlap，避免信息丢失。

    这是第四段内容，测试overlap功能是否正常工作。
    """

    chunks = chunker._split_text(text)

    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} ---")
        print(chunk)
        print()
