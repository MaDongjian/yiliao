"""
向量存储 - 使用 FAISS 进行向量索引和检索
移除 SQLite，改用内存存储 + JSON 文件持久化
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np


class VectorStore:
    """向量存储 - FAISS索引 + JSON元数据"""

    def __init__(
        self,
        index_dir: str = "./data/index",
        dimension: int = 384
    ):
        """
        初始化向量存储

        Args:
            index_dir: FAISS索引文件目录
            dimension: 向量维度
        """
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / "faiss.index"
        self.metadata_path = self.index_dir / "metadata.json"
        self.dimension = dimension

        # 确保目录存在
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # 内存中存储元数据
        self.chunks_metadata = []  # 存储所有文本块元数据
        self.documents_metadata = {}  # 存储文档元数据 {doc_id: metadata}

        # FAISS索引
        self.index = None
        self._load_or_create_index()

    def _load_or_create_index(self):
        """加载或创建FAISS索引"""
        try:
            import faiss
        except ImportError:
            raise ImportError("请安装 faiss-cpu: pip install faiss-cpu")

        # 加载元数据
        self._load_metadata()

        if self.index_path.exists():
            # 加载现有索引
            self.index = faiss.read_index(str(self.index_path))
            print(f"已加载现有索引，包含 {self.index.ntotal} 个向量")
        else:
            # 创建新索引
            self.index = faiss.IndexFlatL2(self.dimension)
            print(f"创建新索引，维度: {self.dimension}")

    def _load_metadata(self):
        """从JSON文件加载元数据（兼容旧版数组格式和新版对象格式）"""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # 兼容旧版数组格式 [...]
                if isinstance(data, list):
                    self.chunks_metadata = data
                    self.documents_metadata = {}
                    print(f"已加载 {len(self.chunks_metadata)} 个文本块元数据（数组格式）")
                # 新版对象格式 {"chunks": [...], "documents": {}}
                elif isinstance(data, dict):
                    self.chunks_metadata = data.get('chunks', [])
                    self.documents_metadata = data.get('documents', {})
                    print(f"已加载 {len(self.chunks_metadata)} 个文本块元数据")
                else:
                    self.chunks_metadata = []
                    self.documents_metadata = {}
                    print(f"警告: 元数据格式不正确")

    def _save_metadata(self):
        """保存元数据到JSON文件"""
        data = {
            'chunks': self.chunks_metadata,
            'documents': self.documents_metadata
        }
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def save_index(self):
        """保存FAISS索引和元数据到磁盘"""
        import faiss
        faiss.write_index(self.index, str(self.index_path))
        self._save_metadata()
        print(f"索引已保存到: {self.index_path}")
        print(f"元数据已保存到: {self.metadata_path}")

    def add_chunks(
        self,
        chunks: List,
        embeddings: np.ndarray,
        doc_metadata: Dict,
        summary: str = None
    ) -> int:
        """
        添加文本块和向量到存储

        Args:
            chunks: TextChunk对象列表
            embeddings: 向量数组
            doc_metadata: 文档元数据
            summary: 文档概要（可选）

        Returns:
            文档ID
        """
        import faiss

        if len(chunks) != len(embeddings):
            raise ValueError(f"文本块数量 {len(chunks)} 与向量数量 {len(embeddings)} 不匹配")

        # 生成文档ID：取已有最大ID + 1，避免追加时ID重复
        if self.documents_metadata:
            doc_id = max(int(k) for k in self.documents_metadata.keys()) + 1
        else:
            doc_id = 1

        # 保存文档元数据（包含概要）
        self.documents_metadata[doc_id] = {
            'id': doc_id,
            'filename': doc_metadata.get('filename', ''),
            'filepath': doc_metadata.get('filepath', ''),
            'format': doc_metadata.get('format', ''),
            'size': doc_metadata.get('size', 0),
            'summary': summary  # 添加概要字段
        }

        # 添加向量到FAISS索引
        start_idx = self.index.ntotal
        self.index.add(embeddings.astype('float32'))

        # 保存文本块元数据
        for i, chunk in enumerate(chunks):
            vector_index = start_idx + i
            self.chunks_metadata.append({
                'doc_id': doc_id,
                'chunk_id': chunk.chunk_id,
                'text': chunk.text,
                'page_number': chunk.page_number,
                'metadata': chunk.metadata,
                'vector_index': vector_index,
                'filename': doc_metadata.get('filename', ''),
                'filepath': doc_metadata.get('filepath', '')
            })

        # 保存元数据
        self._save_metadata()

        print(f"已添加 {len(chunks)} 个文本块，文档ID: {doc_id}")

        return doc_id

    def add_documents(
        self,
        documents: List[Dict],
        chunks_list: List[List],
        embeddings_list: List[np.ndarray]
    ) -> List[int]:
        """
        批量添加文档

        Args:
            documents: 文档元数据列表
            chunks_list: 每个文档的文本块列表
            embeddings_list: 每个文档的向量列表

        Returns:
            文档ID列表
        """
        doc_ids = []

        for doc_meta, chunks, embeddings in zip(documents, chunks_list, embeddings_list):
            doc_id = self.add_chunks(chunks, embeddings, doc_meta['metadata'])
            doc_ids.append(doc_id)

        # 保存索引
        self.save_index()

        return doc_ids

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        min_score: float = None
    ) -> List[Dict]:
        """
        向量搜索

        Args:
            query_embedding: 查询向量
            top_k: 返回前K个结果
            min_score: 最小相似度阈值（可选）

        Returns:
            搜索结果列表
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # FAISS搜索（返回L2距离）
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)

        # 转换L2距离为相似度分数
        results = []

        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # 无效索引
                continue

            # 计算相似度分数
            score = 1 / (1 + dist)

            # 过滤低分结果
            if min_score and score < min_score:
                continue

            # 获取元数据
            if idx < len(self.chunks_metadata):
                chunk_meta = self.chunks_metadata[idx]
                results.append({
                    'text': chunk_meta['text'],
                    'score': float(score),
                    'filename': chunk_meta['filename'],
                    'filepath': chunk_meta['filepath'],
                    'page_number': chunk_meta.get('page_number'),
                    'metadata': chunk_meta.get('metadata')
                })

        return results

    def keyword_search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        关键词搜索（遍历内存中的文本）

        Args:
            query: 关键词
            top_k: 返回前K个结果

        Returns:
            搜索结果列表
        """
        query_lower = query.lower()
        results = []
        seen_files = set()  # 用于去重，每个文件只返回最相关的结果

        for chunk_meta in self.chunks_metadata:
            # 检查是否包含关键词
            if query_lower in chunk_meta['text'].lower():
                filename = chunk_meta['filename']

                # 如果该文件已经有结果，跳过（保持每个文件只返回一个结果）
                if filename in seen_files:
                    continue

                seen_files.add(filename)
                results.append({
                    'text': chunk_meta['text'],
                    'score': 1.0,  # 关键词匹配给满分
                    'filename': filename,
                    'filepath': chunk_meta['filepath'],
                    'page_number': chunk_meta.get('page_number'),
                    'metadata': chunk_meta.get('metadata')
                })

                if len(results) >= top_k:
                    break

        return results

    def get_stats(self) -> Dict:
        """获取存储统计信息"""
        # 统计文档数量
        doc_count = len(self.documents_metadata)

        # 统计文本块数量
        chunk_count = len(self.chunks_metadata)

        # 向量数量
        vector_count = self.index.ntotal

        return {
            'document_count': doc_count,
            'chunk_count': chunk_count,
            'vector_count': vector_count,
            'index_path': str(self.index_path),
            'metadata_path': str(self.metadata_path)
        }

    def clear(self):
        """清空所有数据"""
        import faiss

        # 重建索引
        self.index = faiss.IndexFlatL2(self.dimension)

        # 清空元数据
        self.chunks_metadata = []
        self.documents_metadata = {}

        # 删除文件
        if self.index_path.exists():
            self.index_path.unlink()
        if self.metadata_path.exists():
            self.metadata_path.unlink()

        print("已清空所有数据")


if __name__ == "__main__":
    # 测试代码
    store = VectorStore(dimension=384)

    # 打印统计信息
    stats = store.get_stats()
    print("存储统计:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
