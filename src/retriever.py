"""
文档检索器 - 整合所有模块，提供统一的检索接口
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Union
from tqdm import tqdm

# 兼容相对导入和绝对导入
try:
    from .parsers import DocumentParser, scan_documents
    from .chunker import TextChunker, TextChunk
    from .embedder import EmbeddingModel
    from .storage import VectorStore
except ImportError:
    from parsers import DocumentParser, scan_documents
    from chunker import TextChunker, TextChunk
    from embedder import EmbeddingModel
    from storage import VectorStore


class DocumentRetriever:
    """文档检索系统主类"""

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        index_dir: str = "./data/index",
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        """
        初始化文档检索系统

        Args:
            model_name: 向量化模型名称
            index_dir: 索引目录
            chunk_size: 分块大小
            chunk_overlap: 分块重叠
        """
        self.parser = DocumentParser()
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.embedder = EmbeddingModel(model_name=model_name)

        # 向量存储（在embedder加载后初始化）
        self.storage = None

        self.index_dir = index_dir

    def initialize(self):
        """初始化系统（加载模型和索引）"""
        print("正在初始化文档检索系统...")

        # 加载向量化模型
        self.embedder.load()

        # 初始化向量存储
        dimension = self.embedder.get_dimension()
        self.storage = VectorStore(
            index_dir=self.index_dir,
            dimension=dimension
        )

        print("系统初始化完成！")

    def add_document(self, file_path: str, generate_summary: bool = True) -> Dict:
        """
        添加单个文档到索引

        Args:
            file_path: 文档路径
            generate_summary: 是否生成文档概要（默认True）

        Returns:
            处理结果
        """
        if self.storage is None:
            self.initialize()

        print(f"\n正在处理文档: {file_path}")

        # 解析文档
        print("  [1/5] 解析文档...")
        doc = self.parser.parse(file_path)

        # 分块
        print("  [2/5] 文本分块...")
        chunks = self.chunker.chunk(
            text=doc['text'],
            source_file=doc['metadata']['filepath'],
            pages=doc.get('pages')
        )
        print(f"         分割为 {len(chunks)} 个文本块")

        # 生成概要
        summary = None
        if generate_summary:
            print("  [3/5] 生成文档概要...")
            try:
                # 兼容相对导入和绝对导入
                try:
                    from .summary import DocumentSummaryGenerator
                except ImportError:
                    from summary import DocumentSummaryGenerator

                summary_gen = DocumentSummaryGenerator()
                summary_result = summary_gen.generate_summary_for_chunks(
                    chunks=chunks,
                    filename=doc['metadata']['filename'],
                    save_to_file=True
                )
                summary = summary_result['summary']
            except Exception as e:
                print(f"         概要生成失败: {e}")
                summary = None
        else:
            print("  [3/5] 跳过概要生成...")

        # 向量化
        print("  [4/5] 生成向量...")
        embeddings = self.embedder.encode_chunks(chunks, show_progress=False)

        # 存储索引
        print("  [5/5] 存储索引...")
        doc_id = self.storage.add_chunks(chunks, embeddings, doc['metadata'], summary)

        # 保存索引
        self.storage.save_index()

        return {
            'doc_id': doc_id,
            'filename': doc['metadata']['filename'],
            'chunk_count': len(chunks),
            'summary': summary
        }

    def add_directory(self, directory: str, extensions: Optional[List[str]] = None) -> Dict:
        """
        添加目录中的所有文档

        Args:
            directory: 目录路径
            extensions: 文件扩展名列表

        Returns:
            处理结果统计
        """
        if self.storage is None:
            self.initialize()

        # 扫描文档
        if extensions is None:
            extensions = ['.pdf', '.docx', '.pptx']

        print(f"\n正在扫描目录: {directory}")
        file_paths = scan_documents(directory, extensions)

        if not file_paths:
            print("未找到任何文档")
            return {'total': 0, 'success': 0, 'failed': 0}

        print(f"找到 {len(file_paths)} 个文档")

        # 批量处理
        results = {
            'total': len(file_paths),
            'success': 0,
            'failed': 0,
            'files': []
        }

        for file_path in tqdm(file_paths, desc="处理文档"):
            try:
                result = self.add_document(file_path)
                results['success'] += 1
                results['files'].append({
                    'file': file_path,
                    'status': 'success',
                    'doc_id': result['doc_id']
                })
            except Exception as e:
                results['failed'] += 1
                results['files'].append({
                    'file': file_path,
                    'status': 'failed',
                    'error': str(e)
                })
                print(f"\n警告: 处理失败 {file_path}: {e}")

        # 打印统计
        print(f"\n处理完成:")
        print(f"  总数: {results['total']}")
        print(f"  成功: {results['success']}")
        print(f"  失败: {results['failed']}")

        return results

    def search(
        self,
        query: str,
        top_k: int = 5,
        method: str = "semantic",
        min_score: float = None
    ) -> List[Dict]:
        """
        搜索文档

        Args:
            query: 查询文本
            top_k: 返回前K个结果
            method: 搜索方法 ("semantic" 语义搜索, "keyword" 关键词搜索)
            min_score: 最小相似度分数

        Returns:
            搜索结果列表
        """
        if self.storage is None:
            self.initialize()

        if method == "keyword":
            # 关键词搜索
            results = self.storage.keyword_search(query, top_k=top_k)
        else:
            # 语义搜索
            query_embedding = self.embedder.encode(query)
            results = self.storage.search(
                query_embedding,
                top_k=top_k,
                min_score=min_score
            )

        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[Dict]:
        """
        混合搜索（语义 + 关键词）

        Args:
            query: 查询文本
            top_k: 返回前K个结果
            semantic_weight: 语义搜索权重
            keyword_weight: 关键词搜索权重

        Returns:
            搜索结果列表
        """
        if self.storage is None:
            self.initialize()

        # 语义搜索
        query_embedding = self.embedder.encode(query)
        semantic_results = self.storage.search(query_embedding, top_k=top_k * 2)

        # 关键词搜索
        keyword_results = self.storage.keyword_search(query, top_k=top_k * 2)

        # 合并结果
        combined = {}

        # 添加语义搜索结果
        for result in semantic_results:
            key = (result['filepath'], result['text'][:50])  # 使用唯一键
            combined[key] = {
                **result,
                'semantic_score': result['score'],
                'keyword_score': 0
            }

        # 添加关键词搜索结果并合并分数
        for result in keyword_results:
            key = (result['filepath'], result['text'][:50])
            if key in combined:
                combined[key]['keyword_score'] = result['score']
            else:
                combined[key] = {
                    **result,
                    'semantic_score': 0,
                    'keyword_score': result['score']
                }

        # 计算综合分数
        for key in combined:
            combined[key]['score'] = (
                combined[key]['semantic_score'] * semantic_weight +
                combined[key]['keyword_score'] * keyword_weight
            )

        # 排序并返回top_k
        results = sorted(combined.values(), key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def get_stats(self) -> Dict:
        """获取系统统计信息"""
        if self.storage is None:
            self.initialize()

        return self.storage.get_stats()

    def clear(self):
        """清空所有索引"""
        if self.storage is None:
            self.initialize()
        self.storage.clear()


def format_results(results: List[Dict], show_score: bool = True) -> str:
    """格式化搜索结果用于显示"""
    if not results:
        return "未找到相关结果"

    output = []
    output.append(f"找到 {len(results)} 个相关结果:\n")

    for i, result in enumerate(results, 1):
        output.append(f"[{i}] {result['filename']}")
        if show_score:
            output.append(f"    相似度: {result['score']:.4f}")
        if result.get('page_number'):
            output.append(f"    页码: {result['page_number']}")
        output.append(f"    文本: {result['text'][:200]}...")
        output.append("")

    return "\n".join(output)


if __name__ == "__main__":
    # 测试代码
    retriever = DocumentRetriever()
    retriever.initialize()

    # 添加文档
    # retriever.add_document("test.docx")

    # 搜索
    # results = retriever.search("机器学习", top_k=3)
    # print(format_results(results))

    # 统计信息
    stats = retriever.get_stats()
    print("\n系统统计:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
