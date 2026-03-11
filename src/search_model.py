# -*- coding: utf-8 -*-
"""
简化的搜索模型 - 一句话完成搜索
"""

import sys
from typing import List, Dict, Literal

# 兼容相对导入和绝对导入
try:
    from .retriever import DocumentRetriever
    from .sentence_finder import SentenceFinder
except ImportError:
    from retriever import DocumentRetriever
    from sentence_finder import SentenceFinder


class SearchModel:
    """
    简化的搜索模型 - 统一的搜索接口

    使用方法:
        model = SearchModel()
        results = model.search("感染管理", method="semantic")
    """

    def __init__(self, auto_init: bool = True):
        """
        初始化搜索模型

        Args:
            auto_init: 是否自动初始化（加载模型和索引）
        """
        self._retriever = None
        self._sentence_finder = None

        if auto_init:
            self.initialize()

    def initialize(self):
        """初始化模型（如果 auto_init=False 时需要手动调用）"""
        if self._retriever is None:
            self._retriever = DocumentRetriever()
            self._retriever.initialize()

        if self._sentence_finder is None:
            self._sentence_finder = SentenceFinder(self._retriever)

    def search(
        self,
        query: str,
        method: Literal["semantic", "keyword", "hybrid"] = "semantic",
        top_k: int = 5,
        level: Literal["chunk", "sentence"] = "chunk"
    ) -> List[Dict]:
        """
        统一的搜索接口

        Args:
            query: 搜索查询词
            method: 搜索方式
                - "semantic": 语义搜索（理解含义，推荐）
                - "keyword": 精准关键词搜索
                - "hybrid": 混合搜索（语义+关键词）
            top_k: 返回前K个结果
            level: 返回级别
                - "chunk": 文本块级别（返回较大段落）
                - "sentence": 句子级别（返回精确句子）

        Returns:
            搜索结果列表，每个结果包含:
            {
                'filename': str,      # 文件名
                'filepath': str,      # 完整文件路径
                'text': str,          # 匹配的文本内容
                'score': float,       # 相似度分数 (0-1)
                'page': int,          # 页码（如果有）
                'method': str         # 使用的搜索方法
            }

        示例:
            >>> model = SearchModel()
            >>> results = model.search("感染管理", method="semantic")
            >>> for r in results:
            ...     print(f"{r['filename']}: {r['text'][:50]}...")
        """
        # 确保已初始化
        if self._retriever is None:
            self.initialize()

        # 根据级别和方法选择搜索方式
        if level == "sentence":
            # 句子级别搜索
            if method == "keyword":
                results = self._sentence_finder.find_by_keyword(query, top_k=top_k)
            else:
                results = self._sentence_finder.find_most_relevant_sentences(query, top_k=top_k)

            # 格式化返回结果
            formatted = []
            for r in results:
                formatted.append({
                    'filename': r['filename'],
                    'filepath': r['filepath'],
                    'text': r['sentence'],
                    'score': r['score'],
                    'page': r.get('page_number'),
                    'method': method
                })
            return formatted

        else:
            # 文本块级别搜索
            if method == "keyword":
                results = self._retriever.search(query, top_k=top_k, method='keyword')
            elif method == "hybrid":
                results = self._retriever.hybrid_search(query, top_k=top_k)
            else:
                results = self._retriever.search(query, top_k=top_k, method='semantic')

            # 格式化返回结果
            formatted = []
            for r in results:
                formatted.append({
                    'filename': r['filename'],
                    'filepath': r['filepath'],
                    'text': r['text'],
                    'score': r['score'],
                    'page': r.get('page_number'),
                    'method': method
                })
            return formatted

    def search_in_file(
        self,
        query: str,
        filepath: str,
        method: Literal["semantic", "keyword"] = "semantic",
        top_k: int = 5
    ) -> List[Dict]:
        """
        在指定文件中搜索

        Args:
            query: 搜索查询词
            filepath: 文件路径
            method: 搜索方式 ("semantic" 或 "keyword")
            top_k: 返回前K个结果

        Returns:
            搜索结果列表
        """
        # 确保已初始化
        if self._sentence_finder is None:
            self.initialize()

        results = self._sentence_finder.find_in_document(
            query,
            filepath,
            top_k=top_k,
            method=method
        )

        # 格式化返回结果
        formatted = []
        for r in results:
            formatted.append({
                'filename': r['filename'],
                'filepath': r['filepath'],
                'text': r['sentence'],
                'score': r['score'],
                'page': r.get('page_number'),
                'method': method
            })
        return formatted

    def get_stats(self) -> Dict:
        """获取系统统计信息"""
        if self._retriever is None:
            self.initialize()
        return self._retriever.get_stats()


# 便捷函数：直接使用，无需手动创建实例
_default_model = None


def search(
    query: str,
    method: Literal["semantic", "keyword", "hybrid"] = "semantic",
    top_k: int = 5,
    level: Literal["chunk", "sentence"] = "sentence"
) -> List[Dict]:
    """
    一行代码完成搜索 - 便捷函数

    Args:
        query: 搜索查询词
        method: 搜索方式 ("semantic", "keyword", "hybrid")
        top_k: 返回前K个结果
        level: 返回级别 ("chunk" 文本块, "sentence" 句子)

    Returns:
        搜索结果列表

    示例:
        >>> from search_model import search
        >>> results = search("感染管理", method="semantic")
        >>> for r in results:
        ...     print(f"{r['filename']}: {r['text'][:50]}")
    """
    global _default_model
    if _default_model is None:
        _default_model = SearchModel()

    return _default_model.search(query, method=method, top_k=top_k, level=level)


if __name__ == "__main__":
    # 测试代码
    print("=" * 70)
    print("SearchModel 测试")
    print("=" * 70)

    model = SearchModel()

    # 测试1: 语义搜索
    print("\n【测试1】语义搜索: 感染管理")
    print("-" * 70)
    results = model.search("感染管理", method="semantic", level="sentence")
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r['filename']}")
        print(f"    相似度: {r['score']:.4f}")
        print(f"    文本: {r['text'][:100]}...")

    # 测试2: 关键词搜索
    print("\n\n【测试2】关键词搜索: 培训")
    print("-" * 70)
    results = model.search("培训", method="keyword", level="sentence")
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r['filename']}")
        print(f"    相似度: {r['score']:.4f}")
        print(f"    文本: {r['text'][:100]}...")

    # 测试3: 便捷函数
    print("\n\n【测试3】便捷函数: 驻场")
    print("-" * 70)
    results = search("驻场", method="keyword", top_k=3)
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r['filename']}")
        print(f"    相似度: {r['score']:.4f}")
        print(f"    文本: {r['text'][:100]}...")
