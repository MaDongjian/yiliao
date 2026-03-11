# -*- coding: utf-8 -*-
"""
句子级别的精细检索 - 在文档中找到最相关的句子
"""

import re
import sys
from typing import List, Dict, Tuple

# 兼容相对导入和绝对导入
try:
    from .embedder import EmbeddingModel
    from .retriever import DocumentRetriever
except ImportError:
    from embedder import EmbeddingModel
    from retriever import DocumentRetriever


class SentenceFinder:
    """句子级别检索器 - 在文档中找到最相关的句子"""

    def __init__(self, retriever: DocumentRetriever = None):
        """
        初始化句子检索器

        Args:
            retriever: 文档检索器实例
        """
        if retriever is None:
            retriever = DocumentRetriever()
            retriever.initialize()

        self.retriever = retriever
        self.embedder = retriever.embedder

    def split_into_sentences(self, text: str) -> List[str]:
        """
        将文本分割成句子

        Args:
            text: 输入文本

        Returns:
            句子列表
        """
        # 中文句子分隔符
        chinese_delimiters = ['。', '！', '？', '；', '\n']
        # 英文句子分隔符
        english_delimiters = ['. ', '! ', '? ', '; ']

        sentences = []
        current = ""

        for char in text:
            current += char
            if char in chinese_delimiters:
                sentence = current.strip()
                if sentence:
                    sentences.append(sentence)
                current = ""
            elif char in ['.', '!', '?', ';']:
                # 检查后面是否是空格
                pass

        # 处理最后一段
        if current.strip():
            sentences.append(current.strip())

        # 过滤过短的句子
        sentences = [s for s in sentences if len(s) > 5]

        return sentences

    def find_most_relevant_sentences(
        self,
        query: str,
        filepath: str = None,
        top_k: int = 5,
        min_sentence_length: int = 10,
        min_score: float = 0.3
    ) -> List[Dict]:
        """
        在文档中找到最相关的句子

        Args:
            query: 查询文本
            filepath: 指定文件路径（可选，不指定则搜索所有文档）
            top_k: 返回前K个最相关的句子
            min_sentence_length: 最小句子长度
            min_score: 最小相似度阈值（默认0.3，低于此值的结果将被过滤）

        Returns:
            相关句子列表，每个包含:
            {
                'sentence': str,        # 句子文本
                'score': float,         # 相似度分数
                'filename': str,        # 文件名
                'filepath': str,        # 文件路径
                'context': str          # 上下文（前后文）
            }
        """
        # 首先进行文档级搜索（设置相似度阈值）
        if filepath:
            # 搜索特定文件
            search_results = self.retriever.search(query, top_k=50, min_score=min_score)
            search_results = [r for r in search_results if r['filepath'] == filepath]
        else:
            # 搜索所有文件
            search_results = self.retriever.search(query, top_k=50, min_score=min_score)

        if not search_results:
            return []

        # 收集所有句子
        all_sentences = []

        for result in search_results:
            text = result['text']
            filename = result['filename']
            filepath = result['filepath']

            # 分割成句子
            sentences = self.split_into_sentences(text)

            for sentence in sentences:
                if len(sentence) < min_sentence_length:
                    continue

                all_sentences.append({
                    'sentence': sentence,
                    'filename': filename,
                    'filepath': filepath,
                    'context': text
                })

        if not all_sentences:
            return []

        # 计算所有句子的向量
        sentence_texts = [s['sentence'] for s in all_sentences]
        query_embedding = self.embedder.encode(query)
        sentence_embeddings = self.embedder.encode(sentence_texts, show_progress=False)

        # 计算相似度
        import numpy as np
        similarities = np.dot(sentence_embeddings, query_embedding)

        # 添加分数
        for i, sentence_dict in enumerate(all_sentences):
            sentence_dict['score'] = float(similarities[i])

        # 按相似度排序并过滤低于阈值的结果
        all_sentences.sort(key=lambda x: x['score'], reverse=True)
        # 只保留相似度高于阈值的结果
        filtered_sentences = [s for s in all_sentences if s['score'] >= min_score]

        # 返回top_k（最多）
        return filtered_sentences[:top_k]

    def find_by_keyword(
        self,
        keyword: str,
        filepath: str = None,
        top_k: int = 10,
        window_size: int = 50
    ) -> List[Dict]:
        """
        通过关键词查找句子（精确匹配）

        Args:
            keyword: 关键词
            filepath: 指定文件路径（可选）
            top_k: 返回前K个结果
            window_size: 上下文窗口大小

        Returns:
            匹配的句子列表
        """
        # 使用关键词搜索
        search_results = self.retriever.search(
            keyword,
            top_k=50,
            method='keyword'
        )

        if filepath:
            search_results = [r for r in search_results if r['filepath'] == filepath]

        all_sentences = []

        for result in search_results:
            text = result['text']

            # 找到关键词所在的位置
            keyword_lower = keyword.lower()
            text_lower = text.lower()

            start_idx = 0
            while True:
                pos = text_lower.find(keyword_lower, start_idx)
                if pos == -1:
                    break

                # 提取上下文
                context_start = max(0, pos - window_size)
                context_end = min(len(text), pos + len(keyword) + window_size)

                # 提取句子（前后句号之间的内容）
                sentence_start = text.rfind('。', 0, pos) + 1
                if sentence_start == 0:
                    sentence_start = text.rfind('\n', 0, pos) + 1
                if sentence_start == 0:
                    sentence_start = max(0, pos - 50)

                sentence_end = text.find('。', pos)
                if sentence_end == -1:
                    sentence_end = text.find('\n', pos)
                if sentence_end == -1:
                    sentence_end = min(len(text), pos + 100)

                sentence = text[sentence_start:sentence_end].strip()

                if len(sentence) > 5:
                    all_sentences.append({
                        'sentence': sentence,
                        'score': 1.0,  # 关键词匹配给满分
                        'filename': result['filename'],
                        'filepath': result['filepath'],
                        'context': text[context_start:context_end]
                    })

                start_idx = pos + 1

                if len(all_sentences) >= top_k:
                    break

            if len(all_sentences) >= top_k:
                break

        return all_sentences[:top_k]

    def find_in_document(
        self,
        query: str,
        filepath: str,
        top_k: int = 5,
        method: str = 'semantic'
    ) -> List[Dict]:
        """
        在指定文档中查找最相关的句子

        Args:
            query: 查询文本
            filepath: 文件路径
            top_k: 返回前K个结果
            method: 搜索方法 ('semantic' 或 'keyword')

        Returns:
            相关句子列表
        """
        if method == 'keyword':
            return self.find_by_keyword(query, filepath, top_k=top_k)
        else:
            return self.find_most_relevant_sentences(query, filepath, top_k=top_k)


def format_sentence_results(results: List[Dict]) -> str:
    """格式化句子检索结果"""
    if not results:
        return "未找到相关句子"

    output = []
    output.append(f"找到 {len(results)} 个相关句子:\n")

    for i, result in enumerate(results, 1):
        output.append(f"[{i}] {result['filename']}")
        output.append(f"    相似度: {result['score']:.4f}")
        output.append(f"    句子: {result['sentence']}")
        if result.get('context') and len(result['context']) > len(result['sentence']):
            output.append(f"    上下文: ...{result['context'][:100]}...")
        output.append("")

    return "\n".join(output)


if __name__ == "__main__":
    # 测试代码
    finder = SentenceFinder()

    # 测试1: 语义搜索找句子
    print("=" * 70)
    print("测试1: 语义搜索找句子 - '感染管理'")
    print("=" * 70)
    results = finder.find_most_relevant_sentences("感染管理", top_k=3)
    print(format_sentence_results(results))

    # 测试2: 关键词搜索找句子
    print("\n" + "=" * 70)
    print("测试2: 关键词搜索找句子 - '培训'")
    print("=" * 70)
    results = finder.find_by_keyword("培训", top_k=3)
    print(format_sentence_results(results))
