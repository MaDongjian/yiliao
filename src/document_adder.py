# -*- coding: utf-8 -*-
"""
文档追加模块 - 简化的文档向量化追加接口
"""

import sys
from pathlib import Path
from typing import List, Dict, Union

try:
    from .retriever import DocumentRetriever
except ImportError:
    from retriever import DocumentRetriever


# 全局单例
_adder_instance = None


class DocumentAdder:
    """
    文档追加器 - 一键完成文档向量化并追加到索引

    使用方法:
        adder = DocumentAdder()
        adder.add_file("path/to/document.docx")
    """

    def __init__(self, auto_init: bool = True):
        """
        初始化文档追加器

        Args:
            auto_init: 是否自动初始化
        """
        self._retriever = None

        if auto_init:
            self.initialize()

    def initialize(self):
        """初始化系统"""
        if self._retriever is None:
            self._retriever = DocumentRetriever()
            self._retriever.initialize()

    def add_file(self, file_path: str) -> Dict:
        """
        添加单个文档到向量库

        Args:
            file_path: 文档路径（支持 .pdf, .docx, .pptx）

        Returns:
            {
                'success': bool,        # 是否成功
                'filename': str,        # 文件名
                'doc_id': int,          # 文档ID
                'chunk_count': int,     # 生成的文本块数量
                'message': str          # 提示信息
            }
        """
        # 确保已初始化
        if self._retriever is None:
            self.initialize()

        file_path = Path(file_path)

        if not file_path.exists():
            return {
                'success': False,
                'filename': file_path.name,
                'message': f'文件不存在: {file_path}'
            }

        # 检查文件格式
        ext = file_path.suffix.lower()
        supported_formats = ['.pdf', '.docx', '.pptx']
        if ext not in supported_formats:
            return {
                'success': False,
                'filename': file_path.name,
                'message': f'不支持的文件格式: {ext}。支持的格式: {supported_formats}'
            }

        try:
            # 添加文档
            result = self._retriever.add_document(str(file_path))

            return {
                'success': True,
                'filename': result['filename'],
                'doc_id': result['doc_id'],
                'chunk_count': result['chunk_count'],
                'message': f'成功添加文档: {result["filename"]}，生成 {result["chunk_count"]} 个文本块'
            }

        except Exception as e:
            return {
                'success': False,
                'filename': file_path.name,
                'message': f'添加失败: {str(e)}'
            }

    def add_files(self, file_paths: List[str]) -> List[Dict]:
        """
        批量添加文档

        Args:
            file_paths: 文档路径列表

        Returns:
            每个文件的添加结果列表
        """
        results = []

        for file_path in file_paths:
            result = self.add_file(file_path)
            results.append(result)

        return results

    def add_directory(self, directory: str, extensions: List[str] = None) -> Dict:
        """
        添加整个目录的文档

        Args:
            directory: 目录路径
            extensions: 文件扩展名列表（默认: ['.pdf', '.docx', '.pptx']）

        Returns:
            {
                'success': bool,
                'total': int,           # 总数
                'success_count': int,   # 成功数
                'failed_count': int,    # 失败数
                'results': List[Dict]   # 详细结果
            }
        """
        # 确保已初始化
        if self._retriever is None:
            self.initialize()

        if extensions is None:
            extensions = ['.pdf', '.docx', '.pptx']

        try:
            result = self._retriever.add_directory(str(directory), extensions)

            return {
                'success': True,
                'total': result['total'],
                'success_count': result['success'],
                'failed_count': result['failed'],
                'results': result['files'],
                'message': f'处理完成: {result["success"]}/{result["total"]} 成功'
            }

        except Exception as e:
            return {
                'success': False,
                'total': 0,
                'success_count': 0,
                'failed_count': 0,
                'results': [],
                'message': f'添加目录失败: {str(e)}'
            }

    def get_stats(self) -> Dict:
        """获取当前统计信息"""
        if self._retriever is None:
            self.initialize()
        return self._retriever.get_stats()


# 便捷函数
def add_document(file_path: str) -> Dict:
    """
    添加单个文档 - 便捷函数

    Args:
        file_path: 文档路径

    Returns:
        添加结果

    示例:
        >>> from document_adder import add_document
        >>> result = add_document("path/to/document.pdf")
        >>> if result['success']:
        ...     print(f"成功! 文档ID: {result['doc_id']}")
    """
    global _adder_instance
    if _adder_instance is None:
        _adder_instance = DocumentAdder()

    return _adder_instance.add_file(file_path)


def add_documents(file_paths: List[str]) -> List[Dict]:
    """
    批量添加文档 - 便捷函数

    Args:
        file_paths: 文档路径列表

    Returns:
        添加结果列表
    """
    global _adder_instance
    if _adder_instance is None:
        _adder_instance = DocumentAdder()

    return _adder_instance.add_files(file_paths)


if __name__ == "__main__":
    # 测试代码
    print("=" * 70)
    print("DocumentAdder 测试")
    print("=" * 70)

    adder = DocumentAdder()

    # 测试添加单个文件
    print("\n【测试】添加文件: file_info/情况说明.docx")
    print("-" * 70)

    result = adder.add_file("file_info/情况说明.docx")

    if result['success']:
        print(f"成功!")
        print(f"  文件名: {result['filename']}")
        print(f"  文档ID: {result['doc_id']}")
        print(f"  文本块数: {result['chunk_count']}")
    else:
        print(f"失败: {result['message']}")

    # 显示统计
    print("\n当前统计:")
    stats = adder.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
