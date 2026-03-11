# -*- coding: utf-8 -*-
"""
文本概要生成模块
使用千问模型生成文档概要
"""

import os
from pathlib import Path
from typing import Optional


# 设置离线模式
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'


class TextSummarizer:
    """
    文本概要生成器
    使用千问模型生成详细文档概要
    """

    def __init__(
        self,
        model_path: str = None,
        quantization: str = "4bit",
        max_summary_length: int = 400
    ):
        """
        初始化概要生成器

        Args:
            model_path: 千问模型路径
            quantization: 量化类型
            max_summary_length: 最大概要长度（字符数，默认400）
        """
        from src.llm_integration import QwenLocalLLM

        if model_path is None:
            # 默认使用项目中的模型路径
            project_root = Path(__file__).resolve().parents[1]
            model_path = project_root / "models" / "Qwen2.5-0.5B-Instruct"

        self.max_summary_length = max_summary_length
        self.llm = QwenLocalLLM(
            model_path=str(model_path),
            quantization=quantization
        )

    def generate_summary(self, text: str, filename: str = "") -> str:
        """
        生成文本概要（快速版本）

        Args:
            text: 文档文本内容
            filename: 文件名（可选，用于上下文）

        Returns:
            概要文本
        """
        import re
        import time

        start_time = time.time()

        # 快速提取关键内容（大幅减少输入）
        text = self._extract_key_content_fast(text, max_length=800)

        # 极简提示词
        if filename:
            prompt = f"文件《{filename}》\n{text}\n\n用一句话概括核心内容："
        else:
            prompt = f"{text}\n\n用一句话概括核心内容："

        try:
            # 快速生成（限制 max_length=128）
            summary = self.llm.generate(prompt, context="", max_length=128)

            elapsed = time.time() - start_time
            if elapsed > 30:
                print(f"  [警告] 概要生成耗时 {elapsed:.1f} 秒")

            # 快速清理
            summary = summary.strip()
            summary = re.sub(r'^[\d\s、\-\.]+', '', summary)

            # 限制长度
            if len(summary) > 200:
                summary = summary[:200].rsplit('，', 1)[0] + '。'

            return summary

        except Exception as e:
            # 快速回退：直接提取
            return self._fast_extract_summary(text, filename)

    def _extract_key_content_fast(self, text: str, max_length: int = 800) -> str:
        """
        快速提取关键内容（只取开头+关键句）

        Args:
            text: 原始文本
            max_length: 最大长度

        Returns:
            提取的内容
        """
        import re

        if len(text) <= max_length:
            return text

        # 只取前500字符（通常包含核心信息）
        result = text[:500]

        # 如果有换行，取第一段
        paragraphs = text.split('\n\n')
        if paragraphs and len(paragraphs[0]) > 50:
            result = paragraphs[0][:max_length]

        return result

    def _fast_extract_summary(self, text: str, filename: str = "") -> str:
        """
        快速提取概要（规则方案，秒级完成）

        Args:
            text: 文本内容
            filename: 文件名

        Returns:
            提取的概要
        """
        import re

        # 取前300字符
        summary = text[:300].strip()

        # 按句子分割，取前两句
        sentences = re.split(r'[。！？]', summary)
        if len(sentences) >= 2:
            summary = sentences[0] + '。' + sentences[1] + '。'
        else:
            summary = sentences[0] + '。' if sentences else summary

        # 移除多余空白
        summary = re.sub(r'\s+', ' ', summary).strip()

        # 添加文件名前缀
        if filename:
            summary = f"{filename}：{summary}"

        return summary

    def _extract_key_content(self, text: str, max_length: int = 2500) -> str:
        """
        智能提取文本关键内容

        优先级：
        1. 标题和开头
        2. 包含关键词的句子（规范、标准、要求、适用等）
        3. 结尾部分

        Args:
            text: 原始文本
            max_length: 最大提取长度

        Returns:
            提取的关键内容
        """
        import re

        if len(text) <= max_length:
            return text

        # 关键词列表
        keywords = [
            '规范', '标准', '要求', '规定', '适用', '范围',
            '管理', '制度', '技术', '操作', '流程', '程序',
            '总则', '原则', '定义', '术语', '目的', '依据'
        ]

        # 按行分割
        lines = text.split('\n')

        # 1. 提取标题行（以#开头或全大写/数字开头）
        title_lines = []
        for line in lines[:50]:  # 只检查前50行
            line = line.strip()
            if line and (line.startswith('#') or
                        re.match(r'^[一二三四五六七八九十]+[、\.]', line) or
                        re.match(r'^\d+[、\.]', line) or
                        re.match(r'^[第][一二三四五六七八九十]+[章节条款]', line)):
                title_lines.append(line)

        # 2. 提取包含关键词的句子
        sentences = re.split(r'[。！？?！\n]', text)
        key_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and len(sentence) < 200:
                for keyword in keywords:
                    if keyword in sentence:
                        key_sentences.append(sentence)
                        break

        # 3. 构建提取内容
        result_parts = []

        # 开头部分（前800字符）
        result_parts.append(text[:800])

        # 添加关键句子（最多10条）
        if key_sentences:
            result_parts.append('\n'.join(key_sentences[:10]))

        # 结尾部分（后500字符）
        result_parts.append(text[-500:])

        result = '\n\n'.join(result_parts)

        # 如果还是太长，截断
        if len(result) > max_length:
            result = result[:max_length]

        return result

    def _fallback_summary(self, text: str, filename: str = "") -> str:
        """
        回退方案：简单提取关键信息

        Args:
            text: 文本内容
            filename: 文件名

        Returns:
            简单概要
        """
        import re

        # 提取第一段作为概要
        first_paragraph = text.split('\n\n')[0][:300]

        # 移除多余空白
        first_paragraph = re.sub(r'\s+', ' ', first_paragraph).strip()

        if filename:
            return f"{filename}\n{first_paragraph}"
        return first_paragraph

    def generate_summary_with_keywords(
        self,
        text: str,
        filename: str = "",
        keywords: list = None
    ) -> dict:
        """
        生成概要并提取关键词

        Args:
            text: 文档文本内容
            filename: 文件名
            keywords: 可选的关键词列表

        Returns:
            {
                'summary': str,
                'keywords': list
            }
        """
        summary = self.generate_summary(text, filename)

        # 如果提供了关键词，直接使用
        if keywords:
            return {
                'summary': summary,
                'keywords': keywords
            }

        # 否则从文本中提取关键词（简单实现）
        # 这里可以添加更复杂的关键词提取逻辑
        return {
            'summary': summary,
            'keywords': []
        }


# 全局单例
_summarizer_instance = None


def get_summarizer(model_path: str = None, quantization: str = "4bit") -> TextSummarizer:
    """
    获取概要生成器实例（单例模式）

    Args:
        model_path: 模型路径
        quantization: 量化类型

    Returns:
        TextSummarizer 实例
    """
    global _summarizer_instance

    if _summarizer_instance is None:
        _summarizer_instance = TextSummarizer(
            model_path=model_path,
            quantization=quantization
        )

    return _summarizer_instance


def generate_summary(text: str, filename: str = "") -> str:
    """
    便捷函数：生成文本概要

    Args:
        text: 文档文本内容
        filename: 文件名

    Returns:
        概要文本

    示例:
        >>> summary = generate_summary("这是一篇关于医疗标准的文档...", "标准.pdf")
        >>> print(summary)
    """
    summarizer = get_summarizer()
    return summarizer.generate_summary(text, filename)


if __name__ == '__main__':
    # 测试代码
    test_text = """
    # 医疗机构消毒技术规范

    ## 第一章 总则

    第一条 为规范医疗机构消毒工作，保障医疗安全，根据《中华人民共和国传染病防治法》
    和相关法律法规，制定本规范。

    第二条 本规范适用于各级各类医疗机构。其他医疗机构参照执行。

    第三条 医疗机构应当建立消毒管理责任制，法定代表人是第一责任人。

    ## 第二章 消毒方法

    第四条 医疗机构应当根据消毒对象的性质，选择合适的消毒方法。
    常用的消毒方法包括：热力消毒、化学消毒、辐射消毒等。

    第五条 热力消毒适用于耐热物品，包括压力蒸汽灭菌、干热灭菌等。

    第六条 化学消毒应根据消毒对象选择合适的消毒剂，严格掌握浓度和作用时间。
    """

    print("=" * 70)
    print("文本概要生成测试")
    print("=" * 70)

    summary = generate_summary(test_text, "消毒技术规范.pdf")
    print(f"\n概要：\n{summary}")
