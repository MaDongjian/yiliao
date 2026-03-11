# -*- coding: utf-8 -*-
"""
答案格式化器 - 将RAG结果转换为富文本格式
支持：表格格式、段落格式
"""

from typing import Dict, List
import re


class AnswerFormatter:
    """答案格式化器"""

    def format_table(self, question: str, result: Dict) -> Dict:
        """
        格式化为表格形式

        Args:
            question: 用户问题
            result: RAG查询结果

        Returns:
            {
                'question': str,
                'format_type': 'table',
                'columns': ['标准名称', '标准类型', '标准性质', '提及的原文'],
                'rows': List[Dict]
            }
        """
        sources = result.get('sources', [])

        # 解析来源文档信息，构建表格行
        rows = []
        for i, source in enumerate(sources, 1):
            filename = source.get('filename', '')
            content = source.get('content', '')

            # 从文件名提取标准信息
            standard_info = self._parse_standard_info(filename)

            row = {
                'index': i,
                'name': standard_info.get('name', filename),
                'type': standard_info.get('type', '行业标准'),
                'nature': standard_info.get('nature', '推荐性'),
                'excerpt': content[:200] + '...' if len(content) > 200 else content,
                'similarity': round(source.get('similarity', 0), 2),
                'filename': filename
            }
            rows.append(row)

        return {
            'question': question,
            'format_type': 'table',
            'columns': ['标准名称', '标准类型', '标准性质', '提及的原文'],
            'rows': rows,
            'total': len(rows)
        }

    def format_paragraph(self, question: str, result: Dict) -> Dict:
        """
        格式化为段落形式

        Args:
            question: 用户问题
            result: RAG查询结果

        Returns:
            {
                'question': str,
                'format_type': 'paragraph',
                'sections': {
                    'overview': str,      # 标准概述
                    'key_points': List[str],  # 关键要点
                    'details': List[Dict]     # 详细信息
                }
            }
        """
        answer = result.get('answer', '')
        sources = result.get('sources', [])

        # 构建概述
        overview = self._generate_overview(question, answer, sources)

        # 提取关键要点
        key_points = self._extract_key_points(answer, sources)

        # 构建详细信息
        details = []
        for source in sources:
            standard_info = self._parse_standard_info(source.get('filename', ''))
            detail = {
                'title': standard_info.get('name', source.get('filename', '')),
                'standard_code': standard_info.get('code', ''),
                'type': standard_info.get('type', ''),
                'content': source.get('content', ''),
                'similarity': round(source.get('similarity', 0), 2)
            }
            details.append(detail)

        return {
            'question': question,
            'format_type': 'paragraph',
            'sections': {
                'overview': overview,
                'key_points': key_points,
                'details': details
            }
        }

    def format_html_table(self, question: str, result: Dict) -> str:
        """
        生成HTML表格格式

        Args:
            question: 用户问题
            result: RAG查询结果

        Returns:
            HTML字符串
        """
        table_data = self.format_table(question, result)

        html = f"""
        <div class="qa-table">
            <h3>提及到的标准</h3>
            <table>
                <thead>
                    <tr>
                        <th>标准名称</th>
                        <th>标准类型</th>
                        <th>标准性质</th>
                        <th>提及的原文</th>
                    </tr>
                </thead>
                <tbody>
        """

        for row in table_data['rows']:
            html += f"""
                    <tr>
                        <td>
                            <a href="#" class="standard-link" data-filename="{row['filename']}">
                                {row['name']} <span class="index">①</span>
                            </a>
                        </td>
                        <td>{row['type']}</td>
                        <td>{row['nature']}</td>
                        <td>{row['excerpt']}</td>
                    </tr>
            """

        html += f"""
                </tbody>
            </table>
            <p class="more-link">查看剩下{table_data['total']}条 &gt;&gt;</p>
        </div>
        """

        return html

    def format_html_paragraph(self, question: str, result: Dict) -> str:
        """
        生成HTML段落格式

        Args:
            question: 用户问题
            result: RAG查询结果

        Returns:
            HTML字符串
        """
        paragraph_data = self.format_paragraph(question, result)

        html = f"""
        <div class="qa-paragraph">
            <div class="section overview">
                <h4>标准概述</h4>
                <p>{paragraph_data['sections']['overview']}</p>
            </div>

            <div class="section key-points">
                <h4>关键定义</h4>
                <ul>
        """

        for point in paragraph_data['sections']['key_points']:
            html += f"                    <li>{point}</li>\n"

        html += """
                </ul>
            </div>

            <div class="section details">
                <h4>标准详情</h4>
        """

        for detail in paragraph_data['sections']['details']:
            html += f"""
                <div class="detail-item">
                    <h5>{detail['title']}</h5>
                    <p><strong>标准类型:</strong> {detail['type']}</p>
                    <p><strong>相似度:</strong> {detail['similarity']}</p>
                    <p class="content">{detail['content'][:300]}...</p>
                </div>
            """

        html += """
            </div>
        </div>
        """

        return html

    def _parse_standard_info(self, filename: str) -> Dict:
        """
        从文件名解析标准信息

        Args:
            filename: 文件名

        Returns:
            {
                'name': str,
                'code': str,
                'type': str,
                'nature': str
            }
        """
        info = {
            'name': filename,
            'code': '',
            'type': '行业标准',
            'nature': '推荐性'
        }

        # 匹配标准号模式 (如 GB, WS, DB等)
        patterns = [
            r'(GB\s*[\d.]+-[\d]+)',      # 国标
            r'(WS/T\s*[\d.]+-[\d]+)',    # 卫生标准
            r'(DB\d*/T\s*[\d.]+-[\d]+)', # 地标
            r'(WST\s*[\d.]+-[\d]+)',     # 卫生推荐标准
        ]

        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                info['code'] = match.group(1)
                # 提取标准名称
                name_part = filename.replace(info['code'], '').strip()
                # 移除文件扩展名和括号内容
                name_part = re.sub(r'\.(pdf|docx?|xlsx?)$', '', name_part)
                name_part = re.sub(r'\([^)]*\)', '', name_part).strip()
                info['name'] = name_part if name_part else filename
                break

        # 判断标准类型
        if filename.startswith('GB'):
            info['type'] = '国家标准'
        elif filename.startswith('WS') or filename.startswith('WST'):
            info['type'] = '卫生标准'
        elif filename.startswith('DB'):
            info['type'] = '地方标准'

        # 判断标准性质
        if '/T' in filename or '推荐' in filename:
            info['nature'] = '推荐性'
        elif '强制' in filename:
            info['nature'] = '强制性'
        else:
            info['nature'] = '推荐性'

        return info

    def _generate_overview(self, question: str, answer: str, sources: List[Dict]) -> str:
        """生成概述段落"""
        if not answer:
            return f"针对问题\"{question}\"，在知识库中找到{len(sources)}个相关标准。"

        # 截取答案的前几句作为概述
        overview = answer[:200]
        if len(answer) > 200:
            overview += "..."
        return overview

    def _extract_key_points(self, answer: str, sources: List[Dict]) -> List[str]:
        """从答案中提取关键要点"""
        key_points = []

        # 尝试从答案中提取要点（按句号或换行分割）
        if answer:
            # 按句号、问号、感叹号分割
            sentences = re.split(r'[。！？\n]', answer)
            for sentence in sentences[:5]:  # 最多5个要点
                sentence = sentence.strip()
                if len(sentence) > 10:  # 过滤太短的句子
                    key_points.append(sentence)

        # 如果答案中没有提取到足够要点，从来源中提取
        if len(key_points) < 3:
            for source in sources[:3]:
                filename = source.get('filename', '')
                content = source.get('content', '')
                if content:
                    # 取第一句话
                    first_sentence = content.split('。')[0].strip()
                    if first_sentence:
                        key_points.append(f"{filename}: {first_sentence}")

        return key_points[:6]  # 最多返回6个要点


# 便捷函数
def format_answer(question: str, result: Dict, format_type: str = 'table') -> Dict:
    """
    格式化答案

    Args:
        question: 用户问题
        result: RAG查询结果
        format_type: 'table' 或 'paragraph'

    Returns:
        格式化后的数据
    """
    formatter = AnswerFormatter()

    if format_type == 'table':
        return formatter.format_table(question, result)
    elif format_type == 'paragraph':
        return formatter.format_paragraph(question, result)
    else:
        raise ValueError(f"不支持的格式类型: {format_type}")


def format_answer_html(question: str, result: Dict, format_type: str = 'table') -> str:
    """
    格式化答案为HTML

    Args:
        question: 用户问题
        result: RAG查询结果
        format_type: 'table' 或 'paragraph'

    Returns:
        HTML字符串
    """
    formatter = AnswerFormatter()

    if format_type == 'table':
        return formatter.format_html_table(question, result)
    elif format_type == 'paragraph':
        return formatter.format_html_paragraph(question, result)
    else:
        raise ValueError(f"不支持的格式类型: {format_type}")


# 测试代码
if __name__ == '__main__':
    from qwen_ask import qwen_ask_with_sources

    # 测试问答
    question = "医院医疗废物相关标准"
    result = qwen_ask_with_sources(question)

    formatter = AnswerFormatter()

    print("=" * 70)
    print("表格格式:")
    print("=" * 70)
    table_data = formatter.format_table(question, result)
    print(f"问题: {table_data['question']}")
    print(f"列: {table_data['columns']}")
    print(f"行数: {table_data['total']}")
    for row in table_data['rows'][:3]:
        print(f"  - {row['name']} | {row['type']} | {row['nature']}")

    print("\n" + "=" * 70)
    print("段落格式:")
    print("=" * 70)
    paragraph_data = formatter.format_paragraph(question, result)
    print(f"概述: {paragraph_data['sections']['overview']}")
    print(f"要点数量: {len(paragraph_data['sections']['key_points'])}")
    for i, point in enumerate(paragraph_data['sections']['key_points'][:3], 1):
        print(f"  {i}. {point[:80]}...")
