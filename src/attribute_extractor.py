"""
文档属性提取模块
从文档文本中提取结构化属性信息
"""

import re
import json
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path


# 属性配置表 - 定义要提取的属性及其对应的语义关键词/正则表达式
ATTRIBUTE_CONFIG = {
    "standard_number": {
        "name": "标准号",
        "keywords": ["标准号", "标准编号", "Standard No.", "编号"],
        "patterns": [
            r'(?:标准号|标准编号|编号)\s*[:：]\s*([A-Z/]+[\/]?[A-Z]*\s*\d+[-—–]\d+)',
            r'^([A-Z/]+[\/]?[A-Z]*\s*\d+[-—–]\d+)\s',  # 文档开头的标准号
            r'([A-Z]+[\/]?[A-Z]*\s*\d+[—\-]\d+)',  # 通用标准号格式（支持全角和半角破折号，支持斜杠）
        ]
    },
    "chinese_name": {
        "name": "中文名称",
        "keywords": ["中文名称", "标准名称", "名称", "标题"],
        "patterns": [
            r'(?:中文名称|标准名称|名称|标题)\s*[:：]\s*(.+?)(?:\n|$)',
        ]
    },
    "english_name": {
        "name": "英文名称",
        "keywords": ["英文名称", "English name"],
        "patterns": [
            r'(?:英文名称|English name)\s*[:：]\s*(.+?)(?:\n|$)',
        ]
    },
    "release_date": {
        "name": "发布日期",
        "keywords": ["发布日期", "发布时间", "发布日", "Release date", "Issued on"],
        "patterns": [
            r'(\d{4})\s*[-—–]\s*(\d{1,2})\s*[-—–]\s*(\d{1,2})\s*发布',  # 2025 - 07 - 30 发布
            r'(?:发布日期|发布时间|发布日|Release date|Issued on)\s*[:：]\s*(\d{4}[-—–年]\d{1,2}[-—–月]\d{1,2}[日]?)',
            r'(\d{4})[-—–年](\d{1,2})[-—–月](\d{1,2})日?\s*发布',
        ]
    },
    "implement_date": {
        "name": "实施日期",
        "keywords": ["实施日期", "施行日期", "实施时间", "生效日期", "Implementation date", "Effective date"],
        "patterns": [
            r'(\d{4})\s*[-—–]\s*(\d{1,2})\s*[-—–]\s*(\d{1,2})\s*实施',  # 2026 - 02 - 01 实施
            r'(?:实施日期|施行日期|实施时间|生效日期|Implementation date|Effective date)\s*[:：]\s*(\d{4}[-—–年]\d{1,2}[-—–月]\d{1,2}[日]?)',
            r'(\d{4})[-—–年](\d{1,2})[-—–月](\d{1,2})日?\s*实施',
        ]
    },
    "replace_standard": {
        "name": "替代标准",
        "keywords": ["替代", "代替", "取代", "Replaces", "Supersedes", "代替标准"],
        "patterns": [
            r'(?:替代|代替|取代|Replaces|Supersedes|代替标准)\s*[:：]\s*([A-Z]+\s*\d+[-—–]\d+)',
        ]
    },
    "publishing_unit": {
        "name": "发布单位",
        "keywords": ["发布单位", "发布部门", "主管部门", "发布机构", "Issued by", "Published by"],
        "patterns": [
            r'(.+?)\s+\t*\s+发布',  # xxxx发布（发布在行尾）
            r'(?:发布单位|发布部门|主管部门|发布机构|Issued by|Published by)\s*[:：]\s*(.+?)(?:\n|发布|实施)',
        ]
    },
    "drafting_unit": {
        "name": "起草单位",
        "keywords": ["起草单位", "起草部门", "编制单位", "Drafted by", "Prepared by"],
        "patterns": [
            r'(?:起草单位|起草部门|编制单位|Drafted by|Prepared by)[：:]\s*(.+?)(?:\n|主要起草人|本标准主要)',
            r'本标准起草单位[：:](.+?)(?:\n|本标准主要)',
        ]
    },
    "scope": {
        "name": "适用范围",
        "keywords": ["适用范围", "范围", "适用", "Scope", "Application scope"],
        "patterns": [
            r'(?:适用范围|范围|适用|Scope|Application(?:\s+scope)?)\s*[:：]\s*(.+?)(?:\n\n|术语|定义)',
        ],
        "is_multiline": True  # 可能跨多行
    },
    "keywords": {
        "name": "关键词",
        "keywords": ["关键词", "主题词", "Keywords"],
        "patterns": [
            r'(?:关键词|主题词|Keywords?)\s*[:：]\s*(.+?)(?:\n|$)',
        ]
    }
}


class AttributeExtractor:
    """文档属性提取器"""

    def __init__(self, config: Dict = None):
        """
        初始化属性提取器

        Args:
            config: 属性配置字典，默认使用 ATTRIBUTE_CONFIG
        """
        self.config = config or ATTRIBUTE_CONFIG

    def extract(self, text: str, use_llm: bool = False, llm=None) -> Dict[str, any]:
        """
        从文本中提取所有属性

        Args:
            text: 文档文本
            use_llm: 是否使用LLM进行语义提取（增强提取效果）
            llm: LLM实例（如果use_llm=True）

        Returns:
            属性字典，键为属性ID，值为提取的属性值
        """
        attributes = {}

        # 先尝试使用正则表达式提取
        for attr_id, attr_config in self.config.items():
            value = self._extract_attribute(text, attr_config)
            if value:
                attributes[attr_id] = {
                    "name": attr_config["name"],
                    "value": value
                }

        # 如果启用LLM且配置了LLM，使用LLM补充提取
        if use_llm and llm:
            llm_attributes = self._extract_with_llm(text, llm)
            # 合并LLM提取的结果（LLM结果优先级较低，仅当正则没找到时使用）
            for attr_id, value in llm_attributes.items():
                if attr_id not in attributes and value:
                    attr_config = self.config.get(attr_id, {})
                    attributes[attr_id] = {
                        "name": attr_config.get("name", attr_id),
                        "value": value
                    }

        return attributes

    def _extract_attribute(self, text: str, attr_config: Dict) -> Optional[str]:
        """
        使用正则表达式提取单个属性

        Args:
            text: 文档文本
            attr_config: 属性配置

        Returns:
            提取的属性值，未找到返回None
        """
        patterns = attr_config.get("patterns", [])
        is_multiline = attr_config.get("is_multiline", False)

        for pattern in patterns:
            try:
                if is_multiline:
                    # 多行匹配模式
                    match = re.search(pattern, text, re.MULTILINE | re.DOTALL | re.IGNORECASE)
                else:
                    match = re.search(pattern, text, re.IGNORECASE)

                if match:
                    # 如果有多个分组，尝试合并所有分组
                    if match.groups() and len(match.groups()) > 1:
                        # 合并所有非空分组
                        parts = [g for g in match.groups() if g is not None]
                        # 检查是否是日期格式（纯数字分组）
                        if all(p.strip().strip('—–-').isdigit() for p in parts if p.strip()):
                            # 日期格式：用短横线连接
                            value = '-'.join(p.strip() for p in parts)
                        else:
                            value = ''.join(parts).strip()
                    else:
                        value = match.group(1).strip()
                    # 清理提取的值
                    value = self._clean_value(value)
                    if value:
                        return value
            except re.error:
                continue

        return None

    def _clean_value(self, value: str) -> str:
        """
        清理提取的属性值

        Args:
            value: 原始值

        Returns:
            清理后的值
        """
        # 去除多余空白
        value = re.sub(r'\s+', ' ', value)
        # 去除首尾空格和标点
        value = value.strip()
        value = value.strip('。,.;;，、；：')
        return value

    def _extract_with_llm(self, text: str, llm) -> Dict[str, str]:
        """
        使用LLM进行语义提取

        Args:
            text: 文档文本
            llm: LLM实例

        Returns:
            属性字典
        """
        # 构建提取提示词
        attr_list = "\n".join([
            f"- {config['name']} ({attr_id}): {', '.join(config['keywords'][:3])}"
            for attr_id, config in self.config.items()
        ])

        # 截取文本前3000字符用于提取
        extract_text = text[:3000] if len(text) > 3000 else text

        prompt = f"""请从以下文档文本中提取结构化信息。如果某个属性在文本中找不到，请返回空字符串""。

文档文本：
{extract_text}

需要提取的属性：
{attr_list}

请以JSON格式返回，格式如下：
{{
  "standard_number": "提取的标准号或空字符串",
  "release_date": "提取的发布日期或空字符串",
  ...
}}

只返回JSON，不要其他内容。"""

        try:
            response = llm.generate(prompt)
            # 解析JSON响应
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                return result
        except Exception as e:
            print(f"  LLM提取失败: {e}")

        return {}

    def save_to_json(self, attributes: Dict[str, any], output_path: str, filename: str):
        """
        保存提取的属性到JSON文件

        Args:
            attributes: 属性字典
            output_path: 输出目录
            filename: 文件名（不含扩展名）
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        json_file = output_dir / f"{filename}.json"

        # 添加元数据
        output_data = {
            "filename": filename,
            "extracted_at": datetime.now().isoformat(),
            "attributes": attributes
        }

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        return json_file


def extract_attributes_from_file(
    text: str,
    filename: str,
    output_dir: str = "./attributes",
    use_llm: bool = False,
    llm=None
) -> Dict[str, any]:
    """
    便捷函数：从文件文本中提取属性并保存

    Args:
        text: 文档文本
        filename: 文件名（不含扩展名）
        output_dir: 属性JSON输出目录
        use_llm: 是否使用LLM进行语义提取
        llm: LLM实例

    Returns:
        提取的属性字典
    """
    extractor = AttributeExtractor()
    attributes = extractor.extract(text, use_llm=use_llm, llm=llm)

    # 保存到JSON
    json_file = extractor.save_to_json(attributes, output_dir, filename)

    print(f"  属性已保存: {json_file}")

    return attributes


if __name__ == "__main__":
    # 测试代码
    test_text = """
    WST 368-2025 医院空气净化管理标准

    标准号：WS/T 368-2025
    发布日期：2025年1月15日
    实施日期：2025年7月1日
    代替标准：WS/T 368-2012

    发布单位：国家卫生健康委员会
    起草单位：北京大学人民医院、北京医院

    适用范围：本标准规定了医院空气净化系统的管理要求、技术要求和监测方法。
    本标准适用于各级各类医疗机构。

    关键词：医院、空气净化、管理标准
    """

    extractor = AttributeExtractor()
    attributes = extractor.extract(test_text)

    print("提取的属性：")
    for attr_id, attr_data in attributes.items():
        print(f"  {attr_data['name']}: {attr_data['value']}")
