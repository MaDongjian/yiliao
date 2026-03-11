# -*- coding: utf-8 -*-
"""
文档概要生成模块
使用 LLM 为文档生成概要并保存到文件
"""

from typing import Dict, Optional
from pathlib import Path
import os


class DocumentSummaryGenerator:
    """文档概要生成器"""

    def __init__(
        self,
        model_path: str = None,
        quantization: str = "none",
        summary_output_dir: str = None
    ):
        """
        初始化概要生成器

        Args:
            model_path: 模型路径
            quantization: 量化类型
            summary_output_dir: 概要输出目录（默认：file_info/test/概要）
        """
        self.model_path = model_path or os.path.join(
            os.getcwd(), "models", "Qwen2.5-0.5B-Instruct"
        )
        self.quantization = quantization
        self.llm = None

        # 设置概要输出目录
        if summary_output_dir is None:
            # 默认保存在项目根目录的 file_info/test/概要 下
            self.summary_output_dir = os.path.join(
                os.getcwd(), "file_info", "test", "概要"
            )
        else:
            self.summary_output_dir = summary_output_dir

        # 确保目录存在
        os.makedirs(self.summary_output_dir, exist_ok=True)

    def _load_llm(self):
        """延迟加载 LLM"""
        if self.llm is None:
            from llm_integration import QwenLocalLLM
            self.llm = QwenLocalLLM(
                model_path=self.model_path,
                quantization=self.quantization
            )

    def _get_summary_filename(self, original_filename: str) -> str:
        """
        生成概要文件名

        Args:
            original_filename: 原始文件名（如：document.pdf）

        Returns:
            概要文件名（如：document_概要.txt）
        """
        # 去掉原扩展名
        name = Path(original_filename).stem
        return f"{name}_概要.txt"

    def _save_summary_to_file(self, summary: str, filename: str):
        """
        将概要保存到文件

        Args:
            summary: 概要内容
            filename: 原始文件名
        """
        summary_filename = self._get_summary_filename(filename)
        summary_path = os.path.join(self.summary_output_dir, summary_filename)

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"# 文档概要\n\n")
            f.write(f"原文档：{filename}\n")
            f.write(f"\n{summary}\n")

        print(f"  概要已保存: {summary_path}")
        return summary_path

    def generate_summary(
        self,
        text: str,
        filename: str = "",
        save_to_file: bool = True
    ) -> Dict:
        """
        为文档文本生成概要

        Args:
            text: 文档文本内容
            filename: 文件名（可选）
            save_to_file: 是否保存到文件

        Returns:
            {
                'summary': str,          # 概要内容
                'summary_path': str,     # 概要文件路径（如果保存）
                'filename': str          # 原文件名
            }
        """
        self._load_llm()

        print(f"  正在生成概要...")

        # 截断过长的文本（保留前 3000 字符）
        if len(text) > 3000:
            text = text[:3000] + "\n...(文档已截断)"

        # 构建概要提示词
        if filename:
            prompt = f"""请为以下文档生成一个简明的概要。

文档名称: {filename}

要求：
1. 概要长度控制在 200-300 字
2. 突出文档的主要内容和关键信息
3. 使用简洁、清晰的语言
4. 不要逐字逐句摘录，要提炼核心要点

文档内容：
{text}

概要："""
        else:
            prompt = f"""请为以下文档生成一个简明的概要。

要求：
1. 概要长度控制在 200-300 字
2. 突出文档的主要内容和关键信息
3. 使用简洁、清晰的语言
4. 不要逐字逐句摘录，要提炼核心要点

文档内容：
{text}

概要："""

        try:
            summary = self.llm.generate(prompt, context="", max_length=512)
            summary = summary.strip()

            # 保存到文件
            summary_path = None
            if save_to_file and filename:
                summary_path = self._save_summary_to_file(summary, filename)

            print(f"  概要生成完成！")

            return {
                'summary': summary,
                'summary_path': summary_path,
                'filename': filename
            }
        except Exception as e:
            print(f"  生成概要失败: {e}")
            # 回退：返回文本前200字符
            fallback_summary = text[:200] + "..."
            if save_to_file and filename:
                self._save_summary_to_file(f"[自动回退概要]\n{fallback_summary}", filename)
            return {
                'summary': fallback_summary,
                'summary_path': None,
                'filename': filename
            }

    def generate_summary_for_chunks(
        self,
        chunks: list,
        filename: str = "",
        max_summary_length: int = 500,
        save_to_file: bool = True
    ) -> Dict:
        """
        为多个文本块生成综合概要

        Args:
            chunks: 文本块列表
            filename: 文件名
            max_summary_length: 最大概要长度
            save_to_file: 是否保存到文件

        Returns:
            {
                'summary': str,
                'summary_path': str,
                'filename': str
            }
        """
        self._load_llm()

        # 合并前几个文本块的内容（最多 3000 字符）
        combined_text = ""
        for chunk in chunks[:10]:  # 最多取前 10 个块
            if len(combined_text) + len(chunk.text) > 3000:
                break
            combined_text += chunk.text + "\n\n"

        return self.generate_summary(combined_text, filename, save_to_file)


# 便捷函数
def generate_document_summary(
    text: str,
    filename: str = "",
    model_path: str = None,
    save_to_file: bool = True
) -> Dict:
    """
    生成文档概要 - 便捷函数

    Args:
        text: 文档文本
        filename: 文件名
        model_path: 模型路径
        save_to_file: 是否保存到文件

    Returns:
        概要信息字典
    """
    generator = DocumentSummaryGenerator(model_path=model_path)
    return generator.generate_summary(text, filename, save_to_file)


def generate_summary_for_chunks(
    chunks: list,
    filename: str = "",
    model_path: str = None,
    save_to_file: bool = True
) -> Dict:
    """
    为文本块生成概要 - 便捷函数

    Args:
        chunks: 文本块列表
        filename: 文件名
        model_path: 模型路径
        save_to_file: 是否保存到文件

    Returns:
        概要信息字典
    """
    generator = DocumentSummaryGenerator(model_path=model_path)
    return generator.generate_summary_for_chunks(chunks, filename, save_to_file)


if __name__ == "__main__":
    # 测试代码
    test_text = """
    # 医院感染管理规范

    ## 第一章 总则

    第一条 为加强医院感染管理，有效预防和控制医院感染，提高医疗质量，保证医疗安全，根据《传染病防治法》、《医疗机构管理条例》等法律法规，制定本规范。

    第二条 本规范适用于中华人民共和国境内各级各类医疗机构，包括综合医院、专科医院、康复医院、护理院、诊所、门诊部、卫生院、村卫生室等。

    第三条 医院感染管理是医疗质量管理的重要组成部分，医疗机构应当建立医院感染管理责任制，制定并落实医院感染管理的规章制度和工作规范，严格执行有关技术操作规范和工作标准，有效预防和控制医院感染。

    第四条 卫生行政部门负责对辖区内医疗机构的医院感染管理工作进行监督管理。

    ## 第二章 组织管理

    第五条 医疗机构主要负责人是医院感染管理的第一责任人，应当全面负责本机构的医院感染管理工作。

    第六条 二级以上医院应当设立医院感染管理委员会，其他医疗机构应当设立医院感染管理小组，由医疗机构主要负责人、感染管理部门、医务部门、护理部门、临床科室、消毒供应室、手术室、临床检验部门、药事管理部门、设备管理部门、后勤管理部门及其他相关部门的主要负责人组成。

    第七条 医院感染管理委员会（小组）的主要职责：
    （一）认真贯彻执行医院感染管理相关的法律法规、技术规范和标准；
    （二）制定本医疗机构医院感染管理的规章制度并组织实施；
    （三）对医院感染的监测、报告、控制工作进行监督检查；
    （四）对医院感染暴发事件进行流行病学调查，提出控制措施并组织实施；
    （五）对医务人员进行医院感染管理相关知识的培训和考核；
    （六）对医院感染管理工作进行总结、分析和评估，提出改进措施。

    第八条 医疗机构应当设立医院感染管理部门，配备相应的专职人员，负责本机构的医院感染管理工作。
    """

    generator = DocumentSummaryGenerator()
    result = generator.generate_summary(test_text, "医院感染管理规范.txt")

    print("=" * 70)
    print("文档概要")
    print("=" * 70)
    print(result['summary'])
    print("=" * 70)
    print(f"概要文件: {result['summary_path']}")
