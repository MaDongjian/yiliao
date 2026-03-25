"""
使用 Qwen2.5 从PDF中提取标准属性
针对医疗标准文档优化的版本
使用 Ollama 本地模型（无需联网）
支持 PDF 扫描件 OCR 识别
"""

import json
import re
import requests
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import io
import numpy as np


# ============ Ollama 配置 ============
OLLAMA_API_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:7b"  # 或 "qwen2.5:7b-instruct" 如果有这个版本


# ============ OCR 引擎 ============
class OCREngine:
    """OCR 引擎类 - 支持 PDF 扫描件文字识别"""

    def __init__(self):
        self.ocr_engine = None

    def get_engine(self):
        """获取 OCR 引擎（延迟初始化）"""
        if self.ocr_engine is None:
            try:
                import os
                # 禁用 oneDNN 以避免兼容性问题
                os.environ['FLAGS_use_mkldnn'] = '0'
                os.environ['OPENBLAS_NUM_THREADS'] = '1'
                os.environ['OMP_NUM_THREADS'] = '1'

                from paddleocr import PaddleOCR
                import warnings
                # 忽略 PaddleOCR 的弃用警告
                warnings.filterwarnings('ignore', category=DeprecationWarning)

                self.ocr_engine = PaddleOCR(
                    use_textline_orientation=True,
                    lang='ch'
                )
            except ImportError:
                print("警告: 未安装 PaddleOCR，无法识别扫描件")
                print("      安装命令: pip install paddleocr paddlepaddle")
                self.ocr_engine = None
            except Exception as e:
                print(f"警告: PaddleOCR 初始化失败: {e}")
                print(f"      提示: 尝试重新安装: pip install --upgrade paddleocr paddlepaddle")
                self.ocr_engine = None

        return self.ocr_engine

    def is_available(self):
        """检查 OCR 是否可用"""
        return self.get_engine() is not None

    def recognize_image(self, image_array):
        """
        识别图片中的文字

        Args:
            image_array: numpy 数组格式的图片

        Returns:
            识别出的文本列表
        """
        engine = self.get_engine()
        if engine is None:
            return []

        try:
            results = engine.ocr(image_array)
            if results and results[0]:
                texts = []
                for line in results[0]:
                    if line and len(line) >= 2:
                        text = line[1][0] if line[1] else ""
                        if text and text.strip():
                            texts.append(text.strip())
                return texts
        except Exception as e:
            print(f"OCR 识别失败: {e}")

        return []

    def recognize_pdf_page(self, page):
        """
        使用 OCR 识别 PDF 页面

        Args:
            page: PyMuPDF 页面对象

        Returns:
            识别出的文本
        """
        # 渲染页面为图片
        mat = fitz.Matrix(2, 2)  # 放大2倍提高识别精度
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")

        # 转换为 PIL Image
        image = Image.open(io.BytesIO(img_data))

        # 转换为 RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # 转换为 numpy 数组
        img_array = np.array(image)

        # OCR 识别
        texts = self.recognize_image(img_array)

        return "\n".join(texts) if texts else ""


# 全局 OCR 实例
_ocr_instance = None


def get_ocr_engine():
    """获取全局 OCR 引擎实例"""
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = OCREngine()
    return _ocr_instance


# ============ 核心 Prompt 设计 ============
EXTRACTION_PROMPT = """你是一个专业的中国国家标准（GB）、卫生标准（WS）、医药标准（YY）等信息提取专家。请从以下文档内容中精确提取指定字段。

【提取字段说明】
1. 标准性质：强制性/推荐性/指导性（通常在标准号后体现：GB为强制性，GB/T为推荐性）
2. 国家标准号：如 GB 15982-2012, WS/T 310-2016, YY 0572-2015 等
3. 标准状态：现行/废止/即将实施（根据当前日期判断，或文档中明确说明）
4. 被代替国标号：如 "代替 GB 15982-1995"
5. 主管部门：如 "中华人民共和国国家卫生健康委员会"、"国家药品监督管理局"
6. 中文标准名称：完整的中文名称，如 "医院消毒卫生标准"
7. 英文标准名称：完整的英文名称，如 "Hygienic standard for disinfection in hospitals"
8. 发布日期：YYYY-MM-DD 格式，从发布/实施信息中提取
9. 实施日期：YYYY-MM-DD 格式
10. 国际标准分类号：ICS 号，如 "11.080"
11. 中国标准分类号：如 "C59"
12. 第一起草单位：排在第一位的起草单位
13. 其他起草单位：其他参与起草的单位，多个用分号；分隔
14. 作者：主要起草人姓名，多个用分号；分隔
15. 采标号：采用的国际标准号，如 "ISO 11137:1995"
16. 采用国际标准类别：ISO/IEC/ASTM等
17. 采用程度：等同采用(IDT)/修改采用(MOD)/非等效采用(NEQ)

【重要规则】
- 日期必须转换为 YYYY-MM-DD 格式，如 "2012-06-29"
- 如果字段信息不存在，使用 null（不要使用"未找到"或空字符串）
- 第一起草单位通常在"前言"或"归口单位"中列出
- 发布和实施日期通常在封面底部，格式如"2012-06-29发布"
- 标准性质判断：GB=强制性、GB/T=推荐性、GB/Z=指导性；WS同理
- ICS号通常在封面左上角，格式如"ICS 11.080"
- 中国标准分类号在ICS下方，如"C59"

【文档内容】
{pdf_text}

【输出格式】
请严格按照以下JSON格式输出，不要添加任何其他文字说明：
```json
{{
  "标准性质": null,
  "国家标准号": null,
  "标准状态": null,
  "被代替国标号": null,
  "主管部门": null,
  "中文标准名称": null,
  "英文标准名称": null,
  "发布日期": null,
  "实施日期": null,
  "国际标准分类号": null,
  "中国标准分类号": null,
  "第一起草单位": null,
  "其他起草单位": null,
  "作者": null,
  "采标号": null,
  "采用国际标准类别": null,
  "采用程度": null
}}
```"""


# ============ Ollama 调用函数 ============
def call_ollama(prompt: str, model: str = None) -> str:
    """
    调用 Ollama API 生成文本

    Args:
        prompt: 用户提示词
        model: 模型名称，默认使用 OLLAMA_MODEL

    Returns:
        模型生成的文本响应
    """
    if model is None:
        model = OLLAMA_MODEL

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,  # 降低温度以获得更确定的结果
            "top_p": 0.95,
            "num_predict": 2048  # 最大生成长度
        }
    }

    try:
        response = requests.post(
            OLLAMA_API_URL,
            json=payload,
            timeout=120  # 2分钟超时
        )
        response.raise_for_status()

        result = response.json()
        return result.get("message", {}).get("content", "")

    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"无法连接到 Ollama 服务 ({OLLAMA_API_URL})。\n"
            f"请确保 Ollama 已安装并运行：\n"
            f"1. 安装 Ollama: https://ollama.ai\n"
            f"2. 运行服务: ollama serve\n"
            f"3. 拉取模型: ollama pull qwen2.5:7b"
        )
    except requests.exceptions.Timeout:
        raise TimeoutError(f"Ollama API 请求超时（超过120秒）")
    except Exception as e:
        raise RuntimeError(f"Ollama API 调用失败: {e}")


# ============ 提取函数 ============
def extract_text_from_pdf(pdf_path, max_pages=None, enable_ocr=True, ocr_threshold=100):
    """
    从 PDF 提取文本，支持扫描件 OCR 识别

    Args:
        pdf_path: PDF 文件路径
        max_pages: 读取的最大页数，None表示读取全部页面
        enable_ocr: 是否启用 OCR（默认 True）
        ocr_threshold: 文本字符数低于此值时启用 OCR（默认 100）

    Returns:
        提取的文本内容
    """
    doc = fitz.open(pdf_path)
    text_content = []
    ocr_engine = get_ocr_engine() if enable_ocr else None
    total_pages = len(doc)

    for i, page in enumerate(doc):
        if max_pages is not None and i >= max_pages:
            break

        # 1. 先尝试直接提取文本
        text = page.get_text()
        text = text.strip()

        # 2. 判断是否需要 OCR
        if len(text) < ocr_threshold and ocr_engine and ocr_engine.is_available():
            print(f"   🔍 页面 {i+1}/{total_pages} 文本较少({len(text)}字符)，使用 OCR 识别...")
            ocr_text = ocr_engine.recognize_pdf_page(page)
            if ocr_text:
                text = ocr_text
                print(f"   ✅ OCR 识别成功，提取 {len(text)} 字符")
            else:
                print(f"   ⚠️  OCR 识别失败，使用原始文本")
        elif text:
            # 文本足够，直接使用
            if max_pages is None or total_pages <= 10:
                # 只在处理少量页面时显示进度
                print(f"   📄 页面 {i+1}/{total_pages} 提取 {len(text)} 字符")

        if text:
            text_content.append(f"===== 第 {i+1} 页 =====\n{text}")

    doc.close()
    return "\n\n".join(text_content)


def parse_date(date_str):
    """解析各种日期格式为 YYYY-MM-DD"""
    if not date_str or date_str == "null":
        return None

    # 匹配各种中文日期格式
    patterns = [
        r'(\d{4})[年\-](\d{1,2})[月\-](\d{1,2})',
        r'(\d{4})\.(\d{1,2})\.(\d{1,2})',
        r'(\d{4})-(\d{1,2})-(\d{1,2})',
    ]

    for pattern in patterns:
        match = re.search(pattern, date_str)
        if match:
            year, month, day = match.groups()
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"

    return None


def clean_json_response(response):
    """清理模型输出的JSON字符串"""
    # 移除可能的markdown代码块标记
    response = response.strip()
    response = re.sub(r'^```json\s*', '', response)
    response = re.sub(r'^```\s*', '', response)
    response = re.sub(r'\s*```$', '', response)

    # 尝试提取JSON部分
    start = response.find('{')
    end = response.rfind('}') + 1

    if start >= 0 and end > start:
        return response[start:end]

    return response


def extract_standard_attributes(pdf_path, max_pages=5, model=None, enable_ocr=True):
    """
    使用 Ollama Qwen 模型提取标准属性

    Args:
        pdf_path: PDF 文件路径
        max_pages: 读取的最大页数
        model: Ollama 模型名称（默认使用 OLLAMA_MODEL）
        enable_ocr: 是否启用 OCR 识别扫描件（默认 True）

    Returns:
        提取的属性字典，失败返回 None
    """
    print(f"📄 正在处理: {Path(pdf_path).name}")

    # 1. 提取PDF文本
    pdf_text = extract_text_from_pdf(pdf_path, max_pages, enable_ocr=enable_ocr)

    if not pdf_text.strip():
        print(f"   ⚠️  无法提取文本内容")
        return None

    # 2. 构建prompt
    prompt = EXTRACTION_PROMPT.format(pdf_text=pdf_text[:8000])  # 限制长度

    # 3. 调用 Ollama 模型
    try:
        response = call_ollama(prompt, model)
    except Exception as e:
        print(f"   ❌ 模型调用失败: {e}")
        return None

    # 4. 解析JSON
    try:
        cleaned_response = clean_json_response(response)
        result = json.loads(cleaned_response)

        # 5. 后处理日期格式
        for key in ['发布日期', '实施日期']:
            if result.get(key) and result[key] != 'null':
                result[key] = parse_date(str(result[key]))

        print(f"   ✅ 提取成功: {result.get('国家标准号', 'N/A')}")
        return result

    except json.JSONDecodeError as e:
        print(f"   ❌ JSON解析失败: {e}")
        print(f"   原始响应: {response[:200]}")
        return None
    except Exception as e:
        print(f"   ❌ 提取失败: {e}")
        return None


# ============ 批量处理 ============
def batch_extract_standard_attributes(
    pdf_dir,
    output_file="提取结果_ollama.xlsx",
    model=None,
    enable_ocr=True,
    save_txt=True,
    txt_output_dir=None
):
    """
    批量提取PDF标准属性并生成文本文件

    Args:
        pdf_dir: PDF文件目录
        output_file: Excel输出文件路径
        model: Ollama模型名称
        enable_ocr: 是否启用OCR识别
        save_txt: 是否保存每个PDF的txt文本文件（默认True）
        txt_output_dir: txt文本文件输出目录，默认与Excel文件同目录
    """

    from openpyxl import Workbook
    import openpyxl

    # 获取所有PDF文件
    pdf_dir = Path(pdf_dir)
    pdf_files = list(pdf_dir.glob("*.pdf"))

    print(f"📁 找到 {len(pdf_files)} 个PDF文件\n")

    # 创建Excel工作簿
    wb = Workbook()
    ws = wb.active
    ws.title = "标准属性提取"

    # 写入表头
    headers = [
        "文件名", "标准性质", "国家标准号", "标准状态", "被代替国标号",
        "主管部门", "中文标准名称", "英文标准名称", "发布日期", "实施日期",
        "国际标准分类号", "中国标准分类号", "第一起草单位", "其他起草单位",
        "作者", "采标号", "采用国际标准类别", "采用程度"
    ]
    ws.append(headers)

    # 设置txt输出目录
    if save_txt:
        if txt_output_dir is None:
            txt_output_dir = Path(output_file).parent / "txt文本"
        else:
            txt_output_dir = Path(txt_output_dir)
        txt_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📝 文本文件将保存到: {txt_output_dir}\n")

    # 批量处理
    results = []
    txt_saved_count = 0

    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] {pdf_file.name}")

        # 提取完整PDF文本内容（用于保存txt，包含所有页面）
        print(f"   📖 提取完整文本内容...")
        pdf_text_full = extract_text_from_pdf(pdf_file, max_pages=None, enable_ocr=enable_ocr)
        total_chars = len(pdf_text_full)
        print(f"   📊 总计提取 {total_chars} 字符")

        # 提取标准属性（只取前5页即可）
        result = extract_standard_attributes(pdf_file, max_pages=5, model=model, enable_ocr=enable_ocr)

        if result:
            result["文件名"] = pdf_file.name
            results.append(result)

            # 写入Excel
            row = [result.get(h, "") for h in headers]
            ws.append(row)

        # 保存txt文本文件（即使属性提取失败也保存txt）
        if save_txt and pdf_text_full.strip():
            txt_filename = pdf_file.stem + ".txt"
            txt_path = txt_output_dir / txt_filename
            try:
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(pdf_text_full)
                print(f"   💾 TXT已保存: {txt_filename} ({total_chars} 字符)")
                txt_saved_count += 1
            except Exception as e:
                print(f"   ⚠️  TXT保存失败: {e}")

        print()  # 换行

    # 保存Excel
    wb.save(output_file)
    print(f"\n✅ 完成！结果已保存到: {output_file}")
    if save_txt:
        print(f"📝 共保存 {txt_saved_count} 个文本文件到: {txt_output_dir}")

    return results


# ============ 测试单文件 ============
def test_single_file(pdf_path, model=None, enable_ocr=True):
    """测试单个文件提取"""

    # 提取
    result = extract_standard_attributes(pdf_path, model=model, enable_ocr=enable_ocr)

    # 打印结果
    if result:
        print("\n" + "="*50)
        print("📋 提取结果:")
        print("="*50)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # 单文件测试
        test_single_file(sys.argv[1])
    else:
        # 批量处理
        batch_extract_standard_attributes(
            pdf_dir="E:/项目/标准",
            # pdf_dir="E:/answerInfo/yiliaozsk1/file_info/test/标准",
            output_file="E:/answerInfo/yiliaozsk1/file_info/test/标准/提取结果_ollama.xlsx"
        )
