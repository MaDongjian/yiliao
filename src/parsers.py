"""
文档解析器 - 支持 Word、PDF、PPT 格式
支持图片 OCR 文字识别（使用 PaddleOCR）
"""

import os
import io
import base64
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import traceback


class DocumentParser:
    """统一的文档解析接口"""

    def __init__(self, enable_ocr: bool = True, ocr_lang: str = 'ch'):
        """
        初始化文档解析器

        Args:
            enable_ocr: 是否启用图片 OCR 识别（默认 True）
            ocr_lang: OCR 语言，'ch'中文，'en'英文（默认 'ch'）
        """
        self.supported_formats = {
            '.docx': self._parse_word,
            '.pdf': self._parse_pdf,
            '.pptx': self._parse_ppt,
        }
        self.enable_ocr = enable_ocr
        self.ocr_lang = ocr_lang
        self._ocr_engine = None  # 延迟初始化

    def _get_ocr_engine(self):
        """获取 OCR 引擎（延迟初始化）"""
        if not self.enable_ocr:
            return None

        if self._ocr_engine is None:
            try:
                from paddleocr import PaddleOCR
                # 初始化 PaddleOCR
                self._ocr_engine = PaddleOCR(
                    use_angle_cls=True,
                    lang=self.ocr_lang,
                    use_gpu=False,  # 默认使用 CPU，可改为 True
                    show_log=False  # 关闭日志输出
                )
            except ImportError:
                print("警告: 未安装 PaddleOCR，图片文字识别功能不可用")
                print("      安装命令: pip install paddleocr paddlepaddle")
                self._ocr_engine = None
            except Exception as e:
                print(f"警告: PaddleOCR 初始化失败: {e}")
                self._ocr_engine = None

        return self._ocr_engine

    def _ocr_image(self, image_data: bytes, image_name: str = "image") -> str:
        """
        对图片进行 OCR 文字识别

        Args:
            image_data: 图片二进制数据
            image_name: 图片名称（用于日志）

        Returns:
            识别出的文本，失败返回空字符串
        """
        ocr = self._get_ocr_engine()
        if ocr is None:
            return ""

        try:
            from PIL import Image
            import numpy as np

            # 将图片数据转换为 PIL Image
            image = Image.open(io.BytesIO(image_data))

            # 转换为 numpy 数组（RGB 格式）
            if image.mode != 'RGB':
                image = image.convert('RGB')
            img_array = np.array(image)

            # 使用 PaddleOCR 识别
            results = ocr.ocr(img_array, cls=True)

            # 提取文字
            if results and results[0]:
                texts = []
                for line in results[0]:
                    if line and len(line) >= 2:
                        # line[0] 是坐标，line[1] 是 (文字, 置信度)
                        text = line[1][0] if line[1] else ""
                        if text and text.strip():
                            texts.append(text.strip())

                if texts:
                    ocr_text = "\n".join(texts)
                    return f"[图片文字 {image_name}]\n{ocr_text}\n"

        except Exception as e:
            print(f"图片 {image_name} OCR 识别失败: {e}")

        return ""

    def parse(self, file_path: str) -> Dict:
        """
        解析文档

        Args:
            file_path: 文档路径

        Returns:
            {
                'text': str,           # 提取的完整文本
                'pages': List[str],    # 按页/段落分割的文本
                'metadata': Dict       # 文档元数据
            }
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = file_path.suffix.lower()

        if ext not in self.supported_formats:
            raise ValueError(f"不支持的文件格式: {ext}。支持的格式: {list(self.supported_formats.keys())}")

        parser_func = self.supported_formats[ext]

        try:
            result = parser_func(str(file_path))
            result['metadata'] = {
                'filename': file_path.name,
                'filepath': str(file_path),
                'format': ext,
                'size': file_path.stat().st_size
            }
            return result
        except Exception as e:
            raise Exception(f"解析文件 {file_path.name} 失败: {str(e)}\n{traceback.format_exc()}")

    def _parse_word(self, file_path: str) -> Dict:
        """解析 Word (.docx) 文件 - 提取段落、表格、页眉、页脚、文本框、图片等"""
        try:
            from docx import Document
            from docx.oxml import parse_xml
        except ImportError:
            raise ImportError("请安装 python-docx: pip install python-docx")

        doc = Document(file_path)
        all_text_parts = []

        # 1. 提取页眉
        for section in doc.sections:
            if section.header:
                header = section.header
                for para in header.paragraphs:
                    text = para.text.strip()
                    if text:
                        all_text_parts.append(f"[页眉] {text}")

        # 2. 提取主段落文本
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                all_text_parts.append(text)

        # 3. 提取表格文本
        for table_idx, table in enumerate(doc.tables):
            for row_idx, row in enumerate(table.rows):
                row_text = " | ".join([cell.text.strip() for cell in row.cells])
                if row_text.strip():
                    all_text_parts.append(row_text)

        # 4. 提取页脚
        for section in doc.sections:
            if section.footer:
                footer = section.footer
                for para in footer.paragraphs:
                    text = para.text.strip()
                    if text:
                        all_text_parts.append(f"[页脚] {text}")

        # 5. 提取文本框（从 XML 中）
        try:
            # 获取文档的 XML
            for para in doc.paragraphs:
                # 检查段落中是否包含文本框（通过检查 XML）
                if para._element.xml:
                    # 查找所有文本框（w:txbxContent 或 w:textbox）
                    xml = para._element.xml
                    if 'txbxContent' in xml or 'textbox' in xml:
                        # 尝试提取文本框内容
                        for child in para._element.iter():
                            if hasattr(child, 'text') and child.text and child.text.strip():
                                text = child.text.strip()
                                if text and text not in all_text_parts:
                                    all_text_parts.append(f"[文本框] {text}")
        except Exception as e:
            # 文本框提取失败不影响主流程
            pass

        # 6. 提取嵌入对象（如 Excel 表格等）
        try:
            from docx.oxml.ns import qn
            for para in doc.paragraphs:
                # 检查嵌入对象
                for obj in para._element.iter():
                    if obj.tag.endswith('}object'):  # 嵌入对象
                        # 尝试提取对象的数据
                        for child in obj.iter():
                            if hasattr(child, 'text') and child.text and child.text.strip():
                                text = child.text.strip()
                                if text and text not in all_text_parts:
                                    all_text_parts.append(f"[嵌入对象] {text}")
        except Exception as e:
            pass

        # ===== 7. 提取并识别图片中的文字 =====
        if self.enable_ocr:
            try:
                # 从文档中提取图片
                # Word 文档的图片存储在 document.xml.rels 中
                image_count = 0
                for rel in doc.part.rels.values():
                    if "image" in rel.target_ref:
                        try:
                            image_data = rel.target_part.blob
                            image_count += 1

                            # 使用 OCR 识别图片中的文字
                            ocr_text = self._ocr_image(
                                image_data,
                                image_name=f"word_img{image_count}"
                            )

                            if ocr_text:
                                all_text_parts.append(ocr_text)

                        except Exception as e:
                            # 单个图片识别失败不影响其他图片
                            continue
            except Exception as e:
                # 图片提取失败不影响文本解析
                pass

        # 合并所有文本
        full_text = "\n".join(all_text_parts)

        # 按段落分页（用于后续处理）
        paragraphs = [t for t in all_text_parts if not t.startswith('[页眉]') and not t.startswith('[页脚]')]
        pages = paragraphs if paragraphs else [full_text]

        return {
            'text': full_text,
            'pages': pages,
            'type': 'word'
        }

    def _parse_pdf(self, file_path: str) -> Dict:
        """解析 PDF 文件 - 支持中文，多方法回退"""
        # 方法1: 尝试使用 PyMuPDF (fitz) - 对中文支持最好
        try:
            import fitz  # PyMuPDF
            return self._parse_pdf_with_pymupdf(file_path)
        except ImportError:
            print("警告: 未安装 PyMuPDF，尝试使用 pdfplumber (pip install pymupdf 以获得更好的中文支持)")
        except Exception as e:
            print(f"PyMuPDF 解析失败: {e}，尝试使用 pdfplumber...")

        # 方法2: 回退到 pdfplumber，使用 layout 参数
        try:
            import pdfplumber
            return self._parse_pdf_with_pdfplumber(file_path)
        except ImportError:
            raise ImportError("请安装 PDF 解析库: pip install pymupdf pdfplumber")
        except Exception as e:
            # 方法3: 最后尝试使用 PyPDF2
            try:
                import PyPDF2
                return self._parse_pdf_with_pypdf2(file_path)
            except ImportError:
                raise ImportError("请安装 PDF 解析库: pip install pymupdf 或 pip install pdfplumber")
            except Exception as e2:
                raise Exception(f"所有 PDF 解析方法都失败了: pdfplumber({e}), PyPDF2({e2})")

    def _parse_pdf_with_pymupdf(self, file_path: str) -> Dict:
        """使用 PyMuPDF 解析 PDF - 对中文支持最好，支持图片 OCR"""
        import fitz  # PyMuPDF

        doc = fitz.open(file_path)
        pages_text = []
        full_text_parts = []

        for page_num, page in enumerate(doc):
            page_text = ""

            # 方法1: 使用 text 模式提取（最可靠）
            text = page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES)
            if text and text.strip():
                text = text.strip()
                # 检测是否是乱码
                if self._is_valid_chinese_text(text):
                    page_text = text
                else:
                    # 方法2: 尝试 blocks 模式并按位置排序
                    blocks = page.get_text("blocks")
                    if blocks:
                        # 按 (y0, x0) 排序，确保阅读顺序正确（从上到下，从左到右）
                        sorted_blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
                        blocks_text = "\n".join([b[4] for b in sorted_blocks if b[4] and b[4].strip()])
                        if blocks_text:
                            page_text = blocks_text.strip()

                    # 方法3: 如果 blocks 也没用，尝试 dict 模式
                    if not page_text:
                        try:
                            text_dict = page.get_text("dict")
                            if text_dict and "blocks" in text_dict:
                                dict_text_parts = []
                                for block in text_dict["blocks"]:
                                    if "lines" in block:
                                        for line in block["lines"]:
                                            if "spans" in line:
                                                for span in line["spans"]:
                                                    if "text" in span and span["text"].strip():
                                                        dict_text_parts.append(span["text"])
                                if dict_text_parts:
                                    page_text = "\n".join(dict_text_parts)
                        except:
                            pass

            # 方法4: 最后尝试原始模式（不带 flags）
            if not page_text:
                raw_text = page.get_text("text")
                if raw_text and raw_text.strip():
                    page_text = raw_text.strip()

            # ===== 提取并识别图片中的文字 =====
            if self.enable_ocr:
                try:
                    # 获取页面中的图片列表
                    image_list = page.get_images()
                    if image_list:
                        for img_index, img in enumerate(image_list):
                            try:
                                # 获取图片的 xref
                                xref = img[0]

                                # 提取图片
                                base_image = doc.extract_image(xref)
                                if base_image and "image" in base_image:
                                    image_bytes = base_image["image"]
                                    image_ext = base_image.get("ext", "png")

                                    # 使用 OCR 识别图片中的文字
                                    ocr_text = self._ocr_image(
                                        image_bytes,
                                        image_name=f"page{page_num + 1}_img{img_index + 1}"
                                    )

                                    if ocr_text:
                                        page_text += f"\n{ocr_text}"

                            except Exception as e:
                                # 单个图片识别失败不影响其他图片
                                continue
                except Exception as e:
                    # 图片提取失败不影响文本解析
                    pass

            if page_text:
                pages_text.append(page_text)
                full_text_parts.append(page_text)

        doc.close()

        full_text = "\n\n".join(full_text_parts)

        return {
            'text': full_text,
            'pages': pages_text,
            'type': 'pdf'
        }

    def _parse_pdf_with_pdfplumber(self, file_path: str) -> Dict:
        """使用 pdfplumber 解析 PDF"""
        import pdfplumber

        pages_text = []
        full_text_parts = []

        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                # 使用 layout 模式提取，对表格和复杂布局更好
                text = page.extract_text(layout=True)
                if not text or not text.strip():
                    # 回退到普通模式
                    text = page.extract_text()

                if text:
                    text = text.strip()
                    if self._is_valid_chinese_text(text):
                        pages_text.append(text)
                        full_text_parts.append(text)

        full_text = "\n\n".join(full_text_parts)

        return {
            'text': full_text,
            'pages': pages_text,
            'type': 'pdf'
        }

    def _parse_pdf_with_pypdf2(self, file_path: str) -> Dict:
        """使用 PyPDF2 解析 PDF - 最后的回退选项"""
        import PyPDF2

        pages_text = []
        full_text_parts = []

        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text = text.strip()
                    pages_text.append(text)
                    full_text_parts.append(text)

        full_text = "\n\n".join(full_text_parts)

        return {
            'text': full_text,
            'pages': pages_text,
            'type': 'pdf'
        }

    def _is_valid_chinese_text(self, text: str) -> bool:
        """检测文本是否包含有效的中文内容（非乱码）"""
        if not text or len(text) < 10:
            return True  # 太短的文本跳过检测

        # 检查是否包含中文字符
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        total_chars = len(text)

        # 如果中文字符占比超过 5%，认为是有效中文文本
        if chinese_chars / total_chars > 0.05:
            return True

        # 检查是否是乱码（大量重复的特殊字符）
        # 例如: "ÿÿþýüûÿþþýüýý"
        special_chars = set('ÿþüûýþ')
        special_count = sum(1 for c in text if c in special_chars)
        if special_count > total_chars * 0.3:
            return False  # 可能是乱码

        # 检查是否有基本的有效字符（字母、数字、中文、标点）
        valid_chars = sum(1 for c in text
                          if c.isalnum() or
                          '\u4e00' <= c <= '\u9fff' or
                          c in '，。！？、：；""''（）【】《》')
        if valid_chars < total_chars * 0.3:
            return False

        return True

    def _parse_ppt(self, file_path: str) -> Dict:
        """解析 PowerPoint (.pptx) 文件 - 提取文本框、表格、SmartArt、图表、备注、图片等"""
        try:
            from pptx import Presentation
        except ImportError:
            raise ImportError("请安装 python-pptx: pip install python-pptx")

        prs = Presentation(file_path)
        slides_text = []
        full_text_parts = []

        # 用于存储 PPT 中的图片（延迟处理）
        ppt_images = []

        for slide_idx, slide in enumerate(prs.slides):
            slide_content = []

            # 提取所有形状中的文本
            for shape in slide.shapes:
                # 1. 普通文本框和占位符
                if hasattr(shape, "text") and shape.text.strip():
                    slide_content.append(shape.text.strip())

                # 2. 表格
                if shape.has_table:
                    table = shape.table
                    for row in table.rows:
                        row_text = " | ".join([cell.text.strip() for cell in row.cells])
                        if row_text.strip():
                            slide_content.append(row_text)

                # 3. 组（Group）中的形状
                if shape.shape_type == 6:  # Group shape
                    try:
                        for sub_shape in shape.shapes:
                            if hasattr(sub_shape, "text") and sub_shape.text.strip():
                                slide_content.append(sub_shape.text.strip())
                    except:
                        pass

                # 4. SmartArt（作为图表处理）
                if shape.shape_type == 18:  # SmartArt
                    try:
                        # 尝试提取 SmartArt 中的文本
                        for node in shape.shapes:
                            if hasattr(node, "text") and node.text.strip():
                                slide_content.append(f"[SmartArt] {node.text.strip()}")
                    except:
                        # 如果无法访问子形状，尝试 XML 提取
                        try:
                            xml = shape.element.xml
                            # 查找所有文本节点
                            import re
                            texts = re.findall(r'<a:t>([^<]+)</a:t>', xml)
                            for text in texts:
                                if text.strip():
                                    slide_content.append(f"[SmartArt] {text.strip()}")
                        except:
                            pass

                # 5. 图表（Chart）
                if shape.has_chart:
                    try:
                        chart = shape.chart
                        # 提取图表标题
                        if chart.chart_title.has_text_frame:
                            title = chart.chart_title.text_frame.text.strip()
                            if title:
                                slide_content.append(f"[图表标题] {title}")

                        # 提取系列数据
                        for series in chart.series:
                            series_name = series.name if series.name else "系列"
                            slide_content.append(f"[图表系列] {series_name}")

                        # 提取类别（X轴标签）
                        if chart.category_collection:
                            categories = [cat.value if cat.value else "" for cat in chart.category_collection]
                            if categories:
                                slide_content.append(f"[图表类别] {', '.join(categories)}")
                    except Exception as e:
                        pass

                # 6. OLE 对象（嵌入对象）
                if shape.shape_type == 7:  # OLE Object
                    try:
                        # 尝试从 XML 中提取文本
                        xml = shape.element.xml
                        import re
                        texts = re.findall(r'<a:t>([^<]+)</a:t>', xml)
                        for text in texts:
                            if text.strip():
                                slide_content.append(f"[嵌入对象] {text.strip()}")
                    except:
                        pass

                # ===== 7. 提取图片（延迟 OCR 处理） =====
                if self.enable_ocr and shape.shape_type == 13:  # Picture
                    try:
                        # 获取图片数据
                        if hasattr(shape, 'image'):
                            image = shape.image
                            image_data = image.blob
                            image_filename = image.filename

                            # 存储图片信息，稍后统一处理
                            ppt_images.append({
                                'data': image_data,
                                'name': f"slide{slide_idx + 1}_{image_filename}",
                                'slide_idx': slide_idx
                            })
                    except Exception as e:
                        # 单个图片提取失败不影响其他图片
                        continue

            # 8. 提取备注
            try:
                if hasattr(slide, "notes_slide") and slide.notes_slide:
                    notes_text = slide.notes_slide.notes_text_frame.text.strip()
                    if notes_text:
                        slide_content.append(f"[备注] {notes_text}")
            except:
                pass

            # 合并幻灯片内容
            slide_text = "\n".join(slide_content)
            if slide_text:
                slides_text.append(slide_text)
                full_text_parts.append(slide_text)

        # ===== 处理所有图片的 OCR =====
        if self.enable_ocr and ppt_images:
            # 按幻灯片索引分组图片
            images_by_slide = {}
            for img in ppt_images:
                slide_idx = img['slide_idx']
                if slide_idx not in images_by_slide:
                    images_by_slide[slide_idx] = []
                images_by_slide[slide_idx].append(img)

            # 为每个幻灯片添加 OCR 文字
            for slide_idx, images in images_by_slide.items():
                ocr_texts = []
                for img in images:
                    ocr_text = self._ocr_image(img['data'], img['name'])
                    if ocr_text:
                        ocr_texts.append(ocr_text)

                # 将 OCR 文字添加到对应幻灯片的文本中
                if ocr_texts and slide_idx < len(full_text_parts):
                    full_text_parts[slide_idx] += "\n" + "\n".join(ocr_texts)

        full_text = "\n\n---\n\n".join(full_text_parts)

        return {
            'text': full_text,
            'pages': slides_text,
            'type': 'ppt'
        }

    def batch_parse(self, file_paths: List[str]) -> List[Dict]:
        """批量解析文档"""
        results = []
        for file_path in file_paths:
            try:
                result = self.parse(file_path)
                results.append(result)
            except Exception as e:
                print(f"警告: 解析文件 {file_path} 失败: {e}")
                continue
        return results


def scan_documents(directory: str, extensions: Optional[List[str]] = None) -> List[str]:
    """
    扫描目录中的所有支持格式的文档

    Args:
        directory: 目录路径
        extensions: 文件扩展名列表，如 ['.pdf', '.docx']

    Returns:
        文件路径列表
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")

    if extensions is None:
        extensions = ['.pdf', '.docx', '.pptx']

    file_paths = []

    for ext in extensions:
        file_paths.extend(directory.rglob(f"*{ext}"))

    return [str(p) for p in file_paths]


if __name__ == "__main__":
    # 测试代码
    parser = DocumentParser()

    # 测试单个文件解析
    # result = parser.parse("test.docx")
    # print(f"提取文本长度: {len(result['text'])}")
    # print(f"分段数量: {len(result['pages'])}")

    # 测试目录扫描
    # files = scan_documents("./documents")
    # print(f"找到 {len(files)} 个文档")
    # for f in files:
    #     print(f"  - {f}")
    pass
