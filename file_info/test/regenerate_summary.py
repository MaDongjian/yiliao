# -*- coding: utf-8 -*-
"""
重新生成文档概要工具
解决之前概要不精准、截断添加"..."的问题
根据文件名更新数据库中的概要字段
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, List
import re

# 设置离线模式
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.text_summarizer import TextSummarizer


class ImprovedSummarizer:
    """
    改进的概要生成器
    - 不截断文本，智能分段处理
    - 不添加"..."结尾
    - 生成更精准的概要
    """

    def __init__(self, model_path: str = None):
        """
        初始化改进的概要生成器

        Args:
            model_path: 千问模型路径
        """
        if model_path is None:
            model_path = project_root / "models" / "Qwen2.5-0.5B-Instruct"

        # 使用项目中的TextSummarizer，但覆盖max_summary_length
        self.summarizer = TextSummarizer(
            model_path=str(model_path),
            max_summary_length=1000  # 增加限制，但实际不强制截断
        )

    def _split_text_intelligently(self, text: str, max_chunk_size: int = 5000) -> List[str]:
        """
        智能分割文本，保留段落完整性

        Args:
            text: 原始文本
            max_chunk_size: 每块最大字符数

        Returns:
            分割后的文本块列表
        """
        if len(text) <= max_chunk_size:
            return [text]

        chunks = []
        current_chunk = ""
        paragraphs = text.split('\n')

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 如果单个段落就超过限制，需要按句子分割
            if len(para) > max_chunk_size:
                sentences = self._split_long_paragraph(para, max_chunk_size)
                for sent in sentences:
                    if len(current_chunk) + len(sent) > max_chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sent
                    else:
                        current_chunk += "\n" + sent if current_chunk else sent
            else:
                # 正常段落处理
                if len(current_chunk) + len(para) > max_chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para
                else:
                    current_chunk += "\n" + para if current_chunk else para

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _split_long_paragraph(self, paragraph: str, max_size: int) -> List[str]:
        """分割过长的段落"""
        # 按句子分割（中文和英文句号）
        sentences = re.split(r'([。.!！？?])', paragraph)

        result = []
        current = ""

        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else "")
            if len(current) + len(sentence) > max_size:
                if current:
                    result.append(current.strip())
                current = sentence
            else:
                current += sentence

        if current:
            result.append(current.strip())

        return result if result else [paragraph[:max_size]]

    def generate_summary_complete(
        self,
        text: str,
        filename: str = "",
        target_length: int = 400
    ) -> str:
        """
        生成完整、精准的概要（不截断、不加"..."）

        Args:
            text: 文档文本内容
            filename: 文件名
            target_length: 目标概要长度（字符数）

        Returns:
            完整的概要文本（不截断）
        """
        # 智能分割文本
        text_chunks = self._split_text_intelligently(text, max_chunk_size=5000)

        # 取前两块作为主要内容
        main_content = "\n\n".join(text_chunks[:2])

        # 构建更精准的提示词
        prompt = f"""请为以下文档生成精准的概要（约{target_length}字）：

文件名：{filename}

文档内容：
{main_content}

要求：
1. 概要长度约{target_length}字（可适当浮动，但不要低于200字或高于600字）
2. 必须包含完整的信息，不要用"..."结尾
3. 优先包含：文档类型、核心内容、适用范围、关键要点
4. 使用精炼专业的语言
5. 直接输出概要内容，不要有开场白

概要："""

        try:
            # 生成概要
            summary = self.summarizer.llm.generate(prompt, context="")

            # 清理结果
            summary = summary.strip()

            # 移除可能的"..."结尾
            if summary.endswith("..."):
                summary = summary[:-3].strip()

            # 移除可能的开场白
            开场白列表 = [
                "以下是该文档的概要：",
                "概要如下：",
                "该文档的概要是：",
                "文档概要：",
                "概要为："
            ]
            for 开场白 in 开场白列表:
                if summary.startswith(开场白):
                    summary = summary[len(开场白):].strip()

            # 移除可能的结尾说明
            结尾说明列表 = [
                "以上是概要。",
                "以上是文档概要。",
                "（完）",
                "——完——"
            ]
            for 结尾 in 结尾说明列表:
                if summary.endswith(结尾):
                    summary = summary[:-len(结尾)].strip()

            return summary

        except Exception as e:
            return f"概要生成失败: {str(e)}"


# 全局实例
_summarizer_instance = None


def get_improved_summarizer() -> ImprovedSummarizer:
    """获取改进的概要生成器实例"""
    global _summarizer_instance
    if _summarizer_instance is None:
        _summarizer_instance = ImprovedSummarizer()
    return _summarizer_instance


def regenerate_summary_for_file(
    file_path: str,
    update_db: bool = True
) -> Dict:
    """
    为单个文件重新生成概要并更新数据库

    Args:
        file_path: 文件路径
        update_db: 是否更新数据库

    Returns:
        处理结果字典
    """
    from src.parsers import DocumentParser
    from flask import Flask
    import settings
    from core.database import db
    from model.table.document_attribute import DocumentAttribute

    file_path = Path(file_path)

    print(f"\n{'='*70}")
    print(f"重新生成概要: {file_path.name}")
    print(f"{'='*70}")

    # 检查文件是否存在
    if not file_path.exists():
        return {
            'status': 'error',
            'file': str(file_path),
            'reason': 'file_not_found'
        }

    # 解析文件
    print(f"[1/3] 解析文件...")
    parser = DocumentParser()
    try:
        parsed = parser.parse(str(file_path))
        print(f"  解析成功，文本长度: {len(parsed['text'])} 字符")
    except Exception as e:
        print(f"  解析失败: {e}")
        return {
            'status': 'error',
            'file': str(file_path),
            'reason': 'parse_failed',
            'error': str(e)
        }

    # 生成新概要
    print(f"[2/3] 生成新概要...")
    summarizer = get_improved_summarizer()
    new_summary = summarizer.generate_summary_complete(
        text=parsed['text'],
        filename=file_path.name,
        target_length=400
    )
    print(f"  新概要长度: {len(new_summary)} 字符")
    print(f"  新概要: {new_summary[:150]}..." if len(new_summary) > 150 else f"  新概要: {new_summary}")

    # 更新数据库
    if update_db:
        print(f"[3/3] 更新数据库...")
        temp_app = Flask(__name__)
        temp_app.config.from_object(settings)
        db.init_app(temp_app)

        with temp_app.app_context():
            # 根据文件名查找记录
            record = DocumentAttribute.query.filter_by(
                filename=file_path.name
            ).first()

            if record:
                old_summary = record.summary
                record.summary = new_summary
                record.updated_at = db.func.now()
                db.session.commit()

                print(f"  数据库已更新")
                print(f"    记录ID: {record.id}")
                print(f"    旧概要长度: {len(old_summary) if old_summary else 0}")
                print(f"    新概要长度: {len(new_summary)}")

                return {
                    'status': 'success',
                    'file': str(file_path),
                    'db_record_id': record.id,
                    'old_summary_length': len(old_summary) if old_summary else 0,
                    'new_summary_length': len(new_summary),
                    'new_summary': new_summary
                }
            else:
                print(f"  警告: 数据库中未找到文件 '{file_path.name}' 的记录")
                return {
                    'status': 'warning',
                    'file': str(file_path),
                    'reason': 'db_record_not_found',
                    'new_summary': new_summary
                }

    return {
        'status': 'success',
        'file': str(file_path),
        'new_summary': new_summary
    }


def regenerate_summary_for_directory(
    source_dir: str = None,
    update_db: bool = True,
    file_pattern: str = None
) -> Dict:
    """
    批量为目录下的文件重新生成概要

    Args:
        source_dir: 源文件目录（默认 E:/answerInfo/yiliaozsk1/file_info/test/file_info）
        update_db: 是否更新数据库
        file_pattern: 文件名模式过滤（如包含"标准"的文件）

    Returns:
        批量处理结果
    """
    from flask import Flask
    import settings

    # 默认源目录
    if source_dir is None:
        source_dir = Path("E:/answerInfo/yiliaozsk1/file_info/test/file_info")
    else:
        source_dir = Path(source_dir)

    source_dir = source_dir.resolve()

    print(f"\n{'='*70}")
    print(f"批量重新生成概要")
    print(f"{'='*70}")
    print(f"源目录: {source_dir}")
    if file_pattern:
        print(f"文件过滤: 包含 '{file_pattern}'")
    print(f"{'='*70}\n")

    # 检查目录
    if not source_dir.exists():
        return {
            'status': 'error',
            'reason': 'directory_not_found',
            'directory': str(source_dir)
        }

    # 获取支持的文件
    supported_formats = {'.pdf', '.docx', '.pptx', '.doc'}
    all_files = []

    for ext in supported_formats:
        files = list(source_dir.glob(f"*{ext}"))
        all_files.extend(files)

    # 按文件名过滤
    if file_pattern:
        all_files = [f for f in all_files if file_pattern in f.name]

    all_files = sorted(all_files)

    if not all_files:
        print(f"没有找到支持的文件\n")
        return {
            'status': 'no_files',
            'total': 0,
            'success': 0,
            'failed': 0,
            'results': []
        }

    print(f"找到 {len(all_files)} 个文件\n")

    # 统计
    stats = {
        'total': len(all_files),
        'success': 0,
        'failed': 0,
        'warning': 0,
        'results': []
    }

    # 逐个处理
    for i, file_path in enumerate(all_files, 1):
        print(f"\n[{i}/{len(all_files)}] 处理: {file_path.name}")

        result = regenerate_summary_for_file(
            file_path=str(file_path),
            update_db=update_db
        )

        stats['results'].append(result)

        if result['status'] == 'success':
            stats['success'] += 1
        elif result['status'] == 'warning':
            stats['warning'] += 1
        else:
            stats['failed'] += 1

    # 汇总
    print(f"\n{'='*70}")
    print(f"批量处理完成!")
    print(f"{'='*70}")
    print(f"  总文件数: {stats['total']}")
    print(f"  成功: {stats['success']}")
    print(f"  警告: {stats['warning']}")
    print(f"  失败: {stats['failed']}")
    print(f"{'='*70}\n")

    return stats


def regenerate_summary_by_db_filename(
    filename: str,
    update_db: bool = True
) -> Dict:
    """
    根据数据库中的文件名重新生成概要

    Args:
        filename: 文件名（精确匹配）
        update_db: 是否更新数据库

    Returns:
        处理结果
    """
    from flask import Flask
    import settings
    from core.database import db
    from model.table.document_attribute import DocumentAttribute

    print(f"\n{'='*70}")
    print(f"根据数据库文件名重新生成概要")
    print(f"{'='*70}")
    print(f"文件名: {filename}")

    temp_app = Flask(__name__)
    temp_app.config.from_object(settings)
    db.init_app(temp_app)

    with temp_app.app_context():
        # 查找记录
        record = DocumentAttribute.query.filter_by(filename=filename).first()

        if not record:
            print(f"  错误: 数据库中未找到文件 '{filename}'")
            return {
                'status': 'error',
                'reason': 'db_record_not_found',
                'filename': filename
            }

        print(f"  找到记录: ID={record.id}")
        print(f"  文件路径: {record.filepath}")

        # 检查文件是否存在
        file_path = Path(record.filepath)
        if not file_path.exists():
            print(f"  错误: 文件不存在: {file_path}")
            return {
                'status': 'error',
                'reason': 'file_not_found',
                'filepath': str(file_path)
            }

        # 重新生成概要
        result = regenerate_summary_for_file(str(file_path), update_db=update_db)
        result['db_record_id'] = record.id
        return result


def regenerate_summary_from_db(
    limit: int = None,
    file_type: str = None,
    update_db: bool = True
) -> Dict:
    """
    从数据库读取记录并重新生成概要

    Args:
        limit: 处理记录数量限制
        file_type: 文件类型过滤（如 'pdf', 'docx'）
        update_db: 是否更新数据库

    Returns:
        处理结果统计
    """
    from flask import Flask
    import settings
    from core.database import db
    from model.table.document_attribute import DocumentAttribute

    print(f"\n{'='*70}")
    print(f"从数据库重新生成概要")
    print(f"{'='*70}")
    if limit:
        print(f"处理数量限制: {limit}")
    if file_type:
        print(f"文件类型过滤: {file_type}")
    print(f"{'='*70}\n")

    temp_app = Flask(__name__)
    temp_app.config.from_object(settings)
    db.init_app(temp_app)

    with temp_app.app_context():
        query = DocumentAttribute.query

        # 按文件类型过滤
        if file_type:
            query = query.filter_by(file_type=file_type.lower())

        # 限制数量
        if limit:
            query = query.limit(limit)

        records = query.all()

        if not records:
            print(f"没有找到记录\n")
            return {
                'status': 'no_records',
                'total': 0,
                'success': 0,
                'failed': 0,
                'results': []
            }

        print(f"找到 {len(records)} 条记录\n")

        # 统计
        stats = {
            'total': len(records),
            'success': 0,
            'failed': 0,
            'warning': 0,
            'results': []
        }

        # 逐个处理
        for i, record in enumerate(records, 1):
            print(f"\n[{i}/{len(records)}] 处理: {record.filename}")

            # 检查文件是否存在
            file_path = Path(record.filepath)
            if not file_path.exists():
                print(f"  跳过: 文件不存在")
                stats['failed'] += 1
                stats['results'].append({
                    'status': 'skipped',
                    'reason': 'file_not_found',
                    'filename': record.filename,
                    'filepath': str(file_path)
                })
                continue

            # 重新生成概要
            result = regenerate_summary_for_file(
                file_path=str(file_path),
                update_db=update_db
            )

            stats['results'].append(result)

            if result['status'] == 'success':
                stats['success'] += 1
            elif result['status'] == 'warning':
                stats['warning'] += 1
            else:
                stats['failed'] += 1

        # 汇总
        print(f"\n{'='*70}")
        print(f"批量处理完成!")
        print(f"{'='*70}")
        print(f"  总记录数: {stats['total']}")
        print(f"  成功: {stats['success']}")
        print(f"  警告: {stats['warning']}")
        print(f"  失败: {stats['failed']}")
        print(f"{'='*70}\n")

        return stats


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description='重新生成文档概要工具')
    parser.add_argument(
        'file',
        nargs='?',
        help='文件路径（支持 .pdf, .docx, .pptx, .doc）'
    )
    parser.add_argument(
        '--batch',
        '-b',
        action='store_true',
        help='批量处理目录下的所有文件'
    )
    parser.add_argument(
        '--from-db',
        '-d',
        action='store_true',
        help='从数据库读取记录并重新生成概要'
    )
    parser.add_argument(
        '--source-dir',
        '-s',
        type=str,
        default=None,
        help='源文件目录（默认为 E:/answerInfo/yiliaozsk1/file_info/test/file_info）'
    )
    parser.add_argument(
        '--pattern',
        '-p',
        type=str,
        default=None,
        help='文件名过滤模式（只处理包含该字符串的文件）'
    )
    parser.add_argument(
        '--db-filename',
        type=str,
        default=None,
        help='根据数据库中的文件名处理'
    )
    parser.add_argument(
        '--limit',
        '-l',
        type=int,
        default=None,
        help='处理数量限制（配合 --from-db 使用）'
    )
    parser.add_argument(
        '--file-type',
        '-t',
        type=str,
        default=None,
        help='文件类型过滤（如 pdf, docx）'
    )
    parser.add_argument(
        '--no-db',
        action='store_true',
        help='不更新数据库（仅生成概要）'
    )

    args = parser.parse_args()

    # 从数据库处理
    if args.from_db:
        result = regenerate_summary_from_db(
            limit=args.limit,
            file_type=args.file_type,
            update_db=not args.no_db
        )
        print(f"\nResult: {result}")
        return

    # 根据数据库文件名处理
    if args.db_filename:
        result = regenerate_summary_by_db_filename(
            filename=args.db_filename,
            update_db=not args.no_db
        )
        print(f"\nResult: {result}")
        return

    # 批量处理
    if args.batch:
        result = regenerate_summary_for_directory(
            source_dir=args.source_dir,
            update_db=not args.no_db,
            file_pattern=args.pattern
        )
        print(f"\nResult: {result}")
        return

    # 单个文件
    if args.file:
        result = regenerate_summary_for_file(
            file_path=args.file,
            update_db=not args.no_db
        )
        print(f"\nResult: {result}")
        return

    # 没有参数，显示帮助
    parser.print_help()


if __name__ == "__main__":
    main()
