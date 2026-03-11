"""
单文件向量化工具
传入PDF/Word/PPT文件路径，自动追加到向量库并生成TXT文本
支持的格式: .pdf, .docx, .pptx

功能：
1. 解析文档并保存为TXT
2. 提取文档属性
3. 生成文本概要（千问）
4. 向量化并存储到FAISS
5. 保存元数据到数据库
"""

import os
import sys
from pathlib import Path
import json

# 设置离线模式
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.parsers import DocumentParser
from src.chunker import TextChunker
from src.embedder import EmbeddingModel
from src.attribute_extractor import AttributeExtractor, extract_attributes_from_file
from src.text_summarizer import generate_summary
import sqlalchemy as sa


def add_single_file(
    file_path: str,
    output_dir: str = "./data",
    model_path: str = "E:/answerInfo/yiliaozsk1/models/paraphrase-multilingual-MiniLM-L12-v2",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    save_txt: bool = True,
    txt_dir: str = "./txt",
    extract_attributes: bool = True,
    generate_summary_flag: bool = True,
    save_to_db: bool = True
):
    """
    添加单个文件到向量库（支持 PDF/Word/PPT）

    Args:
        file_path: 文件路径（支持 .pdf, .docx, .pptx）
        output_dir: 向量库输出目录（默认./data）
        model_path: 向量化模型路径
        chunk_size: 分块大小（默认512）
        chunk_overlap: 分块重叠大小（默认50）
        save_txt: 是否保存解析后的文本为txt文件（默认True）
        txt_dir: txt文件保存目录（默认./txt）
        extract_attributes: 是否提取文档属性（默认True）
        generate_summary_flag: 是否生成文本概要（默认True）
        save_to_db: 是否保存到数据库（默认True）

    Returns:
        dict: 处理结果统计信息
    """
    file_path = Path(file_path)
    output_dir = Path(output_dir)
    txt_dir_path = Path(txt_dir)

    # 检查文件是否存在
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    # 检查文件格式
    supported_formats = {'.pdf', '.docx', '.pptx'}
    if file_path.suffix.lower() not in supported_formats:
        raise ValueError(f"不支持的文件格式: {file_path.suffix}。支持的格式: {sorted(supported_formats)}")

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_txt:
        txt_dir_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"单文件向量化工具")
    print(f"{'='*70}")
    print(f"文件: {file_path.name}")
    print(f"输出目录: {output_dir}")
    print(f"模型: {Path(model_path).name}")

    # 检查文件是否已索引
    metadata_path = output_dir / "metadata.json"
    existing_metadata = []
    existing_filepaths = set()

    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            existing_metadata = json.load(f)
            existing_filepaths = {
                m.get('filepath', '').replace('\\\\', '\\') for m in existing_metadata
            }

        # 检查是否已存在
        file_rel_path = str(file_path.relative_to(project_root))
        if file_rel_path in existing_filepaths or str(file_path) in existing_filepaths:
            print(f"\n[WARNING] 文件已存在于索引中: {file_path.name}")
            print(f"如需重新索引，请先删除data目录下的faiss.index和metadata.json")
            return {
                'status': 'skipped',
                'reason': 'file_already_indexed',
                'file': str(file_path)
            }

    # 解析文件
    print(f"\n[1/5] 解析文件...")
    parser = DocumentParser()
    try:
        parsed = parser.parse(str(file_path))
        print(f"  解析成功，文本长度: {len(parsed['text'])} 字符")
    except Exception as e:
        print(f"  解析失败: {e}")
        return {
            'status': 'error',
            'reason': 'parse_failed',
            'error': str(e),
            'file': str(file_path)
        }

    # 保存TXT
    if save_txt:
        print(f"\n[2/7] 保存TXT文件...")
        txt_filename = file_path.stem + ".txt"
        txt_filepath = txt_dir_path / txt_filename
        with open(txt_filepath, 'w', encoding='utf-8') as f:
            f.write(f"文件: {file_path.name}\n")
            f.write(f"路径: {file_path}\n")
            f.write(f"{'='*70}\n\n")
            f.write(parsed['text'])
        print(f"  已保存: {txt_filepath}")

    # 生成文本概要
    summary = ""
    if generate_summary_flag:
        print(f"\n[3/6] 生成文本概要（千问）...")
        try:
            summary = generate_summary(parsed['text'], file_path.name)
            print(f"  概要生成成功: {len(summary)} 字符")
            print(f"  概要: {summary[:100]}..." if len(summary) > 100 else f"  概要: {summary}")
        except Exception as e:
            print(f"  概要生成失败: {e}")
            summary = ""

    # 提取属性
    attributes = {}
    if extract_attributes:
        print(f"\n[4/6] 提取文档属性...")
        try:
            extractor = AttributeExtractor()
            attributes = extractor.extract(parsed['text'])
            print(f"  提取到 {len(attributes)} 个属性:")
            for attr_id, attr_data in attributes.items():
                print(f"    - {attr_data['name']}: {attr_data['value'][:50]}..." if len(attr_data['value']) > 50 else f"    - {attr_data['name']}: {attr_data['value']}")
            print(f"  属性信息将保存到数据库")
        except Exception as e:
            print(f"  属性提取失败: {e}")

    # 分块
    print(f"\n[5/6] 文本分块...")
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk(
        text=parsed['text'],
        source_file=parsed['metadata']['filepath'],
        pages=parsed.get('pages')
    )
    print(f"  分块数量: {len(chunks)}")

    # 向量化
    print(f"\n[6/6] 向量化...")
    embedder = EmbeddingModel(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        cache_dir=Path(model_path).parent
    )
    embedder.load()
    embeddings = embedder.encode_chunks(chunks, show_progress=True)
    print(f"  向量维度: {embeddings.shape[1]}")
    print(f"  向量数量: {embeddings.shape[0]}")

    # 处理FAISS索引
    print(f"\n更新向量库...")
    try:
        import faiss
    except ImportError:
        print("  错误: 请安装 faiss-cpu")
        print("    pip install faiss-cpu")
        return {
            'status': 'error',
            'reason': 'faiss_not_installed',
            'file': str(file_path)
        }

    faiss_path = output_dir / "faiss.index"

    # 加载或创建索引
    if faiss_path.exists():
        index = faiss.read_index(str(faiss_path))
        old_count = index.ntotal
        index.add(embeddings.astype('float32'))
        print(f"  追加向量: {old_count} -> {index.ntotal}")
    else:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        print(f"  创建新索引: {index.ntotal} 个向量")

    # 保存索引
    faiss.write_index(index, str(faiss_path))

    # 更新元数据
    if existing_metadata:
        chunks_metadata = existing_metadata
        current_doc_id = max(m.get('doc_id', 0) for m in existing_metadata)
        vector_index = max(m.get('vector_index', -1) for m in existing_metadata) + 1
    else:
        chunks_metadata = []
        current_doc_id = 0
        vector_index = 0

    current_doc_id += 1
    doc_meta = parsed['metadata']

    for chunk in chunks:
        filepath_rel = Path(doc_meta['filepath'])
        try:
            filepath_rel = filepath_rel.relative_to(project_root)
        except ValueError:
            filepath_rel = filepath_rel

        chunks_metadata.append({
            'doc_id': current_doc_id,
            'chunk_id': chunk.chunk_id,
            'text': chunk.text,
            'page_number': chunk.page_number,
            'metadata': chunk.metadata,
            'vector_index': vector_index,
            'filename': doc_meta.get('filename', ''),
            'filepath': str(filepath_rel).replace('\\', '\\\\')
        })
        vector_index += 1

    # 保存元数据
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(chunks_metadata, f, ensure_ascii=False, indent=2)

    # 保存到数据库
    db_record_id = None
    if save_to_db:
        print(f"\n  保存到数据库...")
        try:
            from core.database import db
            import settings
            from flask import Flask
            from model.table.document_attribute import DocumentAttribute

            # 准备属性字典（直接使用中文名称作为键）
            attributes_for_db = {}
            for attr_id, attr_data in attributes.items():
                # 使用中文名称作为键，值作为值
                attributes_for_db[attr_data['name']] = attr_data['value']

            # 创建临时 Flask 应用并保存数据（在应用上下文中）
            temp_app = Flask(__name__)
            temp_app.config.from_object(settings)
            db.init_app(temp_app)

            with temp_app.app_context():
                # 检查是否已存在相同文件的记录
                existing_record = DocumentAttribute.query.filter_by(
                    filename=file_path.name,
                    filepath=str(file_path)
                ).first()

                if existing_record:
                    # 更新已存在的记录
                    existing_record.summary = summary
                    existing_record.attributes = attributes_for_db
                    existing_record.attributes_count = len(attributes)
                    existing_record.text_length = len(parsed['text'])
                    existing_record.chunks_count = len(chunks)
                    existing_record.total_vectors = int(index.ntotal)
                    existing_record.vector_dimension = int(embeddings.shape[1])
                    existing_record.txt_file = str(txt_filepath) if save_txt else None
                    existing_record.status = 'success'
                    existing_record.updated_at = sa.func.now()

                    db.session.commit()
                    db_record_id = existing_record.id
                    print(f"  数据库记录已更新: ID={db_record_id}")
                else:
                    # 创建新记录
                    db_record = DocumentAttribute(
                        filename=file_path.name,
                        filepath=str(file_path),
                        doc_id=current_doc_id,
                        summary=summary,
                        attributes=attributes_for_db,
                        attributes_count=len(attributes),
                        file_type=file_path.suffix.lower().replace('.', ''),
                        file_size=file_path.stat().st_size if file_path.exists() else None,
                        text_length=len(parsed['text']),
                        chunks_count=len(chunks),
                        total_vectors=int(index.ntotal),
                        vector_dimension=int(embeddings.shape[1]),
                        txt_file=str(txt_filepath) if save_txt else None,
                        status='success'
                    )

                    db.session.add(db_record)
                    db.session.commit()
                    db_record_id = db_record.id
                    print(f"  数据库记录已创建: ID={db_record_id}")

        except Exception as e:
            print(f"  数据库保存失败: {e}")
            import traceback
            traceback.print_exc()
            db_record_id = None

    # 汇总
    print(f"\n{'='*70}")
    print(f"处理完成!")
    print(f"{'='*70}")
    print(f"  新增文档: {file_path.name}")
    print(f"  新增文本块: {len(chunks)}")
    print(f"  总向量数量: {index.ntotal}")
    if save_txt:
        print(f"  TXT文件: {txt_filepath}")
    if extract_attributes and attributes:
        print(f"  提取属性: {len(attributes)} 个")
    if summary:
        print(f"  文本概要: {summary[:80]}..." if len(summary) > 80 else f"  文本概要: {summary}")
    if db_record_id:
        print(f"  数据库ID: {db_record_id}")
    print(f"{'='*70}\n")

    # 准备返回的属性字典（直接使用中文名称作为键）
    attributes_for_return = {}
    for attr_id, attr_data in attributes.items():
        attributes_for_return[attr_data['name']] = attr_data['value']

    return {
        'status': 'success',
        'file': str(file_path),
        'chunks_count': len(chunks),
        'total_vectors': int(index.ntotal),
        'txt_file': str(txt_filepath) if save_txt else None,
        'attributes_count': len(attributes),
        'attributes': attributes_for_return,
        'doc_id': current_doc_id,
        'summary': summary,
        'text_length': len(parsed['text']),
        'vector_dimension': int(embeddings.shape[1]),
        'db_record_id': db_record_id
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='文档向量化工具（支持 PDF/Word/PPT）')
    parser.add_argument(
        'file',
        type=str,
        nargs='?',  # 可选参数，用于 --clear 模式
        help='文件路径（支持 .pdf, .docx, .pptx）'
    )
    parser.add_argument(
        '--clear',
        action='store_true',
        help='清空向量库（删除 faiss.index 和 metadata.json）'
    )
    parser.add_argument(
        '--clear-db',
        action='store_true',
        help='清空向量库时同时清空数据库记录'
    )
    parser.add_argument(
        '--batch',
        '-b',
        action='store_true',
        help='批量处理 file_info 目录下的所有文件'
    )
    parser.add_argument(
        '--source-dir',
        '-s',
        type=str,
        default=None,
        help='批量处理时的源文件目录（默认为项目根目录下的 file_info）'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='批量处理时重新索引已存在的文件'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default='./data',
        help='向量库输出目录（默认./data）'
    )
    parser.add_argument(
        '--model',
        '-m',
        type=str,
        default='E:/answerInfo/yiliaozsk1/models/paraphrase-multilingual-MiniLM-L12-v2',
        help='向量化模型路径'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=512,
        help='分块大小（默认512）'
    )
    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=50,
        help='分块重叠大小（默认50）'
    )
    parser.add_argument(
        '--txt-dir',
        '-t',
        type=str,
        default='./txt',
        help='txt文件保存目录（默认./txt）'
    )
    parser.add_argument(
        '--no-txt',
        action='store_true',
        help='不保存TXT文件'
    )
    parser.add_argument(
        '--no-attributes',
        action='store_true',
        help='不提取文档属性'
    )
    parser.add_argument(
        '--no-summary',
        action='store_true',
        help='不生成文本概要'
    )
    parser.add_argument(
        '--no-db',
        action='store_true',
        help='不保存到数据库'
    )
    parser.add_argument(
        '--upload-minio',
        action='store_true',
        help='上传文件到MinIO'
    )
    parser.add_argument(
        '--upload-source-dir',
        type=str,
        default=None,
        help='上传到MinIO的源文件目录（默认为 file_info/test/file_info）'
    )
    parser.add_argument(
        '--upload-date',
        type=str,
        default=None,
        help='上传到MinIO的日期（格式：YYYY-MM-DD，默认为今天）'
    )

    args = parser.parse_args()

    # 处理清空向量库命令
    if args.clear:
        result = clear_vector_index(
            output_dir=args.output,
            clear_db=args.clear_db
        )
        print(f"\nResult: {json.dumps(result, ensure_ascii=False, indent=2)}")
        return

    # 处理MinIO上传命令
    if args.upload_minio:
        result = upload_files_to_minio(
            source_dir=args.upload_source_dir,
            date_str=args.upload_date
        )
        print(f"\nResult: {json.dumps(result, ensure_ascii=False, indent=2)}")
        return

    # 处理批量处理命令
    if args.batch:
        result = batch_add_files(
            source_dir=args.source_dir,
            output_dir=args.output,
            model_path=args.model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            save_txt=not args.no_txt,
            txt_dir=args.txt_dir,
            extract_attributes=not args.no_attributes,
            generate_summary_flag=not args.no_summary,
            save_to_db=not args.no_db,
            skip_existing=not args.force
        )
        print(f"\nResult: {json.dumps(result, ensure_ascii=False, indent=2)}")
        return

    # 如果没有提供文件路径，显示帮助
    if not args.file:
        parser.print_help()
        return

    result = add_single_file(
        file_path=args.file,
        output_dir=args.output,
        model_path=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        save_txt=not args.no_txt,
        txt_dir=args.txt_dir,
        extract_attributes=not args.no_attributes,
        generate_summary_flag=not args.no_summary,
        save_to_db=not args.no_db
    )

    # 打印结果JSON（便于程序调用）
    print(f"\nResult: {json.dumps(result, ensure_ascii=False, indent=2)}")


# 快捷测试函数
def test_add_file(file_path: str):
    """
    快捷函数：添加文件并返回结果

    Args:
        file_path: 文件路径（支持 .pdf, .docx, .pptx）

    Returns:
        dict: 处理结果
    """
    return add_single_file(file_path)


def clear_vector_index(
    output_dir: str = "./data",
    clear_db: bool = False
):
    """
    清空向量库索引和元数据

    Args:
        output_dir: 向量库目录（默认./data）
        clear_db: 是否同时清空数据库中的文档属性记录（默认False）

    Returns:
        dict: 操作结果
    """
    from pathlib import Path
    import json

    output_dir = Path(output_dir)

    print(f"\n{'='*70}")
    print(f"清空向量库")
    print(f"{'='*70}")
    print(f"目录: {output_dir}")

    deleted_files = []
    errors = []

    # 1. 删除 FAISS 索引文件
    faiss_path = output_dir / "faiss.index"
    if faiss_path.exists():
        try:
            faiss_path.unlink()
            deleted_files.append(str(faiss_path))
            print(f"  已删除: {faiss_path.name}")
        except Exception as e:
            errors.append(f"删除 {faiss_path.name} 失败: {e}")
            print(f"  错误: 无法删除 {faiss_path.name}: {e}")
    else:
        print(f"  文件不存在: {faiss_path.name}")

    # 2. 删除元数据文件
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        try:
            metadata_path.unlink()
            deleted_files.append(str(metadata_path))
            print(f"  已删除: {metadata_path.name}")
        except Exception as e:
            errors.append(f"删除 {metadata_path.name} 失败: {e}")
            print(f"  错误: 无法删除 {metadata_path.name}: {e}")
    else:
        print(f"  文件不存在: {metadata_path.name}")

    # 3. 清空数据库记录（可选）
    db_deleted_count = 0
    if clear_db:
        print(f"\n  清空数据库记录...")
        try:
            from core.database import db
            import settings
            from flask import Flask
            from model.table.document_attribute import DocumentAttribute

            # 创建临时 Flask 应用
            temp_app = Flask(__name__)
            temp_app.config.from_object(settings)
            db.init_app(temp_app)

            with temp_app.app_context():
                # 删除所有记录
                count = DocumentAttribute.query.count()
                if count > 0:
                    DocumentAttribute.query.delete()
                    db.session.commit()
                    db_deleted_count = count
                    print(f"  已删除 {count} 条数据库记录")
                else:
                    print(f"  数据库中没有记录")

        except Exception as e:
            errors.append(f"清空数据库失败: {e}")
            print(f"  错误: 清空数据库失败: {e}")

    print(f"\n{'='*70}")
    print(f"清空完成!")
    print(f"{'='*70}")
    print(f"  删除文件: {len(deleted_files)} 个")
    print(f"  删除数据库记录: {db_deleted_count} 条")
    if errors:
        print(f"  错误: {len(errors)} 个")
        for error in errors:
            print(f"    - {error}")
    print(f"{'='*70}\n")

    return {
        'status': 'success' if not errors else 'partial_success',
        'deleted_files': deleted_files,
        'deleted_db_records': db_deleted_count,
        'errors': errors
    }


def batch_add_files(
    source_dir: str = None,
    output_dir: str = "./data",
    model_path: str = "E:/answerInfo/yiliaozsk1/models/paraphrase-multilingual-MiniLM-L12-v2",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    save_txt: bool = True,
    txt_dir: str = "./txt",
    extract_attributes: bool = True,
    generate_summary_flag: bool = True,
    save_to_db: bool = True,
    skip_existing: bool = True
):
    """
    批量处理目录下的所有文件，向量化并保存到数据库

    Args:
        source_dir: 源文件目录（默认为项目根目录下的 file_info）
        output_dir: 向量库输出目录（默认./data）
        model_path: 向量化模型路径
        chunk_size: 分块大小（默认512）
        chunk_overlap: 分块重叠大小（默认50）
        save_txt: 是否保存解析后的文本为txt文件（默认True）
        txt_dir: txt文件保存目录（默认./txt）
        extract_attributes: 是否提取文档属性（默认True）
        generate_summary_flag: 是否生成文本概要（默认True）
        save_to_db: 是否保存到数据库（默认True）
        skip_existing: 是否跳过已索引的文件（默认True）

    Returns:
        dict: 批量处理结果统计
    """
    from pathlib import Path

    # 默认源目录为项目根目录下的 file_info
    if source_dir is None:
        source_dir = project_root / "file_info"
    else:
        source_dir = Path(source_dir)

    source_dir = source_dir.resolve()
    output_dir = Path(output_dir)

    # 支持的文件格式
    supported_formats = {'.pdf', '.docx', '.pptx'}

    # 扫描所有支持的文件
    print(f"\n{'='*70}")
    print(f"批量文档向量化工具")
    print(f"{'='*70}")
    print(f"源目录: {source_dir}")
    print(f"输出目录: {output_dir}")
    print(f"支持的格式: {', '.join(supported_formats)}")

    all_files = []
    for ext in supported_formats:
        files = list(source_dir.rglob(f"*{ext}"))
        all_files.extend(files)

    # 去重并排序
    all_files = sorted(list(set(all_files)))

    if not all_files:
        print(f"\n在 {source_dir} 中没有找到支持的文件")
        return {
            'status': 'no_files',
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'results': []
        }

    print(f"找到 {len(all_files)} 个文件\n")

    # 统计信息
    stats = {
        'total': len(all_files),
        'success': 0,
        'failed': 0,
        'skipped': 0,
        'results': []
    }

    # 读取已索引的文件列表
    indexed_files = set()
    if skip_existing:
        metadata_path = output_dir / "metadata.json"
        if metadata_path.exists():
            import json
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    indexed_files = {
                        m.get('filepath', '').replace('\\\\', '\\') for m in metadata
                    }
            except:
                pass

    # 逐个处理文件
    for i, file_path in enumerate(all_files, 1):
        file_rel_path = str(file_path.relative_to(project_root)) if file_path.is_relative_to(project_root) else str(file_path)

        print(f"\n{'='*70}")
        print(f"[{i}/{len(all_files)}] 处理: {file_path.name}")
        print(f"路径: {file_path}")
        print(f"{'='*70}")

        # 检查是否已索引
        if skip_existing and (file_rel_path in indexed_files or str(file_path) in indexed_files):
            print(f"  文件已索引，跳过")
            stats['skipped'] += 1
            stats['results'].append({
                'file': str(file_path),
                'status': 'skipped',
                'reason': 'already_indexed'
            })
            continue

        # 处理单个文件
        try:
            result = add_single_file(
                file_path=str(file_path),
                output_dir=str(output_dir),
                model_path=model_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                save_txt=save_txt,
                txt_dir=txt_dir,
                extract_attributes=extract_attributes,
                generate_summary_flag=generate_summary_flag,
                save_to_db=save_to_db
            )

            if result['status'] == 'success':
                stats['success'] += 1
                stats['results'].append({
                    'file': str(file_path),
                    'status': 'success',
                    'doc_id': result.get('doc_id'),
                    'chunks_count': result.get('chunks_count'),
                    'db_record_id': result.get('db_record_id')
                })
                print(f"  ✓ 成功: 文档ID={result.get('doc_id')}, 向量数={result.get('total_vectors')}")
            else:
                stats['failed'] += 1
                stats['results'].append({
                    'file': str(file_path),
                    'status': result['status'],
                    'reason': result.get('reason', 'unknown')
                })
                print(f"  ✗ 失败: {result.get('reason', 'unknown')}")

        except Exception as e:
            stats['failed'] += 1
            stats['results'].append({
                'file': str(file_path),
                'status': 'error',
                'error': str(e)
            })
            print(f"  ✗ 异常: {e}")

    # 汇总
    print(f"\n{'='*70}")
    print(f"批量处理完成!")
    print(f"{'='*70}")
    print(f"  总文件数: {stats['total']}")
    print(f"  成功: {stats['success']}")
    print(f"  失败: {stats['failed']}")
    print(f"  跳过: {stats['skipped']}")
    print(f"{'='*70}\n")

    return stats


def query_document_by_filename(
    filename: str = None,
    filepath: str = None,
    doc_id: int = None,
    db_record_id: int = None
):
    """
    根据文件名、文件路径、文档ID或数据库ID查询文档属性信息

    Args:
        filename: 文件名（精确匹配）
        filepath: 文件路径（精确匹配）
        doc_id: 向量库中的文档ID
        db_record_id: 数据库记录ID

    Returns:
        dict: 查询结果，如果找到返回文档信息字典，否则返回 None
              或者返回 {'status': 'error', 'message': '错误信息'}

    示例:
        # 根据文件名查询
        result = query_document_by_filename(filename="标准.pdf")

        # 根据文件路径查询
        result = query_document_by_filename(filepath="E:/documents/标准.pdf")

        # 根据文档ID查询
        result = query_document_by_filename(doc_id=1)

        # 根据数据库ID查询
        result = query_document_by_filename(db_record_id=5)
    """
    try:
        from core.database import db
        import settings
        from flask import Flask
        from model.table.document_attribute import DocumentAttribute

        # 创建临时 Flask 应用
        temp_app = Flask(__name__)
        temp_app.config.from_object(settings)
        db.init_app(temp_app)

        with temp_app.app_context():
            query = DocumentAttribute.query

            # 根据不同的查询条件构建查询
            if db_record_id is not None:
                # 优先使用数据库ID查询
                record = query.filter_by(id=db_record_id).first()
            elif doc_id is not None:
                record = query.filter_by(doc_id=doc_id).first()
            elif filepath is not None:
                record = query.filter_by(filepath=filepath).first()
            elif filename is not None:
                # 文件名查询
                record = query.filter_by(filename=filename).first()
            else:
                return {
                    'status': 'error',
                    'message': '至少提供一个查询参数: filename, filepath, doc_id 或 db_record_id'
                }

            if record:
                return {
                    'status': 'success',
                    'found': True,
                    'data': record.to_dict()
                }
            else:
                return {
                    'status': 'success',
                    'found': False,
                    'message': '未找到匹配的文档记录'
                }

    except Exception as e:
        return {
            'status': 'error',
            'message': f'查询失败: {str(e)}'
        }


def query_all_documents(
    limit: int = None,
    file_type: str = None
):
    """
    查询所有文档属性信息

    Args:
        limit: 限制返回数量（默认返回所有）
        file_type: 过滤文件类型（如 'pdf', 'docx'）

    Returns:
        dict: 查询结果
              {
                  'status': 'success',
                  'count': 10,
                  'data': [...]
              }

    示例:
        # 查询所有文档
        result = query_all_documents()

        # 只查询前10个
        result = query_all_documents(limit=10)

        # 只查询PDF文件
        result = query_all_documents(file_type='pdf')
    """
    try:
        from core.database import db
        import settings
        from flask import Flask
        from model.table.document_attribute import DocumentAttribute

        # 创建临时 Flask 应用
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

            return {
                'status': 'success',
                'count': len(records),
                'data': [record.to_dict() for record in records]
            }

    except Exception as e:
        return {
            'status': 'error',
            'message': f'查询失败: {str(e)}'
        }


def upload_files_to_minio(
    source_dir: str = None,
    date_str: str = None,
    generate_summary_flag: bool = True,
    save_to_db: bool = True,
    force_update: bool = True
):
    """
    解析指定目录下的所有文件，提取概要文本并保存到数据库（不涉及MinIO上传）

    Args:
        source_dir: 源文件目录（默认为 E:/answerInfo/yiliaozsk1/file_info/test/file_info）
        date_str: 日期字符串（仅用于标识，格式：YYYY-MM-DD，默认为今天）
        generate_summary_flag: 是否生成文本概要（默认True）
        save_to_db: 是否保存到数据库（默认True）
        force_update: 是否强制更新已存在的记录（默认True，会替换summary）

    Returns:
        dict: 处理结果
              {
                  'status': 'success',
                  'total_files': 10,
                  'success': 8,
                  'failed': 2,
                  'updated': 5,
                  'created': 3,
                  'results': [...]
              }

    示例:
        # 使用默认目录解析
        result = upload_files_to_minio()

        # 使用指定目录解析
        result = upload_files_to_minio(source_dir='E:/documents')

        # 强制重新提取概要并更新数据库
        result = upload_files_to_minio(force_update=True)
    """
    from datetime import datetime
    import core.database as db_module
    import settings
    from flask import Flask
    from model.table.document_attribute import DocumentAttribute

    # 默认源目录
    if source_dir is None:
        source_dir = Path("E:/answerInfo/yiliaozsk1/file_info/test/file_info")
    else:
        source_dir = Path(source_dir)

    # 默认使用今天的日期（仅用于日志记录）
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    print(f"\n{'='*70}")
    print(f"文档解析提取工具（不涉及MinIO上传）")
    print(f"{'='*70}")
    print(f"源目录: {source_dir}")
    print(f"处理日期: {date_str}")
    print(f"强制更新: {force_update}")
    print(f"{'='*70}\n")

    # 检查源目录是否存在
    if not source_dir.exists():
        return {
            'status': 'error',
            'message': f'源目录不存在: {source_dir}'
        }

    # 获取所有文件
    supported_formats = {'.pdf', '.docx', '.pptx'}
    files = [f for f in source_dir.iterdir() if f.is_file() and f.suffix.lower() in supported_formats]

    if not files:
        return {
            'status': 'success',
            'message': '没有找到需要处理的文件',
            'total_files': 0,
            'success': 0,
            'failed': 0,
            'updated': 0,
            'created': 0,
            'results': []
        }

    print(f"找到 {len(files)} 个文件\n")

    # 处理结果统计
    result = {
        'status': 'success',
        'total_files': len(files),
        'success': 0,
        'failed': 0,
        'updated': 0,
        'created': 0,
        'results': []
    }

    # 创建Flask应用用于数据库操作
    temp_app = Flask(__name__)
    temp_app.config.from_object(settings)
    db_module.db.init_app(temp_app)

    # 处理每个文件
    with temp_app.app_context():
        for file_path in files:
            try:
                print(f"\n{'='*70}")
                print(f"处理文件: {file_path.name}")
                print(f"路径: {file_path}")
                print(f"{'='*70}")

                # 查找数据库中的记录
                existing_record = DocumentAttribute.query.filter_by(
                    filename=file_path.name
                ).first()

                # [1] 解析文件
                print(f"\n[1/2] 解析文件...")
                parser = DocumentParser()
                try:
                    parsed = parser.parse(str(file_path))
                    print(f"  解析成功，文本长度: {len(parsed['text'])} 字符")
                except Exception as e:
                    print(f"  解析失败: {e}")
                    result['failed'] += 1
                    result['results'].append({
                        'file': str(file_path),
                        'status': 'error',
                        'reason': 'parse_failed',
                        'error': str(e)
                    })
                    continue

                # [2] 生成文本概要
                summary = ""

                if generate_summary_flag:
                    print(f"\n[2/2] 生成文本概要（千问）...")
                    try:
                        import time
                        summary_start = time.time()

                        # 极速优化：只使用前1000字符生成概要
                        text_for_summary = parsed['text'][:1000] if len(parsed['text']) > 1000 else parsed['text']
                        summary = generate_summary(text_for_summary, file_path.name)

                        elapsed = time.time() - summary_start
                        print(f"  概要生成成功 ({elapsed:.1f}秒): {len(summary)} 字符")
                        if len(summary) > 60:
                            print(f"  概要: {summary[:60]}...")
                        else:
                            print(f"  概要: {summary}")
                    except Exception as e:
                        print(f"  概要生成失败: {e}")
                        # 快速回退：直接提取前两句
                        import re
                        first_part = parsed['text'][:300].strip()
                        sentences = re.split(r'[。！？]', first_part)
                        if len(sentences) >= 2:
                            summary = sentences[0] + '。' + sentences[1] + '。'
                        else:
                            summary = sentences[0] + '。' if sentences else first_part[:100]
                        summary = f"{file_path.name}：{summary}"
                        print(f"  使用快速回退方案")

                # [3] 保存到数据库
                db_record_id = None
                is_updated = False
                if save_to_db:
                    print(f"\n[3/3] 保存到数据库...")
                    try:
                        if existing_record:
                            # 更新已存在的记录 - 只替换 summary
                            print(f"  找到已存在的记录 (ID={existing_record.id})")
                            if force_update:
                                # 强制更新：只替换 summary
                                existing_record.summary = summary
                                existing_record.text_length = len(parsed['text'])
                                existing_record.file_type = file_path.suffix.lower().replace('.', '')
                                existing_record.file_size = file_path.stat().st_size
                                existing_record.status = 'success'
                                existing_record.error_message = None
                                existing_record.updated_at = datetime.utcnow()

                                db_module.db.session.commit()
                                db_record_id = existing_record.id
                                is_updated = True
                                result['updated'] += 1
                                print(f"  ✓ 数据库记录已更新（summary已替换）: ID={db_record_id}")
                            else:
                                print(f"  跳过更新（force_update=False）")
                                db_record_id = existing_record.id
                        else:
                            # 创建新记录
                            db_record = DocumentAttribute(
                                filename=file_path.name,
                                filepath=str(file_path),
                                summary=summary,
                                file_type=file_path.suffix.lower().replace('.', ''),
                                file_size=file_path.stat().st_size,
                                text_length=len(parsed['text']),
                                status='success'
                            )

                            db_module.db.session.add(db_record)
                            db_module.db.session.commit()
                            db_record_id = db_record.id
                            result['created'] += 1
                            print(f"  ✓ 数据库记录已创建: ID={db_record_id}")

                    except Exception as e:
                        print(f"  数据库保存失败: {e}")
                        import traceback
                        traceback.print_exc()
                        result['failed'] += 1
                        continue

                print(f"\n  ✓ 处理成功")
                result['success'] += 1
                result['results'].append({
                    'file': str(file_path),
                    'status': 'success',
                    'action': 'updated' if is_updated else 'created',
                    'summary': summary,
                    'summary_length': len(summary),
                    'text_length': len(parsed['text']),
                    'db_record_id': db_record_id
                })

            except Exception as e:
                error_msg = f"{file_path.name}: {str(e)}"
                print(f"  ✗ {error_msg}")
                import traceback
                traceback.print_exc()
                result['failed'] += 1
                result['results'].append({
                    'file': str(file_path),
                    'status': 'error',
                    'error': str(e)
                })

    # 打印汇总
    print(f"\n{'='*70}")
    print(f"处理完成!")
    print(f"{'='*70}")
    print(f"  总文件数: {result['total_files']}")
    print(f"  成功: {result['success']}")
    print(f"  失败: {result['failed']}")
    print(f"  更新记录: {result['updated']}")
    print(f"  新增记录: {result['created']}")
    print(f"{'='*70}\n")

    return result


def upload_to_minio_server(
    source_dir: str = None,
    date_str: str = None,
    bucket_name: str = None,
    update_db: bool = True
):
    """
    将指定目录下的所有文件上传到MinIO服务器，并更新数据库中的filepath字段

    Args:
        source_dir: 源文件目录（默认为 E:/answerInfo/yiliaozsk1/file_info/test/file_info）
        date_str: 日期字符串（格式：YYYY-MM-DD，默认为今天，用于构建MinIO路径）
        bucket_name: MinIO存储桶名称（默认使用settings中的配置）
        update_db: 是否更新数据库filepath字段（默认True）

    Returns:
        dict: 上传结果
              {
                  'status': 'success',
                  'total_files': 10,
                  'uploaded': 8,
                  'failed': 2,
                  'updated_db': 8,
                  'minio_paths': {
                      '文件名.pdf': 'medical-doc/2026/03/10/文件名.pdf',
                      ...
                  },
                  'errors': [...]
              }

    MinIO路径格式: medical-doc/YYYY/MM/DD/文件名

    示例:
        # 使用今天的日期上传
        result = upload_to_minio_server()

        # 使用指定日期上传
        result = upload_to_minio_server(date_str='2026-03-10')

        # 指定源目录和日期
        result = upload_to_minio_server(
            source_dir='E:/answerInfo/yiliaozsk1/file_info/test/file_info',
            date_str='2026-03-10'
        )
    """
    from datetime import datetime
    from minio import Minio
    from minio.error import S3Error
    import core.database as db_module
    import settings
    from flask import Flask
    from model.table.document_attribute import DocumentAttribute

    # 默认源目录
    if source_dir is None:
        source_dir = Path("E:/answerInfo/yiliaozsk1/file_info/test/file_info")
    else:
        source_dir = Path(source_dir)

    # 默认使用今天的日期
    if date_str is None:
        date_obj = datetime.now()
    else:
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return {
                'status': 'error',
                'message': f'日期格式错误: {date_str}，正确格式: YYYY-MM-DD'
            }

    # 构建MinIO路径前缀: YYYY/MM/DD
    minio_path_prefix = date_obj.strftime("%Y/%m/%d")

    # MinIO配置
    minio_config = {
        'endpoint': settings.MINIO_ENDPOINT,
        'access_key': settings.MINIO_ACCESS_KEY,
        'secret_key': settings.MINIO_SECRET_KEY,
        'secure': settings.MINIO_SECURE
    }

    if bucket_name is None:
        bucket_name = settings.MINIO_BUCKET_NAME

    print(f"\n{'='*70}")
    print(f"MinIO文件上传工具")
    print(f"{'='*70}")
    print(f"源目录: {source_dir}")
    print(f"目标桶: {bucket_name}")
    print(f"目标路径: {bucket_name}/{minio_path_prefix}/")
    print(f"日期: {date_obj.strftime('%Y-%m-%d')}")
    print(f"更新数据库: {'是' if update_db else '否'}")
    print(f"{'='*70}\n")

    # 检查源目录是否存在
    if not source_dir.exists():
        return {
            'status': 'error',
            'message': f'源目录不存在: {source_dir}'
        }

    # 获取所有支持的文件
    supported_formats = {'.pdf', '.docx', '.pptx', '.jpg', '.jpeg', '.png', '.txt'}
    files = [f for f in source_dir.iterdir() if f.is_file() and f.suffix.lower() in supported_formats]

    if not files:
        return {
            'status': 'success',
            'message': '没有找到需要上传的文件',
            'total_files': 0,
            'uploaded': 0,
            'failed': 0,
            'updated_db': 0,
            'minio_paths': {},
            'errors': []
        }

    print(f"找到 {len(files)} 个文件\n")

    # 初始化MinIO客户端
    try:
        print(f"[1/4] 初始化MinIO客户端...")
        client = Minio(
            minio_config['endpoint'],
            access_key=minio_config['access_key'],
            secret_key=minio_config['secret_key'],
            secure=minio_config['secure']
        )
        print(f"  ✓ MinIO客户端初始化成功")
        print(f"    Endpoint: {minio_config['endpoint']}")
        print(f"    Bucket: {bucket_name}")
    except Exception as e:
        return {
            'status': 'error',
            'message': f'MinIO客户端初始化失败: {str(e)}'
        }

    # 检查/创建存储桶
    try:
        print(f"\n[2/4] 检查存储桶...")
        if not client.bucket_exists(bucket_name):
            print(f"  存储桶不存在，正在创建...")
            client.make_bucket(bucket_name)
            print(f"  ✓ 存储桶创建成功")
        else:
            print(f"  ✓ 存储桶已存在")
    except S3Error as e:
        return {
            'status': 'error',
            'message': f'MinIO存储桶操作失败: {str(e)}'
        }

    # 上传结果
    result = {
        'status': 'success',
        'total_files': len(files),
        'uploaded': 0,
        'failed': 0,
        'updated_db': 0,
        'minio_paths': {},
        'errors': []
    }

    # 上传文件到MinIO
    print(f"\n[3/4] 上传文件到MinIO...")
    for i, file_path in enumerate(files, 1):
        filename = file_path.name
        # MinIO对象路径: YYYY/MM/DD/filename
        object_name = f"{minio_path_prefix}/{filename}"
        # 完整MinIO路径: bucket/YYYY/MM/DD/filename
        minio_full_path = f"{bucket_name}/{object_name}"

        try:
            print(f"\n  [{i}/{len(files)}] 上传: {filename}")
            print(f"    本地路径: {file_path}")
            print(f"    MinIO路径: {minio_full_path}")

            # 上传文件到MinIO
            client.fput_object(
                bucket_name=bucket_name,
                object_name=object_name,
                file_path=str(file_path)
            )
            print(f"    ✓ 上传成功")

            result['uploaded'] += 1
            result['minio_paths'][filename] = minio_full_path

        except S3Error as e:
            error_msg = f"{filename}: MinIO上传失败 - {str(e)}"
            print(f"    ✗ {error_msg}")
            result['failed'] += 1
            result['errors'].append(error_msg)
            continue
        except Exception as e:
            error_msg = f"{filename}: 上传失败 - {str(e)}"
            print(f"    ✗ {error_msg}")
            result['failed'] += 1
            result['errors'].append(error_msg)
            continue

    # 更新数据库中的filepath字段
    if update_db:
        print(f"\n[4/4] 更新数据库filepath字段...")
        try:
            # 创建临时Flask应用
            temp_app = Flask(__name__)
            temp_app.config.from_object(settings)
            db_module.db.init_app(temp_app)

            with temp_app.app_context():
                for filename, minio_path in result['minio_paths'].items():
                    try:
                        # 根据文件名查找记录
                        record = DocumentAttribute.query.filter_by(filename=filename).first()

                        if record:
                            # 更新filepath为MinIO路径
                            old_filepath = record.filepath
                            record.filepath = minio_path
                            db_module.db.session.commit()
                            result['updated_db'] += 1
                            print(f"  ✓ 已更新: {filename}")
                            print(f"    旧路径: {old_filepath}")
                            print(f"    新路径: {minio_path}")
                        else:
                            warning_msg = f"{filename}: 数据库中未找到记录"
                            print(f"  ⚠ {warning_msg}")
                            result['errors'].append(warning_msg)

                    except Exception as e:
                        error_msg = f"{filename}: 数据库更新失败 - {str(e)}"
                        print(f"  ✗ {error_msg}")
                        result['errors'].append(error_msg)

        except Exception as e:
            error_msg = f"数据库连接失败: {str(e)}"
            print(f"  ✗ {error_msg}")
            result['errors'].append(error_msg)

    # 打印汇总
    print(f"\n{'='*70}")
    print(f"上传完成!")
    print(f"{'='*70}")
    print(f"  总文件数: {result['total_files']}")
    print(f"  上传成功: {result['uploaded']}")
    print(f"  上传失败: {result['failed']}")
    print(f"  数据库更新: {result['updated_db']}")
    if result['errors']:
        print(f"  错误数: {len(result['errors'])}")
    print(f"{'='*70}\n")

    return result


if __name__ == "__main__":
    upload_to_minio_server()


