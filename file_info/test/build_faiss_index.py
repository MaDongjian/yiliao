"""
生成FAISS索引和元数据文件
解析当前目录下的PDF文件，进行向量化，生成faiss.index和metadata.json到data文件夹
支持追加模式和文本保存功能
"""

import os
import sys
from pathlib import Path
import json
import numpy as np

# 设置离线模式
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.parsers import DocumentParser, scan_documents
from src.chunker import TextChunker
from src.embedder import EmbeddingModel
from file_info.test.add_single_file import query_document_by_filename


def build_index(
    input_dir: str = ".",
    output_dir: str = "./data",
    model_path: str = "E:/answerInfo/yiliaozsk1/models/paraphrase-multilingual-MiniLM-L12-v2",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    append: bool = False,
    save_txt: bool = False,
    txt_dir: str = "./txt"
):
    """
    构建FAISS索引

    Args:
        input_dir: 输入目录（PDF文件所在目录）
        output_dir: 输出目录（faiss.index和metadata.json保存位置）
        model_path: 向量化模型路径
        chunk_size: 分块大小
        chunk_overlap: 分块重叠大小
        append: 是否追加模式（True=追加到现有索引，False=重建索引）
        save_txt: 是否保存解析后的文本为txt文件
        txt_dir: txt文件保存目录
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # txt目录
    txt_dir_path = Path(txt_dir)
    if save_txt:
        txt_dir_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"FAISS索引生成工具")
    print(f"{'='*70}")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"模型路径: {model_path}")
    print(f"模式: {'追加模式' if append else '重建模式'}")
    if save_txt:
        print(f"TXT保存: {txt_dir_path}")

    # 扫描PDF文件
    print(f"\n[1/7] 扫描PDF文件...")
    pdf_files = scan_documents(str(input_dir), ['.pdf'])

    if not pdf_files:
        print(f"  在 {input_dir} 目录下未找到PDF文件")
        return

    # 追加模式：过滤已索引的文件
    existing_metadata = []
    existing_filepaths = set()

    if append:
        metadata_path = output_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                existing_metadata = json.load(f)
                existing_filepaths = {
                    m.get('filepath', '').replace('\\\\', '\\') for m in existing_metadata
                }
            print(f"  现有索引包含 {len(existing_metadata)} 个chunks")
            print(f"  已索引文件: {len(set(m.get('filename', '') for m in existing_metadata))} 个")

        # 过滤掉已处理的文件
        new_files = [f for f in pdf_files if str(Path(f).relative_to(project_root)) not in existing_filepaths
                     and str(Path(f)) not in existing_filepaths]

        if not new_files:
            print(f"  没有新文件需要处理")
            return

        pdf_files = new_files
        print(f"  过滤后需要处理 {len(pdf_files)} 个新文件")

    print(f"  找到 {len(pdf_files)} 个PDF文件")
    for f in pdf_files:
        print(f"    - {Path(f).name}")

    # 初始化解析器和分块器
    parser = DocumentParser()
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # 解析所有文档
    print(f"\n[2/7] 解析PDF文件...")
    all_documents = []
    for pdf_file in pdf_files:
        try:
            print(f"  解析: {Path(pdf_file).name}")
            parsed = parser.parse(pdf_file)
            all_documents.append(parsed)

            # 保存解析后的文本为txt
            if save_txt:
                txt_filename = Path(pdf_file).stem + ".txt"
                txt_filepath = txt_dir_path / txt_filename
                with open(txt_filepath, 'w', encoding='utf-8') as f:
                    f.write(f"文件: {Path(pdf_file).name}\n")
                    f.write(f"路径: {pdf_file}\n")
                    f.write(f"{'='*70}\n\n")
                    f.write(parsed['text'])
                print(f"    已保存TXT: {txt_filename}")
        except Exception as e:
            print(f"    失败: {e}")
            continue

    print(f"  成功解析 {len(all_documents)} 个文档")

    # 分块
    print(f"\n[3/7] 文本分块...")
    all_chunks = []
    for doc in all_documents:
        chunks = chunker.chunk(
            text=doc['text'],
            source_file=doc['metadata']['filepath'],
            pages=doc.get('pages')
        )
        all_chunks.extend(chunks)

    print(f"  总分块数量: {len(all_chunks)}")

    # 向量化
    print(f"\n[4/7] 加载模型并向量化...")
    embedder = EmbeddingModel(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        cache_dir=Path(model_path).parent
    )
    embedder.load()

    embeddings = embedder.encode_chunks(all_chunks, show_progress=True)
    print(f"  向量维度: {embeddings.shape[1]}")
    print(f"  向量数量: {embeddings.shape[0]}")

    # 创建或加载FAISS索引
    print(f"\n[5/7] 处理FAISS索引...")
    try:
        import faiss
    except ImportError:
        print("  错误: 请安装 faiss-cpu")
        print("    pip install faiss-cpu")
        return

    faiss_path = output_dir / "faiss.index"

    if append and faiss_path.exists():
        # 追加模式：加载现有索引
        print(f"  加载现有索引...")
        index = faiss.read_index(str(faiss_path))
        old_count = index.ntotal
        index.add(embeddings.astype('float32'))
        print(f"  追加向量: {old_count} -> {index.ntotal}")
    else:
        # 重建模式：创建新索引
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        print(f"  创建新索引: {index.ntotal} 个向量")

    # 保存FAISS索引
    faiss.write_index(index, str(faiss_path))
    print(f"  已保存: {faiss_path}")

    # 生成元数据
    print(f"\n[6/7] 生成元数据文件...")

    if append and existing_metadata:
        # 追加模式：合并现有元数据
        chunks_metadata = existing_metadata
        current_doc_id = max(m.get('doc_id', 0) for m in existing_metadata)
        vector_index = max(m.get('vector_index', 0) for m in existing_metadata) + 1
        print(f"  现有元数据: {len(chunks_metadata)} 条")
        print(f"  起始doc_id: {current_doc_id + 1}")
        print(f"  起始vector_index: {vector_index}")
    else:
        # 重建模式：新建元数据
        chunks_metadata = []
        current_doc_id = 0
        vector_index = 0

    for doc in all_documents:
        current_doc_id += 1
        doc_meta = doc['metadata']

        # 获取该文档的chunks
        doc_chunks = [c for c in all_chunks if c.source_file == doc_meta['filepath']]

        for chunk in doc_chunks:
            # 获取相对路径（从项目根目录开始）
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
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(chunks_metadata, f, ensure_ascii=False, indent=2)
    print(f"  已保存: {metadata_path} ({len(chunks_metadata)} 条)")

    # 汇总
    print(f"\n[7/7] 完成")
    print(f"\n{'='*70}")
    print(f"索引生成完成!")
    print(f"{'='*70}")
    print(f"  新增文档数量: {len(all_documents)}")
    print(f"  新增文本块数量: {len(all_chunks)}")
    print(f"  总向量数量: {index.ntotal}")
    print(f"  向量维度: {embeddings.shape[1]}")
    print(f"\n  输出文件:")
    print(f"    - {faiss_path}")
    print(f"    - {metadata_path}")
    if save_txt:
        print(f"\n  TXT文件目录: {txt_dir_path}")
    print(f"{'='*70}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='生成FAISS索引和元数据文件')
    parser.add_argument(
        '--input',
        '-i',
        type=str,
        default='.',
        help='输入目录（默认当前目录）'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default='./data',
        help='输出目录（默认./data）'
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
        '--append',
        '-a',
        action='store_true',
        help='追加模式：追加到现有索引而非重建'
    )
    parser.add_argument(
        '--save-txt',
        '-s',
        action='store_true',
        help='保存解析后的文本为txt文件'
    )
    parser.add_argument(
        '--txt-dir',
        '-t',
        type=str,
        default='./txt',
        help='txt文件保存目录（默认./txt）'
    )

    args = parser.parse_args()

    build_index(
        input_dir=args.input,
        output_dir=args.output,
        model_path=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        append=args.append,
        save_txt=args.save_txt,
        txt_dir=args.txt_dir
    )


if __name__ == "__main__":
    print(query_document_by_filename("GB_T 42392-2023 洁净手术部通用技术要求.pdf"))
