# 文档检索系统

支持 Word、PDF、PPT 的离线文档向量化检索系统

## 功能特性

- 支持多种文档格式: Word (.docx), PDF (.pdf), PowerPoint (.pptx)
- 多语言支持: 使用 paraphrase-multilingual 模型，支持中英文
- 离线运行: 完全本地化，无需联网
- 语义搜索: 基于向量相似度的智能检索
- 关键词搜索: 精确关键词匹配
- 混合搜索: 结合语义和关键词的混合检索
- 增量添加: 支持持续追加新文档

## 目录结构

```
E:\worktool\yiliaozsk\
├── file_info/          # 存放待处理的文档
├── data/
│   ├── index/          # FAISS向量索引
│   └── db/             # SQLite元数据数据库
├── models/             # 离线模型存储
├── src/                # 源代码
│   ├── parsers.py      # 文档解析器
│   ├── chunker.py      # 文本分块器
│   ├── embedder.py     # 向量化模型
│   ├── storage.py      # 向量存储
│   ├── retriever.py    # 检索引擎
│   └── main.py         # 命令行入口
├── quick_start.py      # 快速开始脚本
├── requirements.txt    # 依赖包
└── README.md           # 本文件
```

## 安装依赖

```bash
cd E:\worktool\yiliaozsk
pip install -r requirements.txt
```

## 快速开始

### 方法1: 使用快速开始脚本

```bash
python quick_start.py
```

### 方法2: 使用命令行

1. 添加文档（单个文件）
```bash
python src\main.py add file_info\document.docx
```

2. 添加文档（整个目录）
```bash
python src\main.py add file_info
```

3. 搜索文档
```bash
# 语义搜索
python src\main.py search "机器学习"

# 关键词搜索
python src\main.py search "深度学习" --method keyword

# 混合搜索
python src\main.py hybrid-search "神经网络"
```

4. 查看统计
```bash
python src\main.py stats
```

## 命令参考

### 添加文档

```bash
python src\main.py add <路径> [选项]

选项:
  --extensions      文件扩展名，逗号分隔 (默认: .pdf,.docx,.pptx)
  --chunk-size      分块大小 (默认: 512)
  --chunk-overlap   分块重叠 (默认: 50)
  --model           模型名称
```

### 搜索文档

```bash
python src\main.py search <查询词> [选项]

选项:
  --top-k           返回结果数 (默认: 5)
  --method          搜索方法: semantic(语义) 或 keyword(关键词)
  --min-score       最小相似度分数
  --no-score        不显示相似度分数
```

### 混合搜索

```bash
python src\main.py hybrid-search <查询词> [选项]

选项:
  --top-k               返回结果数 (默认: 5)
  --semantic-weight     语义搜索权重 (默认: 0.7)
  --keyword-weight      关键词搜索权重 (默认: 0.3)
```

### 其他命令

```bash
# 查看统计
python src\main.py stats

# 清空索引
python src\main.py clear --force

# 列出推荐模型
python src\main.py download-model --list-models

# 下载模型
python src\main.py download-model
```

## 使用示例

### Python API 使用

```python
from src.retriever import DocumentRetriever

# 初始化
retriever = DocumentRetriever()
retriever.initialize()

# 添加文档
retriever.add_document("file_info/document.pdf")

# 搜索
results = retriever.search("人工智能", top_k=5)

for r in results:
    print(f"文件: {r['filename']}")
    print(f"相似度: {r['score']:.4f}")
    print(f"内容: {r['text'][:200]}")
    print("-" * 40)
```

## 离线模型使用

首次运行会自动从 HuggingFace 下载模型，约 470MB。

完全离线环境部署:

1. 在有网络的环境下载模型:
```bash
python src\main.py download-model
```

2. 将 `models` 目录复制到离线环境

## 支持的模型

| 模型 | 描述 | 大小 |
|------|------|------|
| paraphrase-multilingual-MiniLM-L12-v2 | 多语言(中英文)，默认模型 | ~470MB |
| paraphrase-multilingual-Mpnet-Base-v2 | 多语言，效果更好 | ~1.1GB |
| BAAI/bge-small-zh-v1.5 | 中文优化，推荐 | ~100MB |
| shibing624/text2vec-base-chinese | 中文优化 | ~400MB |

## 技术架构

- **文档解析**: python-docx, pdfplumber, python-pptx
- **文本分块**: 智能段落分割 + 重叠窗口
- **向量化**: sentence-transformers
- **向量索引**: FAISS
- **元数据存储**: SQLite
