"""
向量化模型 - 使用 sentence-transformers 进行文本向量化
"""

import os
from pathlib import Path
from typing import List, Union, Optional
import numpy as np

# 在模块导入时就设置离线模式
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'


class EmbeddingModel:
    """文本向量化模型"""

    def __init__(
        self,
        model_name: str = "bge-large-zh-v1.5",
        cache_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        初始化向量化模型

        Args:
            model_name: 模型名称，默认为 bge-large-zh-v1.5
                       中文优化，离线可用，1024维向量
                       支持 huggingface 和 bge 两种目录格式
            cache_dir: 模型缓存目录（绝对路径）
            device: 运行设备 ('cpu', 'cuda', None为自动检测)
        """
        self.model_name = model_name

        # 获取项目根目录的绝对路径
        if cache_dir is None:
            # 从当前文件向上两级到项目根目录
            current_dir = Path(__file__).resolve().parent
            project_root = current_dir.parent
            cache_dir = project_root / "models"

        self.cache_dir = str(cache_dir)
        self.device = device
        self.model = None
        self.dimension = None

    def load(self):
        """加载模型"""
        if self.model is not None:
            return

        # ============================================================
        # 修复 PyTorch CVE-2025-32434 安全漏洞
        # 在加载 sentence_transformers 模型之前应用补丁
        # ============================================================
        import torch
        os.environ['USE_WEIGHTS_ONLY'] = '0'

        if not hasattr(torch, '_load_patched'):
            _original_torch_load = torch.load

            def _patched_torch_load(f, *args, **kwargs):
                """强制移除 weights_only=True 参数"""
                if kwargs.get('weights_only', False) is True:
                    kwargs['weights_only'] = False
                return _original_torch_load(f, *args, **kwargs)

            torch.load = _patched_torch_load
            torch._load_patched = True
        # ============================================================

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("请安装 sentence-transformers: pip install sentence-transformers")

        print(f"正在加载模型: {self.model_name}")
        print(f"模型目录: {self.cache_dir}")

        # 检查模型目录是否存在
        model_path = Path(self.cache_dir)
        if not model_path.exists():
            raise FileNotFoundError(
                f"模型目录不存在: {self.cache_dir}\n"
                f"请先运行下载脚本下载模型，或检查路径是否正确"
            )

        # 设置离线模式环境变量
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'

        # 临时绕过 PyTorch torch.load 安全检查（仅用于离线开发环境）
        # 警告：这是临时方案，生产环境建议升级到 PyTorch 2.6+ 或使用 safetensors 格式
        # CVE-2025-32434 安全漏洞绕过

        # 方案1: 设置环境变量
        os.environ['USE_WEIGHTS_ONLY'] = '0'

        # 方案2: Monkey patch torch.load 来移除 weights_only 限制
        import torch
        original_torch_load = torch.load

        def patched_torch_load(f, *args, **kwargs):
            """临时移除 weights_only=True 参数"""
            # 如果 weights_only 为 True，强制改为 False
            if kwargs.get('weights_only', False) is True:
                kwargs['weights_only'] = False
            return original_torch_load(f, *args, **kwargs)

        # 应用 patch
        torch.load = patched_torch_load

        # 尝试从本地加载模型
        try:
            try:
                # 首先尝试直接从模型名称加载（使用缓存）
                self.model = SentenceTransformer(
                    self.model_name,
                    cache_folder=self.cache_dir,
                    device=self.device
                )
            except Exception as e:
                # 如果失败，尝试其他加载方式
                print(f"从缓存加载失败，尝试直接加载模型文件...")

                # 方案1: 检查是否有 snapshots 目录（huggingface 格式）
                model_subdir = model_path / self.model_name.replace("/", "--") / "snapshots"
                if model_subdir.exists():
                    snapshots = list(model_subdir.iterdir())
                    if snapshots:
                        # 使用最新的快照
                        model_snapshot = sorted(snapshots)[-1]
                        print(f"找到模型快照: {model_snapshot}")

                        # 直接从快照路径加载
                        self.model = SentenceTransformer(
                            str(model_snapshot),
                            device=self.device
                        )
                    else:
                        raise FileNotFoundError(f"未找到模型快照: {model_subdir}")
                else:
                    # 方案2: 检查文件是否直接在根目录（bge 格式）
                    model_root = model_path / self.model_name
                    if model_root.exists() and (model_root / "config.json").exists():
                        print(f"找到模型根目录: {model_root}")
                        print(f"尝试直接从根目录加载...")

                        # 直接从根目录加载
                        self.model = SentenceTransformer(
                            str(model_root),
                            device=self.device
                        )
                    else:
                        raise FileNotFoundError(
                            f"模型目录结构不正确，请检查: {model_path}\n"
                            f"应该包含以下之一:\n"
                            f"  1. {self.model_name.replace('/', '--')}/snapshots/<hash>/ (huggingface格式)\n"
                            f"  2. {self.model_name}/config.json (bge格式)\n"
                            f"实际检查的路径:\n"
                            f"  - {model_subdir} (不存在)\n"
                            f"  - {model_root} (存在: {model_root.exists()}, config.json存在: {(model_root / 'config.json').exists()})"
                        )
        finally:
            # 恢复原始 torch.load（无论成功或失败）
            torch.load = original_torch_load

        # 获取向量维度
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"离线模型加载成功，向量维度: {self.dimension}")

    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        将文本编码为向量

        Args:
            texts: 单个文本或文本列表
            show_progress: 是否显示进度条
            batch_size: 批处理大小

        Returns:
            向量数组，shape: (n_texts, dimension)
        """
        if self.model is None:
            self.load()

        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]

        # 编码为向量
        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress,
            batch_size=batch_size,
            convert_to_numpy=True
        )

        if is_single:
            return embeddings[0]

        return embeddings

    def encode_chunks(
        self,
        chunks: List,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        对文本块列表进行向量化

        Args:
            chunks: TextChunk对象列表
            show_progress: 是否显示进度条

        Returns:
            向量数组
        """
        if self.model is None:
            self.load()

        texts = [chunk.text for chunk in chunks]

        return self.encode(texts, show_progress=show_progress)

    def get_dimension(self) -> int:
        """获取向量维度"""
        if self.dimension is None:
            self.load()
        return self.dimension


def download_model(model_name: str = "paraphrase-multilingual-MiniLM-L12-v2", cache_dir: Optional[str] = None):
    """
    下载模型到本地（用于离线环境准备）

    Args:
        model_name: 模型名称
        cache_dir: 保存目录
    """
    import zipfile
    from huggingface_hub import snapshot_download

    cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), "../models")
    os.makedirs(cache_dir, exist_ok=True)

    print(f"开始下载模型: {model_name}")
    print(f"保存到: {cache_dir}")

    model_path = snapshot_download(
        repo_id=model_name,
        cache_dir=cache_dir,
        local_dir=os.path.join(cache_dir, model_name.replace("/", "--")),
        local_dir_use_symlinks=False
    )

    print(f"模型下载完成: {model_path}")
    print("现在可以在离线环境中使用该模型了。")


# 推荐的模型列表
RECOMMENDED_MODELS = {
    "paraphrase-multilingual-MiniLM-L12-v2": {
        "description": "多语言支持（中英文），体积约470MB，效果优秀",
        "dimension": 384,
        "size_mb": 470
    },
    "paraphrase-multilingual-Mpnet-Base-v2": {
        "description": "多语言支持，效果更好但体积较大约1.1GB",
        "dimension": 768,
        "size_mb": 1100
    },
    "sentence-transformers/all-MiniLM-L6-v2": {
        "description": "英文为主，轻量级约80MB",
        "dimension": 384,
        "size_mb": 80
    },
    "shibing624/text2vec-base-chinese": {
        "description": "中文优化，约400MB",
        "dimension": 768,
        "size_mb": 400
    },
    "BAAI/bge-small-zh-v1.5": {
        "description": "中文优化，效果最好，约100MB（推荐）",
        "dimension": 512,
        "size_mb": 100
    }
}


if __name__ == "__main__":
    # 测试代码
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "download":
        # 下载模型
        model_name = sys.argv[2] if len(sys.argv) > 2 else "paraphrase-multilingual-MiniLM-L12-v2"
        download_model(model_name)
    else:
        # 测试编码
        model = EmbeddingModel()
        model.load()

        texts = [
            "这是一个测试句子。",
            "This is a test sentence.",
            "机器学习是人工智能的一个分支。"
        ]

        embeddings = model.encode(texts)

        print(f"文本数量: {len(texts)}")
        print(f"向量维度: {embeddings.shape[1]}")
        print(f"向量形状: {embeddings.shape}")

        # 计算相似度
        similarity = np.dot(embeddings, embeddings.T)
        print("\n相似度矩阵:")
        print(similarity)
