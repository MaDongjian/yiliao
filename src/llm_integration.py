# -*- coding: utf-8 -*-
"""
LLM 集成模块 - 离线大模型问答
支持千问、ChatGLM 等本地模型
"""

import os
from pathlib import Path
from typing import List, Dict, Optional


class OfflineLLM:
    """
    离线大模型基类
    """

    def __init__(self, model_path: str = None):
        """
        初始化离线模型

        Args:
            model_path: 模型路径
        """
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载模型（子类实现）"""
        raise NotImplementedError("请使用具体的模型类")

    def generate(self, prompt: str, context: str = "") -> str:
        """
        生成回答

        Args:
            prompt: 用户问题
            context: 检索到的上下文

        Returns:
            模型回答
        """
        raise NotImplementedError("请使用具体的模型类")


class OllamaLLM(OfflineLLM):
    """
    Ollama 本地模型 - 支持千问、ChatGLM等
    需要先安装 Ollama: https://ollama.com
    """

    def __init__(self, model_name: str = "qwen2.5:0.5b"):
        """
        初始化 Ollama 模型

        Args:
            model_name: 模型名称
                - 千问: "qwen2.5:0.5b", "qwen2.5:7b"
                - ChatGLM: "chatglm3:6b", "chatglm3:9b"
                - Llama3: "llama3:8b", "llama3:70b"
        """
        self.model_name = model_name
        self._check_ollama()

    def _check_ollama(self):
        """检查 Ollama 是否安装"""
        import subprocess
        try:
            result = subprocess.run(['ollama', '--version'],
                                  capture_output=True,
                                  text=True, timeout=5)
            if result.returncode == 0:
                print(f"Ollama 版本: {result.stdout.strip()}")
                return
        except Exception:
            pass

        raise Exception(
            "Ollama 未安装。请访问 https://ollama.com 下载安装。\n"
            "安装后运行: ollama pull qwen2.5:0.5b"
        )

    def _load_model(self):
        """Ollama 不需要预加载模型"""

    def generate(self, prompt: str, context: str = "") -> str:
        """
        使用 Ollama 生成回答

        Args:
            prompt: 用户问题
            context: 检索到的上下文

        Returns:
            模型回答
        """
        import subprocess
        import json

        # 构建完整提示词
        if context:
            full_prompt = f"""请根据以下参考信息回答问题。如果参考信息中没有答案，请说"抱歉，我没有找到相关信息"。

参考信息:
{context}

问题: {prompt}

回答:"""
        else:
            full_prompt = f"问题: {prompt}\n回答:"

        # 调用 Ollama API
        try:
            result = subprocess.run(
                ['ollama', 'run', self.model_name, full_prompt],
                capture_output=True,
                text=True,
                timeout=120000  # 2分钟超时
            )

            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            raise Exception("模型回答超时")
        except Exception as e:
            raise Exception(f"Ollama 调用失败: {e}")


class QwenLocalLLM(OfflineLLM):
    """
    千问本地模型 - 使用 transformers 库
    需要先下载千问模型
    """

    def __init__(self, model_path: str = "./models/qwen", quantization: str = "4bit"):
        """
        初始化千问本地模型

        Args:
            model_path: 模型路径
            quantization: 量化类型
                - "4bit": 4-bit量化（约5-6GB内存，推荐）
                - "8bit": 8-bit量化（约7-8GB内存）
                - "none": 不量化（约14GB内存）
        """
        # 转换为绝对路径并统一使用正斜杠
        self.model_path = Path(model_path).resolve()
        self.quantization = quantization
        # 使用正斜杠路径（避免 Windows 反斜杠问题）
        model_path_str = str(self.model_path).replace('\\', '/')
        super().__init__(model_path_str)

    def _load_model(self):
        """加载千问模型"""
        try:
            import torch
            import gc
        except ImportError:
            raise ImportError("请安装 torch: pip install torch")

        # 使用正斜杠路径
        model_path_str = str(self.model_path).replace('\\', '/')
        print(f"正在加载千问模型: {model_path_str}")
        print(f"量化设置: {self.quantization}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = model_path_str.lower()

        # 清理缓存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # 准备量化参数
        quantization_config = None
        if self.quantization in ["4bit", "8bit"]:
            try:
                from transformers import BitsAndBytesConfig

                if self.quantization == "4bit":
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    print("使用 4-bit 量化 (NF4)")
                else:  # 8bit
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        bnb_8bit_compute_dtype=torch.float16
                    )
                    print("使用 8-bit 量化")
            except ImportError:
                print("警告: bitsandbytes 未安装，将不使用量化")
                print("安装命令: pip install bitsandbytes")
                self.quantization = "none"

        try:
            # 检查是否为 VL 模型
            if 'vl' in model_name or 'vision' in model_name:
                # VL 模型使用 AutoModelForVision2Seq
                from transformers import AutoModelForVision2Seq, AutoProcessor

                self.tokenizer = AutoProcessor.from_pretrained(
                    model_path_str,
                    trust_remote_code=True,
                    local_files_only=True
                )

                load_kwargs = {
                    "trust_remote_code": True,
                    "device_map": "auto",
                    "local_files_only": True
                }

                # 量化配置（仅支持部分VL模型）
                if quantization_config and device == "cuda":
                    load_kwargs["quantization_config"] = quantization_config
                else:
                    load_kwargs["torch_dtype"] = torch.bfloat16 if device == "cuda" else torch.float32

                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_path_str,
                    **load_kwargs
                )
                self.is_vl_model = True
            else:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path_str,
                    trust_remote_code=True,
                    local_files_only=True
                )

                load_kwargs = {
                    "trust_remote_code": True,
                    "local_files_only": True
                }

                # Windows CUDA 环境：不使用 device_map="auto"，直接手动移动模型到 GPU
                if device == "cuda":
                    load_kwargs["torch_dtype"] = torch.float16
                else:
                    load_kwargs["device_map"] = "auto"

                # 添加量化配置
                if quantization_config and device == "cuda":
                    load_kwargs["quantization_config"] = quantization_config

                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path_str,
                    **load_kwargs
                )
                # 手动移动模型到 GPU（避免 device_map="auto" 在 Windows 上的问题）
                if device == "cuda":
                    print("正在移动模型到 GPU...")
                    self.model = self.model.to('cuda')
                    print(f"模型已移动到: {next(self.model.parameters()).device}")
                self.is_vl_model = False

            print(f"模型加载成功，设备: {device}, VL模型: {getattr(self, 'is_vl_model', False)}")

            # 显示内存使用情况
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"GPU内存: 已用 {memory_used:.2f}GB, 预留 {memory_reserved:.2f}GB")

        except Exception as e:
            raise Exception(f"模型加载失败: {e}")

        # 清理加载时的临时变量
        gc.collect()

    def generate(self, prompt: str, context: str = "", max_length: int = 2048, stream: bool = False, temperature: float = None, do_sample: bool = None):
        """
        使用千问模型生成回答（支持 Markdown 富文本格式和流式输出）

        Args:
            prompt: 用户问题
            context: 检索到的上下文
            max_length: 生成的最大长度（默认2048，支持长文本）
            stream: 是否使用流式输出（返回生成器）
            temperature: 采样温度（0.8-1.2 有随机性，None 使用默认）
            do_sample: 是否使用采样（True 有随机性，False 为贪婪解码）

        Returns:
            如果 stream=False: 模型回答字符串
            如果 stream=True: 生成器，每次产生 (text_chunk: str, is_finished: bool)
        """
        import torch
        import traceback

        # 系统提示词 - 详细回答
        SYSTEM_PROMPT = """你是医疗标准知识助手。请基于参考信息回答问题。

【核心原则】
1. 首先准确理解用户的问题意图
2. 严格基于参考文档中的信息回答
3. 确保回答内容与问题高度相关
4. 如果参考文档中没有相关信息，明确说明

【回答要求】
1. 必须在每个句子或信息后面标注来源编号，使用 [来源1] 或 [1] 格式
2. 引用不同来源时，必须使用换行分隔
3. 直接引用参考信息中的具体内容、标准值、技术要求
4. 保持回答的专业性和准确性

【表格格式特别说明】
如果使用 Markdown 表格格式回答：
- 表格第一列必须是"序号"
- 表格第二列是"来源文件"
- 第三列是"内容"，必须准确回答用户问题
- "序号"列必须按行填写数字：1、2、3...
- "序号"列格式：<sup class="source-ref" data-filename="对应文件名" data-ref="序号">序号</sup>
- "来源文件"列直接使用完整的文件名
- **重要**：内容列必须直接、准确地回答用户的问题，不要添加无关信息

表格示例：
| 序号 | 来源文件 | 内容 |
| --- | --- | --- |
| <sup class="source-ref" data-filename="医院感染诊断标准.pdf" data-ref="1">1</sup> | 医院感染诊断标准.pdf | 根据标准，医院感染诊断的具体要求是... |
| <sup class="source-ref" data-filename="医疗质量管理文件.pdf" data-ref="2">2</sup> | 医疗质量管理文件.pdf | 质量管理指标包括... |

注意：序号列中的 data-filename 必须与对应的来源文件名完全一致！"""

        try:
            # VL 模型生成
            if getattr(self, 'is_vl_model', False):
                # VL 模型使用特殊的输入格式
                if context:
                    if len(context) > 1500:
                        context = context[:1500] + "..."
                    user_content = f"{context}\n问题: {prompt}"
                else:
                    user_content = prompt

                # 构建 VL 模型的消息格式 - 纯文本输入
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ]

                # VL 模型需要使用 text 格式
                # 使用 apply_chat_template 并 tokenize=True，然后手动传入
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                # 对于 VL 模型，需要使用标准文本tokenizer处理
                # 获取底层的 tokenizer
                from transformers import AutoTokenizer
                if hasattr(self.tokenizer, 'tokenizer'):
                    base_tokenizer = self.tokenizer.tokenizer
                else:
                    # 创建一个新的纯文本 tokenizer
                    base_tokenizer = AutoTokenizer.from_pretrained(
                        str(self.model_path).replace('\\', '/'),
                        trust_remote_code=True,
                        local_files_only=True
                    )

                inputs = base_tokenizer(text, return_tensors="pt").to(self.model.device)
                input_length = inputs['input_ids'].shape[1]

                # 流式生成
                if stream:
                    # 确定采样策略
                    use_sample = do_sample if do_sample is not None else False
                    temp = temperature if temperature is not None else (0.9 if use_sample else 1.0)
                    return self._stream_generate(
                        self.model, inputs, base_tokenizer, input_length, max_length,
                        do_sample=use_sample, temperature=temp
                    )

                # 非流式生成
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=2048,  # 增加到2048以支持详细回答
                        do_sample=False,  # 贪婪解码
                        use_cache=True,
                        pad_token_id=base_tokenizer.eos_token_id if base_tokenizer.eos_token_id else base_tokenizer.pad_token_id,
                    )

                # 解码输出
                generated_ids = outputs[0][input_length:]
                response = base_tokenizer.decode(generated_ids, skip_special_tokens=True)
                return response.strip()

            # Qwen2+ 模型使用 Chat 格式
            # 优化上下文格式，减少 tokens
            if context:
                # 截断过长的上下文（最多保留 1500 字符）
                if len(context) > 1500:
                    context = context[:1500] + "..."
                user_content = f"{context}\n问题：{prompt}"
            else:
                user_content = f"问题：{prompt}"

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]

            # 使用 chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # 编码输入
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            input_length = inputs['input_ids'].shape[1]
            print(f"[DEBUG] 输入 tokens 数: {input_length}, 设备: {self.model.device}")

            # 流式生成 - GPU 模式可以使用更大的 max_length
            if stream:
                # GPU 模式下可以生成更多 tokens
                stream_max_length = max_length  # 使用完整长度
                # 确定采样策略
                use_sample = do_sample if do_sample is not None else False
                temp = temperature if temperature is not None else (0.9 if use_sample else 1.0)
                return self._stream_generate(
                    self.model, inputs, self.tokenizer, input_length, stream_max_length,
                    do_sample=use_sample, temperature=temp
                )

            # 非流式生成
            print(f"[DEBUG] 开始生成回答...")
            self.model.eval()  # 确保模型在评估模式

            with torch.no_grad():
                # 确定采样策略
                use_sample = do_sample if do_sample is not None else False
                temp = temperature if temperature is not None else (0.9 if use_sample else 1.0)

                # 构建生成参数
                generate_kwargs = {
                    "max_new_tokens": 2048,  # 增加到2048以支持详细回答
                    "use_cache": True,
                    "pad_token_id": self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }

                if use_sample:
                    # 使用采样，添加随机性
                    generate_kwargs.update({
                        "do_sample": True,
                        "temperature": temp,
                        "top_p": 0.9,
                        "top_k": 50,
                    })
                else:
                    # 贪婪解码（确定性）
                    generate_kwargs.update({
                        "do_sample": False,
                        "num_beams": 1,
                        "early_stopping": True,
                    })

                print(f"[DEBUG] 开始生成... (do_sample={use_sample}, temperature={temp})")
                outputs = self.model.generate(**inputs, **generate_kwargs)

            # 解码输出 - 跳过输入部分
            generated_ids = outputs[0][input_length:]
            print(f"[DEBUG] 生成完成，输出 tokens 数: {len(generated_ids)}")
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            return response.strip()

        except Exception as e:
            # 如果 chat template 失败，回退到简单格式
            try:
                if context:
                    text = f"请根据参考信息用Markdown格式回答问题（使用表格、列表等富文本格式）：\n\n参考信息:\n{context}\n\n问题: {prompt}"
                else:
                    text = f"问题: {prompt}\n\n请用Markdown格式回答（使用表格、列表等富文本格式）："

                inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
                input_length = inputs['input_ids'].shape[1]
                print(f"[DEBUG FALLBACK] 输入 tokens 数: {input_length}, 设备: {self.model.device}")

                if stream:
                    # 确定采样策略
                    use_sample = do_sample if do_sample is not None else False
                    temp = temperature if temperature is not None else (0.9 if use_sample else 1.0)
                    return self._stream_generate(
                        self.model, inputs, self.tokenizer, input_length, max_length,
                        do_sample=use_sample, temperature=temp
                    )

                print(f"[DEBUG FALLBACK] 开始生成...")
                self.model.eval()

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=2048,  # 增加到2048以支持详细回答
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id else self.tokenizer.pad_token_id,
                        num_beams=1,
                        early_stopping=True,
                    )

                generated_ids = outputs[0][input_length:]
                print(f"[DEBUG FALLBACK] 生成完成，输出 tokens 数: {len(generated_ids)}")
                return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            except Exception as e2:
                error_msg = f"千问模型生成失败: {e2}\n\n详细错误:\n{traceback.format_exc()}"
                raise Exception(error_msg)

    def _stream_generate(self, model, inputs, tokenizer, input_length, max_length, do_sample=False, temperature=1.0):
        """
        流式生成器的内部实现

        Args:
            model: 模型
            inputs: 输入tokens
            tokenizer: 分词器
            input_length: 输入长度
            max_length: 最大生成长度
            do_sample: 是否使用采样（添加随机性）
            temperature: 采样温度

        Yields:
            (text_chunk: str, is_finished: bool)
        """
        import torch
        from transformers import TextIteratorStreamer
        import threading
        import time
        import queue

        # 检测是否使用 GPU
        is_gpu = hasattr(model, 'device') and 'cuda' in str(model.device)

        # 生成配置
        generation_kwargs = {
            'max_new_tokens': max_length,  # 使用传入的完整长度
            'use_cache': True,
            'pad_token_id': tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id,
        }

        if do_sample:
            # 使用采样，添加随机性
            generation_kwargs.update({
                'do_sample': True,
                'temperature': temperature,
                'top_p': 0.9,
                'top_k': 50,
            })
        else:
            # 贪婪解码（确定性）
            generation_kwargs.update({
                'do_sample': False,
            })

        # 创建 streamer
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        generation_kwargs['streamer'] = streamer

        # 在后台线程中执行生成
        generation_thread = threading.Thread(
            target=model.generate,
            kwargs=dict(**inputs, **generation_kwargs),
            daemon=True
        )
        generation_thread.start()

        full_text = ""
        timeout = 120  # 增加超时时间
        start_time = time.time()

        try:
            # 直接访问 text_queue
            text_queue = streamer.text_queue

            # 根据设备调整轮询间隔
            poll_interval = 0.001 if is_gpu else 0.05

            print(f"[DEBUG] 流式生成 - 设备: {'GPU' if is_gpu else 'CPU'}, 超时: {timeout}s")

            # 使用非阻塞方式获取 tokens - 立即返回每个token
            while generation_thread.is_alive():
                try:
                    # 非阻塞获取
                    text_chunk = text_queue.get_nowait()
                    if text_chunk is None:  # stop_signal
                        break
                    # 立即返回，不累积
                    full_text += text_chunk
                    yield text_chunk, False

                except queue.Empty:
                    # 队列为空，短暂等待
                    if poll_interval > 0:
                        time.sleep(poll_interval)
                    # 超时检查
                    if time.time() - start_time > timeout:
                        print(f"流式生成超时 ({timeout}秒)")
                        break

            # 线程结束后，获取剩余数据
            try:
                while True:
                    text_chunk = text_queue.get_nowait()
                    if text_chunk is None:
                        break
                    full_text += text_chunk
                    yield text_chunk, False
            except queue.Empty:
                pass

        except Exception as e:
            print(f"流式生成错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            generation_thread.join(timeout=5)
            print(f"[DEBUG] 流式生成完成，full_text 长度: {len(full_text)}, 设备: {'GPU' if is_gpu else 'CPU'}")
            yield full_text, True


class RAGQA:
    """
    RAG问答系统 - 检索增强生成
    """

    def __init__(
        self,
        llm: OfflineLLM,
        index_dir: str = "./data/index"
    ):
        """
        初始化RAG问答系统

        Args:
            llm: 离线LLM实例
            index_dir: 索引目录
        """
        from src.search_model import SearchModel

        # 初始化检索系统
        print("正在初始化RAG问答系统...")
        self.search_model = SearchModel()
        self.search_model.initialize()

        self.llm = llm

        # 引用计数器（用于流式输出中连续编号）
        self._citation_counter = 0
        self._citation_sources = []  # 按引用顺序存储的sources

        print("RAG问答系统初始化完成！")

    def _should_use_table_format(self, question: str) -> bool:
        """
        检测问题是否需要表格格式输出

        Args:
            question: 用户问题

        Returns:
            True 如果需要表格格式，否则 False
        """
        # 表格关键词列表
        table_keywords = [
            '对比', '比较', '区别', '差异',
            '列表', '清单', '一览',
            '有哪些', '都有什么', '包括哪些',
            '表格', 'table',
            '分类', '种类', '类型',
            '各种', '多种',
            '步骤', '流程', '程序'
        ]

        question_lower = question.lower()
        for keyword in table_keywords:
            if keyword in question_lower:
                return True

        return False

    def _convert_citations_to_filename(self, text: str, sources: list = None) -> str:
        """
        将文本中的引用标记转换为文件名称格式
        例如: [来源1] -> [医疗质量管理与控制指标汇编6.0版2024.pdf]

        Args:
            text: 包含引用标记的文本
            sources: 来源列表，用于获取文件名
        """
        import re

        if sources is None or not sources:
            return text

        def replace_with_filename(match):
            """将引用标记替换为文件名称"""
            source_index = int(match.group(1)) - 1  # 转换为0-based索引
            if 0 <= source_index < len(sources):
                filename = sources[source_index].get('filename', f'来源{match.group(1)}')
                return f'[{filename}]'
            return match.group(0)  # 如果索引无效，保持原样

        # 替换所有引用标记
        text = re.sub(r'\[来源(\d+)\]', replace_with_filename, text)

        return text

    def _convert_citations(self, text: str, sources: list = None) -> str:
        """
        将文本中的引用标记转换为HTML格式，包含完整的原文内容
        每个引用标记按出现顺序分配连续编号1,2,3,4...

        例如: [来源1] -> <sup class="ref-link" data-ref="1" ...>1</sup>

        Args:
            text: 包含引用标记的文本
            sources: 来源列表，用于获取文件信息
        """
        import re
        import json
        import html

        # 如果没有提供 sources，使用简单的数字格式
        if sources is None or not sources:
            def simple_replace(match):
                self._citation_counter += 1
                new_ref = str(self._citation_counter)
                return f'<sup class="ref-link" data-ref="{new_ref}">{new_ref}</sup>'
            text = re.sub(r'\[来源(\d+)\]', simple_replace, text)
            return text

        def replace_citation(match):
            """替换单个引用"""
            original_ref = match.group(1)
            # 使用实例变量计数器，确保在流式输出中连续
            self._citation_counter += 1
            new_ref = str(self._citation_counter)

            # 从原始sources获取source信息
            try:
                source_index = int(original_ref) - 1  # 转换为0-based索引
                if 0 <= source_index < len(sources):
                    source = sources[source_index]
                    filename = source.get('filename', '未知来源')
                    filepath = source.get('filepath', '')
                    content = source.get('content', '')

                    content_json = json.dumps({
                        'filename': filename,
                        'filepath': filepath,
                        'content': content,
                        'ref': new_ref
                    }, ensure_ascii=False)

                    content_encoded = html.escape(content_json).replace('"', '&quot;')

                    # 添加到实例的sources列表
                    self._citation_sources.append(source)

                    content_preview = content[:80] + '...' if len(content) > 80 else content
                    title_attr = f"{filename}\n{content_preview}"

                    # 使用 f-string 确保标签完整（避免字符串拼接问题）
                    html_tag = (
                        f'<sup class="ref-link" '
                        f'data-ref="{new_ref}" '
                        f'data-filename="{html.escape(filename)}" '
                        f'data-filepath="{html.escape(filepath)}" '
                        f'data-content-json="{content_encoded}" '
                        f'title="{html.escape(title_attr)}"'
                        f'>{new_ref}</sup>'
                    )
                    return html_tag
            except (ValueError, IndexError):
                pass

            # 简单格式（确保完整）
            return f'<sup class="ref-link" data-ref="{new_ref}">{new_ref}</sup>'

        # 替换所有引用
        text = re.sub(r'\[来源(\d+)\]', replace_citation, text)

        return text

    def ask(self, question: str, top_k: int = 8, method: str = "semantic", temperature: float = None, do_sample: bool = None, min_score: float = 0.2) -> Dict:
        """
        提问并生成答案

        Args:
            question: 用户问题
            top_k: 检索的文档数量
            method: 搜索方法 ("semantic" 或 "keyword")
            temperature: 采样温度（0.8-1.2 有随机性）
            do_sample: 是否使用采样（True 有随机性）
            min_score: 最小相似度阈值（默认0.3，低于此值的结果将被过滤）

        Returns:
            {
                'question': str,
                'answer': str,
                'sources': List[Dict],  # 参考来源
                'success': bool
            }
        """
        try:
            # 重置引用计数器和sources列表
            self._citation_counter = 0
            self._citation_sources = []

            # 1. 检索相关文档（同时检索句子和文本块，获取更完整的信息）
            print(f"\n正在搜索相关文档...")

            # 句子级别搜索
            sentence_results = self.search_model.search(
                question,
                method=method,
                top_k=top_k,
                level="sentence"
            )

            # 文本块级别搜索（获取更多上下文）
            chunk_results = self.search_model.search(
                question,
                method=method,
                top_k=max(2, top_k - 1),
                level="chunk"
            )

            if not sentence_results and not chunk_results:
                return {
                    'question': question,
                    'answer': "抱歉，我没有找到与您的问题相关的信息。",
                    'sources': [],
                    'success': True
                }

            # 2. 构建上下文（使用完整的文本内容，不截断）
            context_parts = []
            raw_sources = []
            seen_texts = set()

            # 先添加句子级别的结果（更精准）- 过滤低相似度结果
            for i, r in enumerate(sentence_results, 1):
                # 过滤低于相似度阈值的结果
                if r['score'] < min_score:
                    continue
                text = r['text'].strip()
                if text and text not in seen_texts:
                    context_parts.append(f"[来源{i}] {text}")
                    raw_sources.append({
                        'filename': r['filename'],
                        'filepath': r['filepath'],
                        'content': text,  # 完整内容，不截断
                        'similarity': r['score']
                    })
                    seen_texts.add(text)

            # 再添加文本块级别的结果（更多上下文）- 过滤低相似度结果
            chunk_offset = len(context_parts)
            for i, r in enumerate(chunk_results, chunk_offset + 1):
                # 过滤低于相似度阈值的结果
                if r['score'] < min_score:
                    continue
                text = r['text'].strip()
                if text and text not in seen_texts and len(context_parts) < top_k + 2:
                    context_parts.append(f"[来源{i}] {text}")
                    raw_sources.append({
                        'filename': r['filename'],
                        'filepath': r['filepath'],
                        'content': text,  # 完整内容，不截断
                        'similarity': r['score']
                    })
                    seen_texts.add(text)

            # 合并相同文档，重新构建 context
            seen_docs = {}  # {(filename, filepath): merged_source}
            for source in raw_sources:
                key = (source['filename'], source['filepath'])
                if key in seen_docs:
                    seen_docs[key]['content'] += "\n\n" + source['content']
                else:
                    seen_docs[key] = source.copy()

            # 添加编号并重新构建 context
            sources = []
            merged_context_parts = []
            for idx, (key, source) in enumerate(seen_docs.items(), 1):
                source['index'] = idx
                source_with_number = source.copy()
                source_with_number['content'] = f"{source['filename']} {idx}：{source['content']}"
                sources.append(source_with_number)
                # 使用更清晰的格式
                merged_context_parts.append(
                    f"====== 参考文档 {idx} ======\n"
                    f"文件名：{source['filename']}\n"
                    f"编号：来源{idx}\n"
                    f"内容：{source['content']}\n"
                )

            context = "\n".join(merged_context_parts)

            # 3. 检测是否需要表格格式
            use_table_format = self._should_use_table_format(question)
            if use_table_format:
                # 在 context 前添加表格格式说明
                table_instruction = '''
【重要】请使用 Markdown 表格格式回答，使信息更加清晰易读。

表格格式要求：
1. 第一列标题必须是"来源文件"（不是"来源"）
2. "来源文件"列必须直接使用完整的文件名，不要使用"来源1"、"来源2"等编号
3. 文件名必须与参考信息"文件名："后面的一致

表格示例：
| 来源文件 | 内容 |
| --- | --- |
| 医院感染诊断标准.pdf | 相关内容... |
| 医疗质量管理文件.pdf | 相关内容... |
'''
                context = table_instruction + context
                print(f"检测到表格关键词，将使用表格格式回答")

            # 4. 生成回答
            print(f"正在生成回答...")
            answer = self.llm.generate(question, context, temperature=temperature, do_sample=do_sample)

            # 5. 检测是否使用表格格式
            use_table_format = self._should_use_table_format(question)

            # 6. 处理引用标记
            import re

            if sources:
                # 创建索引到文件名的映射
                index_to_filename = {}
                for source in sources:
                    idx = source.get('index')
                    if idx:
                        index_to_filename[idx] = source.get('filename', '')

                if use_table_format:
                    # 表格格式：只在表格内替换引用标记为文件名，表格外直接移除
                    # 检测 Markdown 表格
                    table_pattern = r'\|[^\n]+\|[\n\r]*\|[-:\s|]+\|[\n\r]*(?:\|[^\n]+\|[\n\r]*)+'

                    def replace_in_table(match):
                        """在表格内替换引用标记为文件名"""
                        table_text = match.group(0)

                        def replace_citation(m):
                            ref_num = int(m.group(1))
                            filename = index_to_filename.get(ref_num)
                            return filename if filename else ''

                        # 替换表格内的引用标记（带方括号）
                        table_text = re.sub(r'\[来源(\d+)\]', replace_citation, table_text)
                        table_text = re.sub(r'\[(\d+)\]', replace_citation, table_text)

                        # 替换表格内的"来源N"文本格式（不带方括号）
                        def replace_source_text(m):
                            ref_num = int(m.group(1))
                            filename = index_to_filename.get(ref_num)
                            # 格式：来源1 文件名.pdf（保留"来源"前缀）
                            return f'来源{ref_num} {filename}' if filename else m.group(0)

                        # 匹配"来源"后面跟数字的格式
                        table_text = re.sub(r'来源(\d+)', replace_source_text, table_text)

                        return table_text

                    # 只在表格内替换
                    answer = re.sub(table_pattern, replace_in_table, answer)

                    # 表格外直接移除所有引用标记
                    lines = answer.split('\n')
                    in_table = False
                    processed_lines = []

                    for line in lines:
                        # 检测是否在表格中
                        if '|' in line and line.strip().startswith('|'):
                            in_table = True
                        elif in_table and not line.strip():
                            in_table = False

                        if in_table:
                            # 表格内保持原样（已经替换过）
                            processed_lines.append(line)
                        else:
                            # 表格外移除引用标记
                            line = re.sub(r'\[来源(\d+)\]', '', line)
                            line = re.sub(r'\[(\d+)\]', '', line)
                            processed_lines.append(line)

                    answer = '\n'.join(processed_lines)
                else:
                    # 非表格格式：直接移除所有引用标记
                    answer = re.sub(r'\[来源(\d+)\]', '', answer)
                    answer = re.sub(r'\[(\d+)\]', '', answer)

            return {
                'question': question,
                'answer': answer,
                'sources': sources,
                'success': True
            }

        except Exception as e:
            return {
                'question': question,
                'answer': f"抱歉，生成答案时出错: {str(e)}",
                'sources': [],
                'success': False
            }

    def ask_stream(self, question: str, top_k: int = 8, method: str = "semantic", temperature: float = None, do_sample: bool = None, min_score: float = 0.2):
        """
        流式提问并生成答案

        Args:
            question: 用户问题
            top_k: 检索的文档数量
            method: 搜索方法 ("semantic" 或 "keyword")
            temperature: 采样温度（0.8-1.2 有随机性）
            do_sample: 是否使用采样（True 有随机性）
            min_score: 最小相似度阈值（默认0.3，低于此值的结果将被过滤）

        Yields:
            {
                'type': 'source' | 'content' | 'done' | 'error',
                'data': any
            }
        """
        # 重置引用计数器和sources列表
        self._citation_counter = 0
        self._citation_sources = []

        # 用于处理跨块的引用标记
        _buffer = ""
        _has_sources = False
        _sources_list = []

        try:
            # 1. 先返回检索阶段
            yield {'type': 'status', 'data': '正在搜索相关文档...'}

            # 2. 检索相关文档
            sentence_results = self.search_model.search(
                question,
                method=method,
                top_k=top_k,
                level="sentence"
            )

            chunk_results = self.search_model.search(
                question,
                method=method,
                top_k=max(2, top_k - 1),
                level="chunk"
            )

            if not sentence_results and not chunk_results:
                yield {'type': 'content', 'data': "抱歉，我没有找到与您的问题相关的信息。"}
                yield {'type': 'done', 'data': {'sources': []}}
                return

            # 3. 先合并相同文档，再构建上下文
            # 使用字典按 (filename, filepath) 合并，避免同一文档出现多次
            seen_docs = {}  # {(filename, filepath): merged_source}
            seen_texts = set()

            # 处理句子级别结果 - 过滤低相似度结果
            for r in sentence_results:
                # 过滤低于相似度阈值的结果
                if r['score'] < min_score:
                    continue
                text = r['text'].strip()
                if text and text not in seen_texts:
                    key = (r['filename'], r['filepath'])
                    if key in seen_docs:
                        # 合并内容
                        seen_docs[key]['content'] += "\n\n" + text
                        # 保留最高的相似度
                        if r['score'] > seen_docs[key]['similarity']:
                            seen_docs[key]['similarity'] = r['score']
                    else:
                        seen_docs[key] = {
                            'filename': r['filename'],
                            'filepath': r['filepath'],
                            'content': text,
                            'similarity': r['score']
                        }
                    seen_texts.add(text)

            # 处理文本块级别结果 - 过滤低相似度结果
            for r in chunk_results:
                # 过滤低于相似度阈值的结果
                if r['score'] < min_score:
                    continue
                text = r['text'].strip()
                if text and text not in seen_texts:
                    key = (r['filename'], r['filepath'])
                    if key in seen_docs:
                        # 合并内容
                        seen_docs[key]['content'] += "\n\n" + text
                        # 保留最高的相似度
                        if r['score'] > seen_docs[key]['similarity']:
                            seen_docs[key]['similarity'] = r['score']
                    else:
                        seen_docs[key] = {
                            'filename': r['filename'],
                            'filepath': r['filepath'],
                            'content': text,
                            'similarity': r['score']
                        }
                    seen_texts.add(text)

            # 添加编号并重新构建 context（使用合并后的编号）
            sources = []
            merged_context_parts = []
            for idx, (key, source) in enumerate(seen_docs.items(), 1):
                source['index'] = idx
                # content 格式：文件名 + 编号 + 内容
                # 例如：医疗质量管理与控制指标汇编6.0版2024.pdf 1：该文件提供了...
                source_with_number = source.copy()
                source_with_number['content'] = f"{source['filename']} {idx}：{source['content']}"
                sources.append(source_with_number)
                # 重新构建 context，使用更清晰的格式
                # 格式强调文件名，便于 LLM 在表格中直接使用
                merged_context_parts.append(
                    f"====== 参考文档 {idx} ======\n"
                    f"文件名：{source['filename']}\n"
                    f"编号：来源{idx}\n"
                    f"内容：{source['content']}\n"
                )

            context = "\n".join(merged_context_parts)

            # 4. 检测是否需要表格格式
            use_table_format = self._should_use_table_format(question)
            if use_table_format:
                # 在 context 前添加表格格式说明和问题重述
                table_instruction = f'''
【重要】请使用 Markdown 表格格式回答，使信息更加清晰易读。

用户问题：{question}

【回答要求】
1. 首先准确理解用户的问题意图
2. 严格基于参考文档中的信息回答
3. 确保回答内容与问题高度相关、准确具体
4. 内容列必须直接回答问题，不要泛泛而谈

表格格式要求：
1. 第一列标题必须是"序号"
2. 第二列标题是"来源文件"
3. 第三列标题是"内容"，必须准确回答用户问题
4. "序号"列必须按行填写数字：1、2、3...
5. "序号"列格式：<sup class="source-ref" data-filename="对应文件名" data-ref="序号">序号</sup>
6. "来源文件"列直接使用完整的文件名
7. 序号列中的 data-filename 必须与对应的来源文件名完全一致

表格示例：
| 序号 | 来源文件 | 内容 |
| --- | --- | --- |
| <sup class="source-ref" data-filename="医院感染诊断标准.pdf" data-ref="1">1</sup> | 医院感染诊断标准.pdf | 根据标准，具体要求是... |
| <sup class="source-ref" data-filename="医疗质量管理文件.pdf" data-ref="2">2</sup> | 医疗质量管理文件.pdf | 质量指标包括... |
'''
                context = table_instruction + context
                print(f"检测到表格关键词，将使用表格格式回答")

            # 5. 返回来源信息
            yield {'type': 'source', 'data': sources}
            _sources_list = sources
            _has_sources = True

            # 6. 流式生成回答
            yield {'type': 'status', 'data': '正在生成回答...'}

            full_answer = ""
            full_answer_processed = ""  # 保存处理后的完整答案

            for text_chunk, is_finished in self.llm.generate(question, context, stream=True, temperature=temperature, do_sample=do_sample):
                if text_chunk and not is_finished:
                    # 使用缓冲区处理跨块的引用
                    _buffer += text_chunk
                    # 处理时暂时不处理表格（因为表格可能不完整）
                    processed = self._process_buffer(_buffer, _sources_list if _has_sources else None, False)
                    # 发送已处理的部分
                    if processed['output']:
                        yield {'type': 'content', 'data': processed['output']}
                        full_answer_processed += processed['output']  # 累加处理后的内容
                    # 保留未处理的部分
                    _buffer = processed['remaining']
                    full_answer += text_chunk
                if is_finished:
                    # 处理剩余的缓冲区内容
                    if _buffer:
                        processed = self._process_buffer(_buffer, _sources_list if _has_sources else None, False)
                        if processed['output']:
                            yield {'type': 'content', 'data': processed['output']}
                            full_answer_processed += processed['output']  # 累加最后的处理内容
                        _buffer = ""
                    if text_chunk and not full_answer:
                        full_answer = text_chunk
                    break

            # 7. 流式输出结束
            import sys
            print(f"[DEBUG] ========== 流式结束 ==========", file=sys.stderr)
            print(f"[DEBUG] full_answer原始长度: {len(full_answer)}", file=sys.stderr)
            print(f"[DEBUG] full_answer_processed长度: {len(full_answer_processed)}", file=sys.stderr)
            print(f"[DEBUG] full_answer_processed前500字符:\n{full_answer_processed[:500]}", file=sys.stderr)

            # 8. 完成，返回处理后的最终答案（使用处理后的版本）
            final_answer = full_answer_processed if full_answer_processed else full_answer
            yield {'type': 'done', 'data': {'sources': sources, 'full_answer': final_answer}}

        except Exception as e:
            yield {'type': 'error', 'data': str(e)}

    def _extract_table_part(self, text: str, sources: list) -> str:
        """
        提取表格部分，将"来源N"替换为 sup 标签 + 文件名

        Args:
            text: 完整文本内容
            sources: 来源列表

        Returns:
            处理后的表格部分，如果没有表格则返回空字符串
        """
        import re
        import sys

        try:
            # 创建索引到文件名的映射
            index_to_filename = {}
            for source in sources:
                idx = source.get('index')
                if idx:
                    index_to_filename[idx] = source.get('filename', '')

            sys.stderr.write(f"[DEBUG TABLE] ===== 开始处理表格 =====\n")
            sys.stderr.write(f"[DEBUG TABLE] index_to_filename={index_to_filename}\n")
            sys.stderr.write(f"[DEBUG TABLE] 完整文本长度: {len(text)}\n")
            sys.stderr.write(f"[DEBUG TABLE] 完整文本前500字符: {text[:500]}\n")
            sys.stderr.flush()

            # 检测 Markdown 表格
            table_pattern = r'(\|[^\n]+\|[\n\r]*\|[-:\s|]+\|[\n\r]*(?:\|[^\n]+\|[\n\r]*)+)'

            # 先查找所有表格
            tables = re.findall(table_pattern, text)
            sys.stderr.write(f"[DEBUG TABLE] 找到 {len(tables)} 个表格\n")
            sys.stderr.flush()

            if tables:
                # 安全地打印表格内容，避免编码问题
                table_content = tables[0]
                sys.stderr.write(f"[DEBUG TABLE] 第一个表格内容长度: {len(table_content)}\n")
                sys.stderr.write(f"[DEBUG TABLE] 表格前300字符: {table_content[:300]}\n")
                sys.stderr.flush()

                def replace_in_table(table_text):
                    """在表格内替换引用标记为 sup 标签 + 文件名"""
                    # table_text 是字符串，不需要调用 match.group()

                    sys.stderr.write(f"[DEBUG TABLE] replace_in_table 被调用\n")
                    sys.stderr.flush()

                    # 检查"来源1"是否存在于表格中
                    if '来源1' in table_text:
                        sys.stderr.write(f"[DEBUG TABLE] ✓ 表格中包含'来源1'\n")
                    else:
                        sys.stderr.write(f"[DEBUG TABLE] ✗ 表格中不包含'来源1'\n")
                        sys.stderr.write(f"[DEBUG TABLE] 表格中的实际内容: {repr(table_text[:100])}\n")
                    sys.stderr.flush()

                    # 测试正则表达式
                    test_matches = re.findall(r'来源(\d+)', table_text)
                    sys.stderr.write(f"[DEBUG TABLE] 正则匹配结果: {test_matches}\n")
                    sys.stderr.flush()

                    # 替换表格内的"来源N"文本格式（不带方括号）
                    def replace_source_text(m):
                        ref_num = int(m.group(1))
                        filename = index_to_filename.get(ref_num, '')
                        if filename:
                            # 格式：来源1 文件名.pdf（保留"来源"前缀）
                            result = f'来源{ref_num} {filename}'
                            sys.stderr.write(f"[DEBUG TABLE] 替换 来源{ref_num} -> {result}\n")
                            return result
                        return m.group(0)

                    # 匹配"来源"后面跟数字的格式
                    table_text = re.sub(r'来源(\d+)', replace_source_text, table_text)

                    sys.stderr.write(f"[DEBUG TABLE] ✓ 表格处理完成\n")
                    sys.stderr.flush()
                    return table_text

                # 处理第一个表格
                result = replace_in_table(table_content)
                sys.stderr.write(f"[DEBUG TABLE] 返回结果长度: {len(result)}\n")
                sys.stderr.write(f"[DEBUG TABLE] ===== 表格处理完成 =====\n")
                sys.stderr.flush()
                return result

            sys.stderr.write(f"[DEBUG TABLE] 没有找到表格！\n")
            sys.stderr.write(f"[DEBUG TABLE] ===== 表格处理完成 =====\n")
            sys.stderr.flush()
            return ""

        except Exception as e:
            sys.stderr.write(f"[DEBUG TABLE] ❌ 处理表格时发生异常: {e}\n")
            import traceback
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            return ""

    def _process_table_format(self, text: str, sources: list) -> str:
        """
        对完整文本进行表格格式处理
        1. 将表格中的"来源N"替换为文件名
        2. 然后移除整个表格部分

        Args:
            text: 完整的文本内容
            sources: 来源列表

        Returns:
            处理后的文本（不包含表格）
        """
        import re
        import sys

        # 创建索引到文件名的映射
        index_to_filename = {}
        for source in sources:
            idx = source.get('index')
            if idx:
                index_to_filename[idx] = source.get('filename', '')

        print(f"[DEBUG TABLE] index_to_filename={index_to_filename}", file=sys.stderr)
        print(f"[DEBUG TABLE] text前300字符: {text[:300]}", file=sys.stderr)

        # 检测 Markdown 表格
        table_pattern = r'\|[^\n]+\|[\n\r]*\|[-:\s|]+\|[\n\r]*(?:\|[^\n]+\|[\n\r]*)+'

        def replace_in_table(match):
            """在表格内替换引用标记为文件名"""
            table_text = match.group(0)

            print(f"[DEBUG TABLE] 找到表格: {table_text[:200]}...", file=sys.stderr)

            def replace_citation(m):
                ref_num = int(m.group(1))
                filename = index_to_filename.get(ref_num)
                return filename if filename else ''

            # 替换表格内的引用标记（带方括号）
            table_text = re.sub(r'\[来源(\d+)\]', replace_citation, table_text)
            table_text = re.sub(r'\[(\d+)\]', replace_citation, table_text)

            # 替换表格内的"来源N"文本格式（不带方括号）
            def replace_source_text(m):
                ref_num = int(m.group(1))
                filename = index_to_filename.get(ref_num)
                result = filename if filename else m.group(0)
                print(f"[DEBUG TABLE] 替换 来源{ref_num} -> {result}", file=sys.stderr)
                return result

            # 匹配"来源"后面跟数字的格式
            table_text = re.sub(r'来源(\d+)', replace_source_text, table_text)

            print(f"[DEBUG TABLE] 表格处理后: {table_text[:200]}...", file=sys.stderr)

            return table_text

        # 先替换表格中的引用标记
        result = re.sub(table_pattern, replace_in_table, text)

        # 然后移除所有表格部分
        result = re.sub(table_pattern, '', result)

        # 移除表格前后的空行
        result = re.sub(r'\n\s*\n\s*\n', '\n\n', result)

        # 移除所有剩余的引用标记（文本内容中的引用）
        result = re.sub(r'\[来源(\d+)\]', '', result)
        result = re.sub(r'\[(\d+)\]', '', result)

        print(f"[DEBUG TABLE] 最终结果前200字符: {result[:200]}", file=sys.stderr)

        return result

    def _process_buffer(self, text: str, sources: list = None, use_table_format: bool = False) -> dict:
        """
        处理缓冲区，返回已处理和剩余部分
        只移除引用标记，不处理表格（表格在流式结束后统一处理）

        Args:
            text: 待处理的文本
            sources: 来源列表
            use_table_format: 是否使用表格格式（此参数已废弃，保留用于兼容）

        Returns:
            {'output': str, 'remaining': str}
        """
        import re
        import sys

        # 查找最后一个未闭合的引用标记
        incomplete_pattern = r'\[来[^\]]*$|\[来源\d*\[?$'
        incomplete_match = re.search(incomplete_pattern, text)

        if incomplete_match:
            # 有未完成的引用，分割文本
            split_pos = incomplete_match.start()
            output_part = text[:split_pos]
            remaining_part = text[split_pos:]
        else:
            # 没有未完成的引用，全部处理
            output_part = text
            remaining_part = ""

        # 移除所有引用标记（表格在流式结束后统一处理）
        output_part = re.sub(r'\[来源(\d+)\]', '', output_part)
        output_part = re.sub(r'\[(\d+)\]', '', output_part)

        return {'output': output_part, 'remaining': remaining_part}


def ask_question(
    question: str,
    top_k: int = 3,
    model: str = "ollama",
    model_name: str = "qwen2.5:0.5b",
    model_path: str = "./models/qwen",
    quantization: str = "4bit"
):
    """
    便捷函数：提问并获取答案

    Args:
        question: 用户问题
        top_k: 检索的文档数量
        model: 模型类型 ("ollama" 或 "qwen")
        model_name: Ollama 模型名称
        model_path: 本地 Qwen 模型路径
        quantization: 量化类型 ("4bit", "8bit", "none")，仅对本地模型有效

    Returns:
        {
            'question': str,
            'answer': str,
            'sources': List[Dict],
            'success': bool
        }

    示例:
        >>> # 使用 Ollama
        >>> result = ask_question("什么是驻场人员？", model="ollama")
        >>> # 使用本地 7B 模型（4-bit 量化）
        >>> result = ask_question("什么是驻场人员？", model="qwen",
        ...                      model_path="./models/Qwen2.5-7B-Instruct",
        ...                      quantization="4bit")
        >>> print(result['answer'])
    """
    # 创建LLM实例
    if model == "ollama":
        llm = OllamaLLM(model_name=model_name)
    elif model == "qwen":
        llm = QwenLocalLLM(model_path=model_path, quantization=quantization)
    else:
        raise ValueError(f"不支持的模型类型: {model}")

    # 创建RAG系统
    rag = RAGQA(llm)

    # 提问
    return rag.ask(question, top_k=top_k)


if __name__ == "__main__":
    # 测试代码
    print("=" * 70)
    print("RAG问答系统测试")
    print("=" * 70)

    try:
        # 测试问答
        result = ask_question("什么是驻场人员？", model="ollama", top_k=2)

        print(f"\n问题: {result['question']}")
        print(f"\n答案:\n{result['answer']}")
        print(f"\n参考来源 ({len(result['sources'])} 个):")
        for i, source in enumerate(result['sources'], 1):
            print(f"\n[{i}] {source['filename']} (相似度: {source['similarity']:.4f})")
            print(f"    {source['content'][:100]}...")

    except Exception as e:
        print(f"\n错误: {e}")
        print("\n请确保：")
        print("1. 安装 Ollama: https://ollama.com")
        print("2. 下载模型: ollama pull qwen2.5:0.5b")
