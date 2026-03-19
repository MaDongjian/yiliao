# -*- coding: utf-8 -*-
"""
LLM 集成模块 - 离线大模型问答
支持千问、ChatGLM 等本地模型
"""

import os
import sys
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
                timeout=300000  # 5分钟超时（增加）
            )

            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            raise Exception("模型回答超时")
        except Exception as e:
            raise Exception(f"Ollama 调用失败: {e}")

    def generate(
        self,
        prompt: str,
        context: str = "",
        max_length: int = 2048,
        stream: bool = False,
        temperature: float = None,
        do_sample: bool = None
    ):
        """
        使用 Ollama 生成回答（支持流式输出）

        Args:
            prompt: 用户问题
            context: 检索到的上下文
            max_length: 生成的最大长度
            stream: 是否使用流式输出
            temperature: 采样温度（Ollama API 支持）
            do_sample: 是否使用采样

        Returns:
            如果 stream=False: 模型回答字符串
            如果 stream=True: 生成器，每次产生 (text_chunk: str, is_finished: bool)
        """
        import subprocess
        import json

        # 构建完整提示词
        if context:
            full_prompt = f"""你是医疗标准知识助手。请根据以下参考信息回答问题。

参考信息:
{context}

问题: {prompt}

回答:"""
        else:
            full_prompt = f"问题: {prompt}\n回答:"

        if not stream:
            # 非流式生成（原有逻辑）
            try:
                result = subprocess.run(
                    ['ollama', 'run', self.model_name, full_prompt],
                    capture_output=True,
                    text=True,
                    timeout=300000  # 5分钟超时
                )
                return result.stdout.strip()
            except subprocess.TimeoutExpired:
                raise Exception("模型回答超时")
            except Exception as e:
                raise Exception(f"Ollama 调用失败: {e}")
        else:
            # 流式生成
            return self._generate_stream(full_prompt, temperature)

    def _generate_stream(self, prompt: str, temperature: float = None):
        """
        Ollama 流式生成（使用 HTTP API）

        Args:
            prompt: 完整提示词
            temperature: 采样温度

        Yields:
            (text_chunk: str, is_finished: bool)
        """
        import requests
        import json

        # Ollama HTTP API 端点
        url = "http://localhost:11434/api/generate"

        # 构建请求
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": 2048,  # 最大生成 token 数
                "temperature": temperature if temperature else 0.7,
                "top_p": 0.9,
            }
        }

        try:
            # 发送流式请求
            response = requests.post(url, json=data, stream=True, timeout=300)  # 5分钟超时
            response.raise_for_status()

            # 逐行读取响应
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')

                    # Ollama API 返回的 JSON 行可能没有 "data: " 前缀
                    # 如果有前缀则移除，否则直接解析
                    if line.startswith('data: '):
                        json_str = line[6:]
                    else:
                        json_str = line

                    try:
                        chunk_data = json.loads(json_str)
                        text_chunk = chunk_data.get('response', '')
                        done = chunk_data.get('done', False)

                        if text_chunk:
                            yield (text_chunk, done)

                        if done:
                            # 流式结束
                            break
                    except json.JSONDecodeError:
                        continue

        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama API 调用失败: {e}")
        except Exception as e:
            raise Exception(f"Ollama 流式生成出错: {e}")


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

        # ============================================================
        # 修复 PyTorch CVE-2025-32434 安全漏洞
        # 在加载 transformers 模型之前应用补丁
        # ============================================================
        import os
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

                # GPU 环境配置
                if device == "cuda":
                    # 使用 device_map="auto" 自动分配到 GPU（推荐方式）
                    load_kwargs["device_map"] = "auto"
                    load_kwargs["torch_dtype"] = torch.float16

                    # 添加量化配置（GPU上推荐使用4bit量化）
                    if quantization_config:
                        load_kwargs["quantization_config"] = quantization_config
                        print(f"✓ GPU模式：启用{self.quantization}量化")
                    else:
                        print("✓ GPU模式：使用FP16精度（未启用量化）")
                else:
                    # CPU 模式
                    load_kwargs["device_map"] = "auto"
                    load_kwargs["torch_dtype"] = torch.float32
                    print("⚠ CPU模式：生成速度会很慢，建议使用GPU")

                print(f"模型加载配置: {list(load_kwargs.keys())}")

                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path_str,
                    **load_kwargs
                )

                # 验证模型确实在正确的设备上
                if device == "cuda":
                    model_device = next(self.model.parameters()).device
                    print(f"✓ 模型已加载到设备: {model_device}")
                    if 'cuda' not in str(model_device):
                        print(f"❌ 警告：期望GPU但模型在 {model_device}")
                        # 强制移动到GPU
                        print("正在强制移动模型到GPU...")
                        self.model = self.model.to('cuda')
                        model_device = next(self.model.parameters()).device
                        print(f"✓ 模型已移动到: {model_device}")
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
- **【最关键】"来源文件"列必须完全复制参考文档中的"文件名：xxx"，不能有任何修改或简化！**
- 内容列必须直接、准确地回答用户的问题，不要添加无关信息

表格示例：
| 序号 | 来源文件 | 内容 |
| --- | --- | --- |
| <sup class="source-ref" data-filename="医院感染诊断标准.pdf" data-ref="1">1</sup> | 医院感染诊断标准.pdf | 根据标准，医院感染诊断的具体要求是... |
| <sup class="source-ref" data-filename="医疗质量管理文件.pdf" data-ref="2">2</sup> | 医疗质量管理文件.pdf | 质量管理指标包括... |

【重要提醒】
- 参考文档中"文件名："后面显示的完整文件名就是表格"来源文件"列应该使用的名称
- 例如：如果参考文档显示"文件名：0-血液净化标准操作规程（2021版）.pdf"，则表格中必须使用"0-血液净化标准操作规程（2021版）.pdf"
- 不得省略前缀、版本号、括号等任何信息
- 序号列中的 data-filename 必须与来源文件列的文件名完全一致！"""

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
                    # 创建表格行计数器
                    table_row_counter = {'count': 0, 'in_table': False}
                    return self._stream_generate(
                        self.model, inputs, base_tokenizer, input_length, max_length,
                        do_sample=use_sample, temperature=temp, table_row_counter=table_row_counter
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
                # 创建表格行计数器
                table_row_counter = {'count': 0, 'in_table': False}
                return self._stream_generate(
                    self.model, inputs, self.tokenizer, input_length, stream_max_length,
                    do_sample=use_sample, temperature=temp, table_row_counter=table_row_counter
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
                    # 创建表格行计数器
                    table_row_counter = {'count': 0, 'in_table': False}
                    return self._stream_generate(
                        self.model, inputs, self.tokenizer, input_length, max_length,
                        do_sample=use_sample, temperature=temp, table_row_counter=table_row_counter
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

    def _stream_generate(self, model, inputs, tokenizer, input_length, max_length, do_sample=False, temperature=1.0, table_row_counter=None):
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
            table_row_counter: 表格行计数器字典 {'count': 0, 'in_table': False}

        Yields:
            (text_chunk: str, is_finished: bool)
        """
        import torch
        from transformers import TextIteratorStreamer
        import threading
        import time
        import queue
        import re

        # 初始化表格行计数器（如果未提供）
        if table_row_counter is None:
            table_row_counter = {'count': 0, 'in_table': False}

        # 检测是否使用 GPU
        model_device = next(model.parameters()).device
        is_gpu = 'cuda' in str(model_device)
        print(f"[GPU INFO] 流式生成设备: {model_device} ({'GPU加速 ✓' if is_gpu else 'CPU模式 ⚠'})", file=sys.stderr)

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
        total_chars = 0
        timeout = 300  # 5分钟超时（增加）
        start_time = time.time()
        last_log_time = start_time

        try:
            # 直接访问 text_queue
            text_queue = streamer.text_queue

            # 根据设备调整轮询间隔
            poll_interval = 0.001 if is_gpu else 0.05

            print(f"[DEBUG] 流式生成开始 - 设备: {'GPU' if is_gpu else 'CPU'}, 超时: {timeout}s")

            # 使用非阻塞方式获取 tokens - 立即返回每个token
            while generation_thread.is_alive():
                try:
                    # 非阻塞获取
                    text_chunk = text_queue.get_nowait()
                    if text_chunk is None:  # stop_signal
                        break
                    # 立即返回，不累积
                    full_text += text_chunk
                    total_chars += len(text_chunk)

                    # ========== 表格序号实时修正 ==========
                    # 检测表格开始
                    if not table_row_counter['in_table'] and ('| 序号 |' in text_chunk or '|序号|' in text_chunk):
                        table_row_counter['in_table'] = True
                        table_row_counter['count'] = 0
                        print(f"[TABLE SEQ] ========== 检测到表格开始 ==========", file=sys.stderr)

                    # 如果在表格中，修正序号
                    if table_row_counter['in_table']:
                        # 方案1：匹配完整的 sup 标签（最理想情况）
                        def replace_sequence_number(match):
                            table_row_counter['count'] += 1
                            old_number = match.group(3)  # 标签内的数字
                            filename = match.group(1)  # 文件名
                            new_number = table_row_counter['count']
                            print(f"[TABLE SEQ] ✓ 序号修正: {old_number} → {new_number}, 文件: {filename[:30]}...", file=sys.stderr)
                            return f'<sup class="source-ref" data-filename="{filename}" data-ref="{new_number}">{new_number}</sup>'

                        # 正则匹配表格数据行（完整sup标签）
                        pattern = r'\| <sup[^>]*data-filename="([^"]*)"[^>]*data-ref="(\d+)"[^>]*>(\d+)</sup>'
                        corrected_chunk = re.sub(pattern, replace_sequence_number, text_chunk)

                        # 方案2：如果没有匹配到完整格式，尝试匹配不完整或被分割的sup标签
                        if corrected_chunk == text_chunk:
                            # 匹配：<sup ...>数字</sup> 格式（可能缺少data-filename）
                            def replace_incomplete_sup(match):
                                table_row_counter['count'] += 1
                                old_number = match.group(2)
                                new_number = table_row_counter['count']
                                print(f"[TABLE SEQ] ✓ 不完整sup修正: {old_number} → {new_number}", file=sys.stderr)
                                # 从同一行提取文件名（如果有）
                                line_start = text_chunk.rfind('\n', 0, match.start())
                                if line_start == -1:
                                    line_start = 0
                                line_end = text_chunk.find('\n', match.end())
                                if line_end == -1:
                                    line_end = len(text_chunk)
                                line = text_chunk[line_start:line_end]
                                # 尝试从该行提取文件名（第二列）
                                parts = line.split('|')
                                if len(parts) >= 3:
                                    potential_filename = parts[2].strip() if len(parts) > 2 else ''
                                    if potential_filename and potential_filename not in ['序号', '来源文件', '内容', '---']:
                                        return f'| <sup class="source-ref" data-filename="{potential_filename}" data-ref="{new_number}">{new_number}</sup> |'
                                return f'| <sup class="source-ref" data-ref="{new_number}">{new_number}</sup> |'

                            # 匹配 <sup ...>数字</sup>
                            corrected_chunk = re.sub(r'\| <sup[^>]*>(\d+)</sup>', replace_incomplete_sup, text_chunk)

                        # 方案3：如果还是没有匹配，尝试纯数字格式
                        if corrected_chunk == text_chunk:
                            # | 2 | 文件名 | ... |
                            # 使用函数而不是lambda，确保count正确递增
                            def replace_simple_number_with_count(match):
                                table_row_counter['count'] += 1
                                old_number = match.group(2)  # 第一列的数字
                                new_number = table_row_counter['count']
                                print(f"[TABLE SEQ] ✓ 纯数字序号修正(首列): {old_number} → {new_number}", file=sys.stderr)
                                return f'{match.group(1)}<sup class="source-ref" data-ref="{new_number}">{new_number}</sup>{match.group(3)}'

                            # 只匹配第一列的数字（避免误匹配其他列）
                            corrected_chunk = re.sub(r'^(\|\s*)(\d+)(\s*\|)', replace_simple_number_with_count, text_chunk, count=1)

                        # 如果序号被修正，使用修正后的内容
                        if corrected_chunk != text_chunk:
                            text_chunk = corrected_chunk

                    # 每秒输出一次速度统计
                    current_time = time.time()
                    if current_time - last_log_time >= 1.0:
                        elapsed = current_time - start_time
                        speed_chars = total_chars / elapsed if elapsed > 0 else 0
                        # 估算token数（中文约2字符/token，英文约4字符/token）
                        estimated_tokens = total_chars / 2.5
                        speed_tokens = estimated_tokens / elapsed if elapsed > 0 else 0
                        print(f"[GPU SPEED] 已生成 {total_chars} 字符 (约{estimated_tokens:.0f} tokens), "
                              f"速度: {speed_chars:.1f} 字符/秒 ({speed_tokens:.1f} tokens/秒)", file=sys.stderr)
                        last_log_time = current_time

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
                    total_chars += len(text_chunk)

                    # ========== 表格序号实时修正（剩余数据，逻辑同上）==========
                    if table_row_counter['in_table']:
                        # 方案1：匹配完整的 sup 标签
                        def replace_sequence_number(match):
                            table_row_counter['count'] += 1
                            old_number = match.group(3)
                            filename = match.group(1)
                            new_number = table_row_counter['count']
                            print(f"[TABLE SEQ] ✓ 序号修正(剩余): {old_number} → {new_number}, 文件: {filename[:30]}...", file=sys.stderr)
                            return f'<sup class="source-ref" data-filename="{filename}" data-ref="{new_number}">{new_number}</sup>'

                        pattern = r'\| <sup[^>]*data-filename="([^"]*)"[^>]*data-ref="(\d+)"[^>]*>(\d+)</sup>'
                        corrected_chunk = re.sub(pattern, replace_sequence_number, text_chunk)

                        # 方案2：匹配不完整的sup标签
                        if corrected_chunk == text_chunk:
                            def replace_incomplete_sup(match):
                                table_row_counter['count'] += 1
                                old_number = match.group(1)
                                new_number = table_row_counter['count']
                                print(f"[TABLE SEQ] ✓ 不完整sup修正(剩余): {old_number} → {new_number}", file=sys.stderr)
                                return f'| <sup class="source-ref" data-ref="{new_number}">{new_number}</sup> |'

                            corrected_chunk = re.sub(r'\| <sup[^>]*>(\d+)</sup>', replace_incomplete_sup, text_chunk)

                        # 方案3：纯数字格式
                        if corrected_chunk == text_chunk:
                            # 使用函数而不是lambda，确保count正确递增
                            def replace_simple_number_with_count(match):
                                table_row_counter['count'] += 1
                                old_number = match.group(2)  # 第一列的数字
                                new_number = table_row_counter['count']
                                print(f"[TABLE SEQ] ✓ 纯数字序号修正(剩余): {old_number} → {new_number}", file=sys.stderr)
                                return f'{match.group(1)}<sup class="source-ref" data-ref="{new_number}">{new_number}</sup>{match.group(3)}'

                            corrected_chunk = re.sub(r'^(\|\s*)(\d+)(\s*\|)', replace_simple_number_with_count, text_chunk, count=1)

                        if corrected_chunk != text_chunk:
                            text_chunk = corrected_chunk

                    yield text_chunk, False
            except queue.Empty:
                pass

        except Exception as e:
            print(f"流式生成错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            generation_thread.join(timeout=5)
            total_time = time.time() - start_time
            if total_time > 0:
                speed_chars = total_chars / total_time
                estimated_tokens = total_chars / 2.5
                speed_tokens = estimated_tokens / total_time
                print(f"[GPU INFO] 流式生成完成!", file=sys.stderr)
                print(f"[GPU INFO] 总耗时: {total_time:.2f}秒", file=sys.stderr)
                print(f"[GPU INFO] 生成内容: {total_chars} 字符 (约{estimated_tokens:.0f} tokens)", file=sys.stderr)
                print(f"[GPU INFO] 平均速度: {speed_chars:.1f} 字符/秒 ({speed_tokens:.1f} tokens/秒)", file=sys.stderr)
                if is_gpu:
                    if speed_tokens < 20:
                        print(f"[GPU WARNING] ⚠ GPU速度较慢 (<20 tokens/秒)，可能需要检查配置", file=sys.stderr)
                    else:
                        print(f"[GPU SUCCESS] ✓ GPU加速正常 (>{speed_tokens:.0f} tokens/秒)", file=sys.stderr)
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

    def ask(self, question: str, top_k: int = 12, method: str = "hybrid", temperature: float = None, do_sample: bool = None, min_score: float = 0.15) -> Dict:
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

            # 标记是否检索到相关信息
            has_context = bool(sentence_results or chunk_results)

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

            # 兜底逻辑：如果过滤后 seen_docs 为空，但检索到了结果
            # 说明所有结果都被 min_score 过滤了，这时至少保留分数最高的 3 个结果
            if not seen_docs and (sentence_results or chunk_results):
                # 合并所有结果并按分数排序
                all_results = list(sentence_results) + list(chunk_results)
                all_results.sort(key=lambda x: x['score'], reverse=True)

                # 保留分数最高的 3 个结果
                for r in all_results[:3]:
                    text = r['text'].strip()
                    if text:
                        key = (r['filename'], r['filepath'])
                        if key not in seen_docs:
                            seen_docs[key] = {
                                'filename': r['filename'],
                                'filepath': r['filepath'],
                                'content': text,
                                'similarity': r['score']
                            }

                if seen_docs:
                    print(f"[兜底] 所有结果被 min_score 过滤，保留 top {len(seen_docs)} 结果")

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

            # 3. 构建提示词和上下文
            # 只有当向量库完全没有检索到结果时，才使用通用知识
            # 如果检索到了结果（即使分数较低），也应该使用这些文档
            if not has_context:
                # 向量库完全没有检索到相关信息，使用通用知识回答（富文本段落风格，严禁使用表格）
                context = f"""你是一个友好的 AI 助手。由于文档库中没有找到与该问题直接相关的信息，请根据你的通用知识回答。

【问题】
{question}

【重要指示 - 必须严格遵守】
1. 回答格式：使用富文本段落格式，禁止使用任何表格格式
2. 严禁使用 Markdown 表格语法（如 |、--- 等符号）
3. 使用分段落的方式组织内容，每段聚焦一个要点
4. 适当使用 emoji 让回答更生动（如 📌、💡、⚠️、✅ 等）
5. 如果涉及医疗健康问题，开头必须说明："⚠️ 以下是基于通用知识的回答，具体请咨询专业医生。"
6. 绝对不要编造或虚构任何来源文件名称

【回答格式示例】
⚠️ 以下是基于通用知识的回答，具体请咨询专业医生。

关于您的问题，我来为您解答：

📌 【要点一】
详细说明...

💡 【要点二】
详细说明...

✅ 总结
简要总结...

请按照上述格式直接给出回答：
"""
            else:
                # 有检索到相关信息，使用文档回答
                # 3. 检测是否需要表格格式
                use_table_format = self._should_use_table_format(question)
                if use_table_format:
                    # 构建文件名白名单
                    available_filenames = '\n'.join([f"  - {source['filename']}" for source in sources])
                    example_filename = sources[0]['filename'] if sources else "示例文件.pdf"

                    # 在 context 前添加增强的表格格式说明
                    table_instruction = f'''
【重要】请使用 Markdown 表格格式回答，使信息更加清晰易读。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【第一步：深入分析用户问题】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

用户问题：{question}

请在回答前先分析：
1. 问题类型是什么？（分类列举？对比差异？方法流程？标准要求？）
2. 用户最想获取什么信息？（具体名称？数值标准？操作步骤？区别要点？）
3. 关键信息要素有哪些？（时间？对象？条件？范围？）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【问题类型与回答策略】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 类型1：列举类（"有哪些"、"包括什么"）→ 只列出项目名称
📊 类型2：分类类（"如何分类"、"分几类"）→ 说明分类标准和类别
⚖️ 类型3：对比类（"区别"、"差异"）→ 重点说明差异点
📝 类型4：标准要求类（"具体要求"、"标准"）→ 列出具体数值、方法

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【可用文件名列表】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{available_filenames}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【表格格式要求】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

第1列："序号" - 填写数字 1、2、3...
第2列："来源文件" - 完整复制上面的文件名
第3列："内容" - 直接回答问题的核心信息

【核心要求】
- 精准回应用户疑问，每一行都直接回答问题核心
- 删除冗余信息，不要包含无关的背景描述
- 提取关键信息，优先回答用户最想知道的内容
- 简洁明了，用最少文字传达最准确信息

表格示例：
| 序号 | 来源文件 | 内容 |
| --- | --- | --- |
| 1 | {example_filename} | 鼠疫 - 甲类传染病<br>霍乱 - 甲类传染病 |
'''
                    context = table_instruction + context
                    print(f"检测到表格关键词，将使用表格格式回答")
                else:
                    # 非表格格式，添加明确的禁止表格指示
                    context = f"""【重要指示】
- 请使用富文本段落格式回答，严禁使用任何表格格式
- 禁止使用 Markdown 表格语法（如 |、--- 等符号）
- 使用分段落、分要点的方式组织内容，适当使用 emoji（如 📌、💡、⚠️、✅ 等）
- 【严禁】在回答内容中生成任何来源文件列表、参考文件列表或参考文献
- 【禁止】在回答末尾添加"来源文件："、"来源："、"参考文件："、"参考资料："等字样及后续列表
- 【禁止】列举任何文件名称或文档来源，系统会自动在回答最后添加编号的来源文件列表
- 只专注于回答问题本身，不要提及任何文件来源

用户问题：{question}

{context}"""

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

            # 7. 验证并修正表格（如果有表格）
            # 检测是否使用了表格格式（更精确的检测）
            has_table = False
            if '|' in answer:
                # 检查是否有表格分隔线
                if '| --- |' in answer or '|---|' in answer:
                    # 检查是否有表头（来源文件列）
                    if '| 来源文件 |' in answer or '|来源文件|' in answer:
                        has_table = True

            # 如果是表格格式，验证并修正表格中的文件名和序号
            if sources and has_table:
                import time
                validate_start = time.time()
                answer = self._validate_table_filenames_only(answer, sources)
                validate_time = (time.time() - validate_start) * 1000
                print(f"[PERF] 表格验证和序号修正耗时: {validate_time:.1f}ms", file=sys.stderr)

            # 8. 添加来源列表（纯文本格式时）
            # 如果不是表格格式且有来源文件，在答案最后添加来源列表
            if sources and not has_table:
                # 构建来源列表，添加 <sup> 标签
                sources_list = "\n\n---\n\n**来源文件：**\n\n"
                for idx, source in enumerate(sources, 1):
                    filename = source['filename']
                    # 添加带属性的 sup 标签
                    sup_tag = f'<sup class="source-ref" data-filename="{filename}" data-ref="{idx}">{idx}</sup>'
                    # 使用 <br> 确保HTML渲染时换行
                    sources_list += f"{sup_tag}. {filename}<br>\n"
                answer += sources_list

            # 8. 返回结果
            # 注意：当没有来源时，不返回 sources 字段，避免前端显示空的来源区域
            if sources:
                return {
                    'question': question,
                    'answer': answer,
                    'sources': sources,
                    'success': True
                }
            else:
                return {
                    'question': question,
                    'answer': answer,
                    'success': True
                }

        except Exception as e:
            return {
                'question': question,
                'answer': f"抱歉，生成答案时出错: {str(e)}",
                'sources': [],
                'success': False
            }

    def ask_stream(self, question: str, top_k: int = 6, method: str = "hybrid", temperature: float = None, do_sample: bool = None, min_score: float = 0.15):
        """
        流式提问并生成答案（性能优化版）

        Args:
            question: 用户问题
            top_k: 检索的文档数量（默认6，优化性能）
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
        import time
        import sys
        from concurrent.futures import ThreadPoolExecutor

        # 重置引用计数器和sources列表
        self._citation_counter = 0
        self._citation_sources = []

        # 用于处理跨块的引用标记
        _buffer = ""
        _has_sources = False
        _sources_list = []

        try:
            # 1. 立即返回状态，让用户知道正在处理
            yield {'type': 'status', 'data': 'AI正在思考...'}

            # 2. 并行检索相关文档（优化性能）
            search_start = time.time()

            # 使用线程池并行执行sentence和chunk搜索
            with ThreadPoolExecutor(max_workers=2) as executor:
                sentence_future = executor.submit(
                    self.search_model.search,
                    question,
                    method=method,
                    top_k=top_k,
                    level="sentence"
                )
                chunk_future = executor.submit(
                    self.search_model.search,
                    question,
                    method=method,
                    top_k=max(2, top_k // 2),
                    level="chunk"
                )

                # 获取结果
                sentence_results = sentence_future.result()
                chunk_results = chunk_future.result()

            search_time = (time.time() - search_start) * 1000
            print(f"[PERF] 并行检索耗时: {search_time:.1f}ms, 结果数: {len(sentence_results)} + {len(chunk_results)}", file=sys.stderr)

            # 标记是否检索到相关信息
            has_context = bool(sentence_results or chunk_results)

            # 3. 快速合并相同文档（优化性能）
            merge_start = time.time()
            seen_docs = {}  # {(filename, filepath): merged_source}

            # 快速处理句子级别结果
            for r in sentence_results:
                if r['score'] < min_score:
                    continue
                text = r['text'].strip()
                if not text:
                    continue
                key = (r['filename'], r['filepath'])
                if key in seen_docs:
                    # 简单合并：只保留最高相似度的内容
                    if r['score'] > seen_docs[key]['similarity']:
                        seen_docs[key]['content'] = text
                        seen_docs[key]['similarity'] = r['score']
                else:
                    seen_docs[key] = {
                        'filename': r['filename'],
                        'filepath': r['filepath'],
                        'content': text,
                        'similarity': r['score']
                    }

            # 快速处理文本块级别结果
            for r in chunk_results:
                if r['score'] < min_score:
                    continue
                text = r['text'].strip()
                if not text:
                    continue
                key = (r['filename'], r['filepath'])
                if key in seen_docs:
                    if r['score'] > seen_docs[key]['similarity']:
                        seen_docs[key]['content'] = text
                        seen_docs[key]['similarity'] = r['score']
                else:
                    seen_docs[key] = {
                        'filename': r['filename'],
                        'filepath': r['filepath'],
                        'content': text,
                        'similarity': r['score']
                    }

            # 兜底逻辑：如果没有结果，取top 3
            if not seen_docs and (sentence_results or chunk_results):
                all_results = list(sentence_results) + list(chunk_results)
                all_results.sort(key=lambda x: x['score'], reverse=True)
                for r in all_results[:3]:
                    text = r['text'].strip()
                    if text:
                        key = (r['filename'], r['filepath'])
                        if key not in seen_docs:
                            seen_docs[key] = {
                                'filename': r['filename'],
                                'filepath': r['filepath'],
                                'content': text,
                                'similarity': r['score']
                            }

            merge_time = (time.time() - merge_start) * 1000
            print(f"[PERF] 文档合并耗时: {merge_time:.1f}ms, 合并后文档数: {len(seen_docs)}", file=sys.stderr)

            # 快速构建context
            sources = []
            merged_context_parts = []
            for idx, (key, source) in enumerate(seen_docs.items(), 1):
                source['index'] = idx
                source_with_number = source.copy()
                source_with_number['content'] = f"{source['filename']} {idx}：{source['content']}"
                sources.append(source_with_number)
                # 简化格式，减少context长度
                merged_context_parts.append(
                    f"文档{idx}|{source['filename']}|{source['content']}"
                )

            context = "\n".join(merged_context_parts)

            # 4. 构建提示词和上下文
            # 只有当向量库完全没有检索到结果时，才使用通用知识
            # 如果检索到了结果（即使分数较低），也应该使用这些文档
            if not has_context:
                # 向量库完全没有检索到相关信息，使用通用知识回答（富文本段落风格，严禁使用表格）
                context = f"""你是一个友好的 AI 助手。由于文档库中没有找到与该问题直接相关的信息，请根据你的通用知识回答。

【问题】
{question}

【重要指示 - 必须严格遵守】
1. 回答格式：使用富文本段落格式，禁止使用任何表格格式
2. 严禁使用 Markdown 表格语法（如 |、--- 等符号）
3. 使用分段落的方式组织内容，每段聚焦一个要点
4. 适当使用 emoji 让回答更生动（如 📌、💡、⚠️、✅ 等）
5. 如果涉及医疗健康问题，开头必须说明："⚠️ 以下是基于通用知识的回答，具体请咨询专业医生。"
6. 绝对不要编造或虚构任何来源文件名称

【回答格式示例】
⚠️ 以下是基于通用知识的回答，具体请咨询专业医生。

关于您的问题，我来为您解答：

📌 【要点一】
详细说明...

💡 【要点二】
详细说明...

✅ 总结
简要总结...

请按照上述格式直接给出回答：
"""
            else:
                # 有检索到相关信息，使用文档回答
                # 4. 检测是否需要表格格式
                use_table_format = self._should_use_table_format(question)
                if use_table_format and sources:
                    # 构建可用文件名列表
                    available_filenames = '\n'.join([f"  - {source['filename']}" for source in sources])
                    example_filename = sources[0]['filename']

                    # 在 context 前添加表格格式说明和问题重述
                    table_instruction = f'''
【重要】请使用 Markdown 表格格式回答，使信息更加清晰易读。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【第一步：深入分析用户问题】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

用户问题：{question}

请在回答前先分析：
1. 问题类型是什么？（分类列举？对比差异？方法流程？标准要求？）
2. 用户最想获取什么信息？（具体名称？数值标准？操作步骤？区别要点？）
3. 关键信息要素有哪些？（时间？对象？条件？范围？）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【问题类型与回答策略】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

根据问题类型，采用相应的策略：

📋 类型1：列举类（"有哪些"、"包括什么"、"包含哪些"）
→ 内容列只需列出项目名称，可附加简短特征
→ 示例："鼠疫 - 甲类传染病"

📊 类型2：分类类（"如何分类"、"分几类"、"分类标准"）
→ 内容列说明分类标准和各类别名称
→ 示例："按甲类管理的乙类传染病：传染性非典型肺炎、炭疽"

⚖️ 类型3：对比类（"区别"、"差异"、"对比"）
→ 内容列重点说明差异点，相同点可省略
→ 示例："甲类：强制隔离；乙类：定点隔离"

📝 类型4：标准要求类（"具体要求"、"标准是什么"、"如何"）
→ 内容列列出具体的数值、方法、步骤
→ 示例："浓度500mg/L，作用时间30分钟"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【可用文件名列表】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

以下是你唯一可以使用的文件名白名单，"来源文件"列必须从以下列表中选择：
{available_filenames}

【最关键】"来源文件"列必须从上面的列表中完全复制，不得修改、简化或创造文件名！

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【表格格式严格要求】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

第1列："序号" - 填写数字 1、2、3...
第2列："来源文件" - 完整复制上面的文件名
第3列："内容" - 直接回答问题的核心信息

格式细节：
✓ 序号列：<sup class="source-ref" data-filename="对应文件名" data-ref="序号">序号</sup>
✓ 序号列的 data-filename 必须与来源文件列完全一致
✓ 内容列换行时使用 HTML <br> 标签
✓ 每个文件只能出现一行，多条信息用 <br> 连接

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【【【核心要求：内容必须针对问题】】】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 精准回应用户疑问：每一行内容都必须直接回答问题的核心
2. 删除冗余信息：不要包含与问题无关的背景描述
3. 提取关键信息：优先回答用户最想知道的内容
4. 简洁明了：用最少的文字传达最准确的信息

❌ 错误示例（问题：甲类传染病有哪些）
| 序号 | 来源文件 | 内容 |
| 1 | xxx.pdf | 传染病防治法规定，传染病分为甲类、乙类、丙类... |
（问题：用户问甲类有哪些，回答却包含分类介绍，信息冗余）

✅ 正确示例（问题：甲类传染病有哪些）
| 序号 | 来源文件 | 内容 |
| 1 | xxx.pdf | 鼠疫 - 甲类传染病，需要强制隔离治疗 |
（直接回答：列出甲类传染病名称）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【回答示例】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

| 序号 | 来源文件 | 内容 |
| --- | --- | --- |
| <sup class="source-ref" data-filename="{example_filename}" data-ref="1">1</sup> | {example_filename} | 鼠疫 - 甲类传染病<br>霍乱 - 甲类传染病 |
'''
                    context = table_instruction + context
                    print(f"检测到表格关键词，将使用表格格式回答")
                    print(f"[DEBUG] 可用文件名: {[s['filename'] for s in sources]}")
                else:
                    # 非表格格式，添加明确的禁止表格指示
                    context = f"""【重要指示】
- 请使用富文本段落格式回答，严禁使用任何表格格式
- 禁止使用 Markdown 表格语法（如 |、--- 等符号）
- 使用分段落、分要点的方式组织内容，适当使用 emoji（如 📌、💡、⚠️、✅ 等）
- 【严禁】在回答内容中生成任何来源文件列表、参考文件列表或参考文献
- 【禁止】在回答末尾添加"来源文件："、"来源："、"参考文件："、"参考资料："等字样及后续列表
- 【禁止】列举任何文件名称或文档来源，系统会自动在回答最后添加编号的来源文件列表
- 只专注于回答问题本身，不要提及任何文件来源

用户问题：{question}

{context}"""

            # 5. 返回来源信息（只在有来源时返回）
            if has_context and sources:
                yield {'type': 'source', 'data': sources}
                _sources_list = sources
                _has_sources = True
            else:
                # 没有来源时不返回 source 事件
                _sources_list = []
                _has_sources = False

            # 6. 流式生成回答
            yield {'type': 'status', 'data': '正在生成回答...'}

            # 先检测是否会使用表格格式
            use_table_format = self._should_use_table_format(question)

            # 导入 sys 用于调试输出
            import sys

            full_answer = ""
            full_answer_processed = ""  # 保存处理后的完整答案

            import time
            generate_start = time.time()

            for text_chunk, is_finished in self.llm.generate(question, context, stream=True, temperature=temperature, do_sample=do_sample):
                if text_chunk and not is_finished:
                    # 使用缓冲区处理跨块的引用
                    _buffer += text_chunk
                    processed = self._process_buffer(_buffer, _sources_list if _has_sources else None, False)

                    if processed['output']:
                        # 直接发送（序号已在 _stream_generate 中修正）
                        yield {'type': 'content', 'data': processed['output']}
                        full_answer_processed += processed['output']

                    _buffer = processed['remaining']
                    full_answer += text_chunk

                if is_finished:
                    # 处理剩余的缓冲区内容
                    if _buffer:
                        processed = self._process_buffer(_buffer, _sources_list if _has_sources else None, False)
                        if processed['output']:
                            yield {'type': 'content', 'data': processed['output']}
                            full_answer_processed += processed['output']
                        _buffer = ""
                    if text_chunk and not full_answer:
                        full_answer = text_chunk
                    break

            # 7. 流式输出结束
            generate_time = (time.time() - generate_start) * 1000
            print(f"[PERF] LLM生成耗时: {generate_time:.1f}ms, 生成字符数: {len(full_answer_processed)}", file=sys.stderr)
            print(f"[DEBUG] ========== 流式结束 ==========", file=sys.stderr)
            print(f"[DEBUG] full_answer原始长度: {len(full_answer)}", file=sys.stderr)
            print(f"[DEBUG] full_answer_processed长度: {len(full_answer_processed)}", file=sys.stderr)
            print(f"[DEBUG] full_answer_processed前500字符:\n{full_answer_processed[:500]}", file=sys.stderr)

            # 8. 验证并修正表格中的文件名（只在有来源时处理）
            final_answer = full_answer_processed if full_answer_processed else full_answer

            # 检测是否使用了表格格式
            # 更精确的表格检测：需要同时满足多个条件
            has_table = False
            if '|' in final_answer:
                # 检查是否有表格分隔线
                if '| --- |' in final_answer or '|---|' in final_answer:
                    # 检查是否有表头（来源文件列）
                    if '| 来源文件 |' in final_answer or '|来源文件|' in final_answer:
                        has_table = True

            import sys
            print(f"[DEBUG SOURCES] _has_sources={_has_sources}, sources数量={len(sources) if sources else 0}, has_table={has_table}", file=sys.stderr)
            print(f"[DEBUG SOURCES] 最终答案包含 | : {'|' in final_answer}", file=sys.stderr)
            print(f"[DEBUG SOURCES] 最终答案包含 '来源文件': {'来源文件' in final_answer}", file=sys.stderr)

            # 8.5 清理LLM生成的不需要的内容（段落格式和表格格式都需要清理）
            if _has_sources and sources:
                import time
                clean_start = time.time()
                final_answer = self._clean_llm_generated_content(final_answer)
                clean_time = (time.time() - clean_start) * 1000
                print(f"[PERF] 清理LLM生成内容耗时: {clean_time:.1f}ms", file=sys.stderr)

            # 表格格式：验证文件名（不合并，因为已在检索阶段合并）
            if _has_sources and sources and has_table:
                import time
                validate_start = time.time()
                final_answer = self._validate_table_filenames_only(final_answer, sources)
                validate_time = (time.time() - validate_start) * 1000
                print(f"[PERF] 文件名验证耗时: {validate_time:.1f}ms", file=sys.stderr)


            # 9. 添加来源列表（纯文本格式时）
            # 如果不是表格格式且有来源文件，在答案最后添加来源列表
            sources_list = ""
            if _has_sources and sources and not has_table:
                # 先按文件名去重，避免来源列表中出现重复文件
                seen_filenames = {}
                unique_sources = []
                for source in sources:
                    filename = source['filename']
                    if filename not in seen_filenames:
                        seen_filenames[filename] = True
                        unique_sources.append(source)

                # 构建来源列表，添加 <sup> 标签
                sources_list = "\n\n---\n\n**来源文件：**\n\n"
                for idx, source in enumerate(unique_sources, 1):
                    filename = source['filename']
                    # 添加带属性的 sup 标签
                    sup_tag = f'<sup class="source-ref" data-filename="{filename}" data-ref="{idx}">{idx}</sup>'
                    # 使用 <br> 确保HTML渲染时换行
                    sources_list += f"{sup_tag}. {filename}<br>\n"
                final_answer += sources_list
                print(f"[DEBUG SOURCES] 已添加来源列表，原始数量：{len(sources)}，去重后：{len(unique_sources)}", file=sys.stderr)
                print(f"[DEBUG SOURCES] 最终答案长度：{len(final_answer)}", file=sys.stderr)

                # 额外发送来源列表作为 content 事件（确保前端能显示）
                yield {'type': 'content', 'data': sources_list}

            # 10. 智能格式化输出（新增）
            # 使用智能分类器识别问题类型并生成结构化数据
            if _has_sources and sources:
                try:
                    from src.smart_answer_formatter import SmartAnswerFormatter

                    # 构造 result 字典
                    result = {
                        'answer': final_answer,
                        'sources': sources,
                        'question': question
                    }

                    # 智能格式化
                    formatter = SmartAnswerFormatter()
                    formatted_result = formatter.format(question, result)

                    # 发送结构化数据事件
                    yield {
                        'type': 'structured_data',
                        'data': {
                            'question_type': formatted_result['question_type'],
                            'format_type': formatted_result['format_type'],
                            'confidence': formatted_result['confidence'],
                            'content': formatted_result['content'],
                            'sources': formatted_result['sources']
                        }
                    }
                    print(f"[DEBUG] 已发送智能格式化数据，类型: {formatted_result['question_type']}, 格式: {formatted_result['format_type']}", file=sys.stderr)
                except Exception as e:
                    # 智能格式化失败不影响正常流程
                    print(f"[WARNING] 智能格式化失败: {e}", file=sys.stderr)
                    import traceback
                    traceback.print_exc()

            # 11. 完成，返回处理后的最终答案
            # 注意：当没有来源时，不返回 sources 字段，避免前端显示空的来源区域

            if _has_sources:
                yield {'type': 'done', 'data': {'sources': sources, 'full_answer': final_answer}}
            else:
                yield {'type': 'done', 'data': {'full_answer': final_answer}}

            # 如果是表格格式，发送完整修正后的表格（替换前端显示）
            if has_table and final_answer:
                # 提取完整的表格部分（从表头到表格结束）
                import re
                # 匹配表格：从 | 序号 | 开始，到最后一个 |...| 行
                lines = final_answer.split('\n')
                table_lines = []
                in_table = False
                table_start_idx = -1

                for i, line in enumerate(lines):
                    # 使用原始行（不strip），保留前导空格
                    # 检测表格开始（包含序号或来源文件列）
                    if not in_table and ('| 序号 |' in line or '|序号|' in line):
                        in_table = True
                        table_start_idx = i
                        table_lines.append(line)
                        continue

                    # 在表格中
                    if in_table:
                        # 检测表格结束（使用更宽松的条件）
                        # 只有当整行为空，或者不包含 | 且不是以 <sup 开头时才结束
                        if not line or not line.strip():
                            # 空行，表格结束
                            break
                        # 检查是否仍为表格行（包含 | 或者是续行）
                        if '|' not in line and not line.strip().startswith('<'):
                            # 不包含 | 且不以 < 开头，可能是表格结束
                            # 但要检查是否为上一行的续行（比如长内容被换行）
                            if table_lines and '|' in table_lines[-1]:
                                # 上一行是表格行，当前行可能是内容续行
                                # 检查上一行是否以 | 结尾
                                if table_lines[-1].rstrip().endswith('|'):
                                    # 上一行以 | 结尾，说明当前行应该是新的一列
                                    break
                                else:
                                    # 上一行不以 | 结尾，当前行可能是内容的续行
                                    # 合并到上一行
                                    table_lines[-1] = table_lines[-1].rstrip() + ' ' + line.strip()
                                    continue
                            else:
                                break
                        table_lines.append(line)

                # 如果找到表格，提取并重新构建
                if table_lines and len(table_lines) >= 3:  # 至少有表头、分隔符、一行数据
                    # 验证表格完整性，检查是否有不完整的行
                    print(f"[DEBUG TABLE] 验证表格完整性，共 {len(table_lines)} 行", file=sys.stderr)
                    validated_lines = []
                    for idx, tline in enumerate(table_lines):
                        print(f"[DEBUG TABLE] 行{idx}: {tline[:100]}", file=sys.stderr)
                        validated_lines.append(tline)

                    corrected_table = '\n'.join(validated_lines)

                    # 发送替换事件，告诉前端用修正后的表格替换
                    yield {'type': 'table_replace', 'data': corrected_table}
                    print(f"[DEBUG] 已发送 table_replace 事件，包含修正后的表格（{len(validated_lines)}行）", file=sys.stderr)

        except Exception as e:
            import traceback
            import sys

            # 打印详细错误信息到控制台
            print(f"\n{'='*70}", file=sys.stderr)
            print(f"❌ ask_stream 发生错误", file=sys.stderr)
            print(f"{'='*70}", file=sys.stderr)
            print(f"错误类型: {type(e).__name__}", file=sys.stderr)
            print(f"错误消息: {str(e)}", file=sys.stderr)
            print(f"\n完整堆栈跟踪:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            print(f"{'='*70}\n", file=sys.stderr)
            sys.stderr.flush()

            yield {'type': 'error', 'data': str(e)}

    def _validate_and_fix_table_filenames(self, text: str, sources: list) -> str:
        """
        验证并修正表格中的文件名，确保所有文件名都来自 sources 列表

        使用更精确的表格解析方法，避免因内容中的 | 字符导致分割错误
        """
        import re
        import sys
        from difflib import SequenceMatcher

        # 构建正确的文件名列表和索引映射
        valid_filenames = {source['filename']: source for source in sources}
        filename_to_index = {source['filename']: idx + 1 for idx, source in enumerate(sources)}

        print(f"[DEBUG FILENAME] ========== 验证文件名 ==========", file=sys.stderr)
        print(f"[DEBUG FILENAME] 正确的文件名列表: {list(valid_filenames.keys())}", file=sys.stderr)

        lines = text.split('\n')
        result_lines = []
        in_table = False
        table_header_found = False

        for line in lines:
            stripped_line = line.strip()

            # 检测表格开始/结束
            if stripped_line.startswith('|') and stripped_line.endswith('|'):
                if not in_table:
                    in_table = True
                    table_header_found = False
            elif in_table and not stripped_line:
                in_table = False
                table_header_found = False

            # 处理表格行
            if in_table and stripped_line.startswith('|'):
                # 检查是否是表头行
                if '来源文件' in stripped_line or '---' in stripped_line:
                    table_header_found = True
                    result_lines.append(line)
                    continue

                # 只处理数据行（表头之后）
                if not table_header_found:
                    result_lines.append(line)
                    continue

                # 使用正则表达式精确分割表格行
                # 匹配 | 之间的内容，保留空单元格
                # 模式：| 内容 | 内容 | 内容 |
                parts = re.split(r'\|', stripped_line)
                # 去掉首尾空元素
                parts = [p.strip() for p in parts if p is not None or len(parts) == 1]

                # 至少需要：| 序号 | 来源文件 | 内容 |
                if len(parts) >= 3:
                    # 第二列（索引1）应该是"来源文件"
                    filename_cell = parts[1]

                    # 检查是否是有效的文件名列
                    is_header_row = (filename_cell in ['序号', '来源文件', '内容', '---', '',
                                   'No.', '编号', '文件', 'File', 'Source'])
                    is_sup_cell = filename_cell.startswith('<sup')
                    is_number_only = re.match(r'^[\d\s]+$', filename_cell)

                    if not is_header_row and not is_sup_cell and not is_number_only:
                        # 这是一个文件名列，需要验证
                        if filename_cell not in valid_filenames:
                            print(f"[DEBUG FILENAME] ✗ 文件名无效: '{filename_cell}'", file=sys.stderr)

                            # 尝试精确匹配
                            best_match = self._find_best_match_filename(filename_cell, valid_filenames.keys())

                            if best_match:
                                print(f"[DEBUG FILENAME] → 替换为: '{best_match}'", file=sys.stderr)
                                parts[1] = best_match
                                # 同时修正序号列中的 data-filename（如果存在）
                                if len(parts) > 0 and '<sup' in parts[0]:
                                    # 替换 data-filename 属性
                                    old_filename_pattern = re.escape(filename_cell)
                                    parts[0] = re.sub(
                                        rf'data-filename=["\']?{old_filename_pattern}["\']?',
                                        f'data-filename="{best_match}"',
                                        parts[0]
                                    )
                                # 重建行
                                line = '|' + '|'.join(parts) + '|'
                            else:
                                # 如果找不到匹配，尝试根据编号映射
                                # 尝试从序号列提取编号
                                if len(parts) > 0:
                                    # 从 <sup> 标签中提取编号
                                    match = re.search(r'data-ref="(\d+)"', parts[0])
                                    if match:
                                        ref_num = int(match.group(1))
                                        if ref_num <= len(sources):
                                            correct_filename = sources[ref_num - 1]['filename']
                                            print(f"[DEBUG FILENAME] → 根据编号映射到: '{correct_filename}'", file=sys.stderr)
                                            parts[1] = correct_filename
                                            # 同时修正 data-filename
                                            parts[0] = re.sub(
                                                rf'data-filename=["\']?[^"\']*["\']?',
                                                f'data-filename="{correct_filename}"',
                                                parts[0]
                                            )
                                            line = '|' + '|'.join(parts) + '|'
                        else:
                            print(f"[DEBUG FILENAME] ✓ 文件名有效: '{filename_cell}'", file=sys.stderr)

            result_lines.append(line)

        # 分组合并：按来源文件合并相同文件的多行内容
        # 只有当存在表格时才执行分组逻辑
        table_start_idx = None
        table_end_idx = None

        # 查找表格边界
        for idx, line in enumerate(result_lines):
            stripped = line.strip()
            if stripped.startswith('|') and stripped.endswith('|'):
                if table_start_idx is None:
                    table_start_idx = idx
                table_end_idx = idx
            elif table_start_idx is not None:
                break  # 表格结束

        if table_start_idx is not None and table_end_idx is not None:
            # 提取表格部分
            table_lines = result_lines[table_start_idx:table_end_idx + 1]

            # 解析表格：找出表头、分隔符、数据行
            header_line = None
            separator_line = None
            data_rows = []

            for i, line in enumerate(table_lines):
                stripped = line.strip()
                if '来源文件' in stripped:
                    header_line = line
                elif '---' in stripped:
                    separator_line = line
                elif stripped.startswith('|') and header_line is not None:
                    # 这是数据行
                    data_rows.append(line)

            if header_line and separator_line and data_rows:
                # 解析数据行，按来源文件分组
                from collections import defaultdict

                # 存储分组数据: {filename: {'序号': xxx, '内容': [content1, content2, ...]}}
                grouped_data = defaultdict(lambda: {'序号': None, '内容列表': []})

                for row in data_rows:
                    # 解析行: | 序号 | 来源文件 | 内容 |
                    parts = [p.strip() for p in row.split('|')]
                    parts = [p for p in parts if p or len(parts) <= 3]  # 保留空单元格但去掉首尾空

                    if len(parts) >= 3:
                        number_cell = parts[0]
                        filename_cell = parts[1]
                        content_cell = parts[2] if len(parts) > 2 else ''

                        # 跳过表头重复行
                        if filename_cell in ['序号', '来源文件', '内容', 'No.', '文件']:
                            continue

                        # 提取序号中的 data-ref 编号
                        import re
                        ref_match = re.search(r'data-ref="(\d+)"', number_cell)
                        ref_num = int(ref_match.group(1)) if ref_match else None

                        # 按文件名分组
                        grouped_data[filename_cell]['序号'] = number_cell
                        grouped_data[filename_cell]['ref_num'] = ref_num
                        grouped_data[filename_cell]['内容列表'].append(content_cell)

                # 重建表格：合并相同文件的内容
                new_table_lines = [header_line, separator_line]

                # 按 ref_num 排序
                sorted_items = sorted(grouped_data.items(), key=lambda x: x[1]['ref_num'] if x[1]['ref_num'] else 999)

                for filename, data in sorted_items:
                    # 合并内容：在 Markdown 表格中使用 <br> 标签换行
                    # 注意：表格中不能直接使用 \n 换行，必须使用 HTML <br> 标签
                    if len(data['内容列表']) > 1:
                        merged_content = '<br><br>'.join(data['内容列表'])
                    else:
                        merged_content = data['内容列表'][0]

                    # 构建新行
                    new_row = f'| {data["序号"]} | {filename} | {merged_content} |'
                    new_table_lines.append(new_row)

                    print(f"[DEBUG MERGE] 合并文件: {filename}, 内容块数: {len(data['内容列表'])}", file=sys.stderr)

                # 替换原表格
                result_lines = (
                    result_lines[:table_start_idx] +
                    new_table_lines +
                    result_lines[table_end_idx + 1:]
                )

                print(f"[DEBUG MERGE] 表格分组完成，原始行数: {len(data_rows)}, 合并后行数: {len(sorted_items)}", file=sys.stderr)

        result = '\n'.join(result_lines)

        # 二次验证：再次检查是否还有未匹配的文件名
        # 提取所有可能的文件名模式
        for filename in valid_filenames.keys():
            # 检查是否有类似的错误文件名
            escaped_filename = re.escape(filename)
            # 如果文件名不在结果中，尝试查找相似模式
            if filename not in result:
                # 查找可能的错误模式（如缺少前缀、后缀等）
                filename_base = filename
                # 移除常见前缀
                for prefix in ['0-', '1-', '2-', '3-', '4-', '5-', '6-', '7-', '8-', '9-']:
                    if filename.startswith(prefix):
                        filename_base = filename[len(prefix):]
                        break

                # 尝试查找没有前缀的版本
                if filename_base != filename and filename_base in result:
                    print(f"[DEBUG FILENAME] 修正: '{filename_base}' -> '{filename}'", file=sys.stderr)
                    result = result.replace(filename_base, filename)

        return result

    def _validate_table_filenames_only(self, text: str, sources: list) -> str:
        """
        处理表格：
        1. 合并多个表格为一个
        2. 验证和修正表格中的文件名
        3. 重新排序序号列（确保序号连续）

        Args:
            text: 完整文本内容
            sources: 来源列表

        Returns:
            验证后的文本
        """
        import re
        import sys
        from difflib import SequenceMatcher

        # 构建正确的文件名列表
        valid_filenames = {source['filename']: source for source in sources}

        print(f"[DEBUG VALIDATE] 开始处理表格，正确文件名: {list(valid_filenames.keys())}", file=sys.stderr)

        lines = text.split('\n')
        result_lines = []
        all_table_data_rows = []  # 收集所有表格的数据行（合并多个表格）
        in_table = False
        table_header_found = False

        for line in lines:
            stripped_line = line.strip()

            # 检测表格开始/结束
            if stripped_line.startswith('|') and stripped_line.endswith('|'):
                if not in_table:
                    in_table = True
                    table_header_found = False
                    print(f"[DEBUG VALIDATE] 检测到表格开始", file=sys.stderr)
            elif in_table and not stripped_line:
                in_table = False
                result_lines.append(line)
                print(f"[DEBUG VALIDATE] 表格结束", file=sys.stderr)
                continue

            # 处理表格行
            if in_table and stripped_line.startswith('|'):
                # 跳过表头和分隔符行（只收集数据行）
                if '来源文件' in stripped_line or '---' in stripped_line or '序号' in stripped_line:
                    if not table_header_found and ('来源文件' in stripped_line or '序号' in stripped_line):
                        table_header_found = True
                        print(f"[DEBUG VALIDATE] 检测到表头: {stripped_line[:60]}", file=sys.stderr)
                    continue

                # 尝试提取文件名（第二列）
                # 使用更简单的分割逻辑
                if '| 来源文件 |' in line or '|来源文件|' in line:
                    # 这是表头，跳过
                    continue

                # 提取数据行：| 序号 | 文件名 | 内容 |
                # 简单方法：找到第二个 | 和第三个 | 之间的内容
                pipe_count = stripped_line.count('|')
                if pipe_count >= 4:  # 至少有 | 序号 | 文件 | 内容 | (4个pipe)
                    # 分割并提取列
                    parts = [p.strip() for p in stripped_line.split('|')]
                    # 去掉首尾的空元素
                    parts = [p for p in parts if p]

                    if len(parts) >= 3:
                        # parts[0]=序号, parts[1]=文件名, parts[2]=内容
                        sequence_cell = parts[0] if len(parts) > 0 else ''
                        filename_cell = parts[1] if len(parts) > 1 else ''

                        print(f"[DEBUG VALIDATE] 提取列: 序号='{sequence_cell[:30]}', 文件='{filename_cell[:30]}'", file=sys.stderr)

                        # 检查是否是真正的数据行（文件名不在表头值中）
                        header_values = ['序号', '来源文件', '内容', 'No.', '编号', 'File', 'Source', '---']
                        is_data_row = filename_cell not in header_values and filename_cell

                        if is_data_row:
                            # 验证并修正文件名
                            if filename_cell not in valid_filenames and filename_cell:
                                best_match = self._find_best_match_filename(filename_cell, valid_filenames.keys())
                                if best_match:
                                    print(f"[DEBUG VALIDATE] 修正文件名: {filename_cell[:30]}... -> {best_match[:30]}...", file=sys.stderr)
                                    # 替换文件名
                                    line = line.replace(filename_cell, best_match)
                                    # 同时替换序号列中的 data-filename
                                    if f'data-filename="{filename_cell}"' in line:
                                        line = line.replace(f'data-filename="{filename_cell}"', f'data-filename="{best_match}"')
                                    elif f"data-filename='{filename_cell}'" in line:
                                        line = line.replace(f"data-filename='{filename_cell}'", f"data-filename='{best_match}'")

                            # 收集数据行
                            all_table_data_rows.append(line)
                            print(f"[DEBUG VALIDATE] ✓ 收集数据行 {len(all_table_data_rows)}: {stripped_line[:60]}", file=sys.stderr)
                            continue

                # 如果不是数据行，保留原行
                result_lines.append(line)
            else:
                result_lines.append(line)

        # 如果收集到了数据行，重新构建一个统一的表格
        if all_table_data_rows:
            print(f"[DEBUG VALIDATE] 共收集到 {len(all_table_data_rows)} 行数据，开始重新构建表格", file=sys.stderr)

            # 添加表头
            result_lines.append("| 序号 | 来源文件 | 内容 |")
            result_lines.append("| --- | --- | --- |")

            # 重新编号并添加数据行
            for idx, data_row in enumerate(all_table_data_rows, 1):
                # 首先提取该行对应的文件名（从第二列获取）
                filename_in_row = None
                parts = [p.strip() for p in data_row.split('|')]
                parts = [p for p in parts if p]  # 去掉空元素
                if len(parts) >= 2:
                    # 尝试从第二列提取文件名
                    potential_filename = parts[1]
                    # 检查是否是有效的文件名（不是表头）
                    header_values = ['序号', '来源文件', '内容', 'No.', '编号', 'File', 'Source']
                    if potential_filename not in header_values:
                        # 验证文件名
                        if potential_filename in valid_filenames:
                            filename_in_row = potential_filename
                        else:
                            # 尝试模糊匹配
                            best_match = self._find_best_match_filename(potential_filename, valid_filenames.keys())
                            if best_match:
                                filename_in_row = best_match
                                # 同时替换文件名
                                data_row = data_row.replace(potential_filename, best_match)

                # 替换序号列
                if filename_in_row:
                    # 使用正确的文件名生成序号标签
                    new_row = re.sub(
                        r'^\|\s*<sup[^>]*data-filename="[^"]*"[^>]*data-ref=["\']?\d+["\']?[^>]*>\s*\d+\s*</sup>\s*\|',
                        f'| <sup class="source-ref" data-filename="{filename_in_row}" data-ref="{idx}">{idx}</sup> |',
                        data_row,
                        count=1
                    )
                else:
                    # 没有找到文件名，直接替换第一个|...|中的序号
                    new_row = re.sub(
                        r'^\|\s*<sup[^>]*>\s*\d+\s*</sup>\s*\|',
                        f'| <sup class="source-ref" data-ref="{idx}">{idx}</sup> |',
                        data_row,
                        count=1
                    )

                # 如果没有 sup 标签，尝试替换纯数字序号（第一列）
                if new_row == data_row:
                    # 匹配 | 数字 | 格式（第一列）
                    new_row = re.sub(
                        r'^(\|\s*)(\d+)(\s*\|)',
                        lambda m: f'{m.group(1)}<sup class="source-ref" data-ref="{idx}">{idx}</sup>{m.group(3)}',
                        data_row,
                        count=1
                    )

                result_lines.append(new_row)
                print(f"[DEBUG VALIDATE] 重新编号第 {idx} 行，文件名: {filename_in_row if filename_in_row else '未找到'}", file=sys.stderr)

            print(f"[DEBUG VALIDATE] 表格重建完成，共 {len(all_table_data_rows)} 行", file=sys.stderr)
        else:
            print(f"[DEBUG VALIDATE] 未检测到数据行", file=sys.stderr)

        result = '\n'.join(result_lines)
        return result

    def _clean_llm_generated_content(self, text: str) -> str:
        """
        清理LLM生成的不需要的内容

        清理内容：
        1. [来源未列出，但为一般感冒处理原则] 类引用标记
        2. "来源文件列表" 标题及其后面的"文件名："列表
        3. 保留最后的"来源文件："或"来源："及编号列表

        Args:
            text: 完整文本内容

        Returns:
            清理后的文本
        """
        import re

        # 步骤1：移除所有"来源未列出"类的引用标记
        text = re.sub(r'\[来源未列出[^\]]*\]', '', text)
        text = re.sub(r'\[来源[未：:][^\]]*\]', '', text)

        # 步骤2：移除"来源文件列表"标题和其后的"文件名："列表
        # 匹配模式：来源文件列表\n文件名：xxx.pdf\n文件名：yyy.pdf\n\n来源文件：
        # 或者：来源文件列表\n文件名：xxx.pdf\n文件名：yyy.pdf\n来源文件：

        # 先定位"来源文件列表"的位置
        sources_list_pattern = r'来源文件列表\s*\n'
        match = re.search(sources_list_pattern, text)

        if match:
            # 找到"来源文件列表"的起始位置
            start_pos = match.start()

            # 从这个位置开始，查找后面的内容
            # 我们需要找到真正的"来源文件："或"来源："的位置
            remaining_text = text[start_pos:]

            # 查找真正的来源标题（带编号的）
            # 匹配：\n\n来源文件：\n\n1. xxx.pdf 或 \n\n来源文件：\n\n<sup...>1</sup> xxx.pdf
            real_sources_pattern = r'\n\n\*\*来源[文件]?[：:]\*\*\s*\n'
            real_match = re.search(real_sources_pattern, remaining_text)

            if real_match:
                # 找到了真正的来源列表，删除中间的部分
                real_start_pos = start_pos + real_match.start()
                # 保留从"来源文件列表"之前到真正的来源列表之前的内容
                # 删除从"来源文件列表"到真正来源列表之前的所有内容
                text = text[:start_pos] + remaining_text[real_match.start():]
                print(f"[DEBUG CLEAN] 已清理LLM生成的来源文件列表部分", file=sys.stderr)
            else:
                # 如果没找到真正的来源列表，删除"来源文件列表"及后面的所有内容
                # 但要保留用户可能需要的其他信息
                # 查找是否有连续的文件名列表（文件名：xxx.pdf\n文件名：yyy.pdf）
                filename_list_pattern = r'(文件名：[^\n]+\n)+'
                filename_match = re.search(filename_list_pattern, remaining_text)

                if filename_match:
                    # 只删除"来源文件列表"和文件名列表，保留后面的内容
                    after_filename_list = remaining_text[filename_match.end():]
                    # 检查后面是否有真正的来源列表
                    if re.search(r'\n\n\*\*来源', after_filename_list):
                        text = text[:start_pos] + after_filename_list
                        print(f"[DEBUG CLEAN] 已清理来源文件列表和文件名列表", file=sys.stderr)
                    else:
                        # 没有真正的来源列表，删除"来源文件列表"开始的所有内容
                        text = text[:start_pos]
                        print(f"[DEBUG CLEAN] 已清理来源文件列表及后续所有内容", file=sys.stderr)
                else:
                    # 没有找到文件名列表，直接删除"来源文件列表"及后续内容
                    text = text[:start_pos]
                    print(f"[DEBUG CLEAN] 已清理来源文件列表标题", file=sys.stderr)

        return text

    def _find_best_match_filename(self, invalid_name, valid_names, threshold=0.6):
        """找到最相似的文件名"""
        from difflib import SequenceMatcher

        best_match = None
        best_ratio = 0.0

        for valid_name in valid_names:
            # 计算相似度
            ratio = SequenceMatcher(None, invalid_name.lower(), valid_name.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = valid_name

        # 相似度阈值检查
        if best_ratio >= threshold:
            return best_match
        return None

    def _merge_table_rows(self, table_rows: list, seen_filenames: dict) -> str:
        """
        合并表格中相同来源文件的多行内容

        Args:
            table_rows: 表格行列表
            seen_filenames: 已见文件名映射

        Returns:
            合并后的完整表格字符串（包含表头）
        """
        import sys
        from collections import defaultdict

        # 解析表格，分离表头、分隔符、数据行
        header_line = None
        separator_line = None
        data_rows = []

        for row in table_rows:
            stripped = row.strip()
            if not stripped:
                continue

            if '来源文件' in stripped and '|' in stripped:
                header_line = stripped
            elif '---' in stripped and '|' in stripped:
                separator_line = stripped
            elif stripped.startswith('|') and stripped.endswith('|'):
                data_rows.append(stripped)

        if not header_line or not separator_line or not data_rows:
            return None

        # 按来源文件分组
        grouped_data = defaultdict(lambda: {'序号': None, '内容列表': []})

        for row in data_rows:
            parts = [p.strip() for p in row.split('|')]
            parts = [p for p in parts if p or len(parts) <= 3]

            if len(parts) >= 3:
                number_cell = parts[0]
                filename_cell = parts[1]
                content_cell = parts[2] if len(parts) > 2 else ''

                if filename_cell in ['序号', '来源文件', '内容', 'No.']:
                    continue

                # 提取序号中的 data-ref 编号
                import re
                ref_match = re.search(r'data-ref="(\d+)"', number_cell)
                ref_num = int(ref_match.group(1)) if ref_match else 999

                grouped_data[filename_cell]['序号'] = number_cell
                grouped_data[filename_cell]['ref_num'] = ref_num
                grouped_data[filename_cell]['内容列表'].append(content_cell)

        # 重建表格
        result_lines = [header_line, separator_line]

        # 按 ref_num 排序
        sorted_items = sorted(grouped_data.items(), key=lambda x: x[1]['ref_num'])

        for filename, data in sorted_items:
            # 合并内容：使用 <br> 标签
            if len(data['内容列表']) > 1:
                merged_content = '<br><br>'.join(data['内容列表'])
            else:
                merged_content = data['内容列表'][0] if data['内容列表'] else ''

            new_row = f"| {data['序号']} | {filename} | {merged_content} |"
            result_lines.append(new_row)

            if len(data['内容列表']) > 1:
                print(f"[DEBUG MERGE STREAM] 合并文件: {filename}, 内容块数: {len(data['内容列表'])}", file=sys.stderr)

        return '\n'.join(result_lines) + '\n'

    def _merge_table_content(self, table_content: str) -> str:
        """
        合并表格内容中相同来源文件的多行

        Args:
            table_content: 完整的表格内容字符串

        Returns:
            合并后的表格内容
        """
        import sys
        import re
        from collections import defaultdict

        lines = table_content.split('\n')
        header_line = None
        separator_line = None
        data_rows = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            if '来源文件' in stripped and '|' in stripped:
                header_line = stripped
            elif '---' in stripped and '|' in stripped:
                separator_line = stripped
            elif stripped.startswith('|') and stripped.endswith('|'):
                data_rows.append(stripped)

        if not header_line or not separator_line or not data_rows:
            return table_content  # 不是完整的表格，返回原内容

        # 按来源文件分组
        grouped_data = defaultdict(lambda: {'序号': None, '内容列表': [], 'ref_num': 999})

        for row in data_rows:
            # 提取序号、文件名、内容
            parts = [p.strip() for p in row.split('|')]
            parts = [p for p in parts if p or len(parts) <= 3]

            if len(parts) >= 3:
                number_cell = parts[0]
                filename_cell = parts[1]
                content_cell = parts[2] if len(parts) > 2 else ''

                if filename_cell in ['序号', '来源文件', '内容', 'No.']:
                    continue

                # 提取 data-ref 编号
                ref_match = re.search(r'data-ref="(\d+)"', number_cell)
                ref_num = int(ref_match.group(1)) if ref_match else 999

                grouped_data[filename_cell]['序号'] = number_cell
                grouped_data[filename_cell]['ref_num'] = ref_num
                grouped_data[filename_cell]['内容列表'].append(content_cell)

        # 重建表格
        result_lines = [header_line, separator_line]

        # 按 ref_num 排序
        sorted_items = sorted(grouped_data.items(), key=lambda x: x[1]['ref_num'])

        for filename, data in sorted_items:
            # 合并内容：使用 <br> 标签（在 Markdown 表格中正确换行）
            if len(data['内容列表']) > 1:
                merged_content = '<br><br>'.join(data['内容列表'])
            else:
                merged_content = data['内容列表'][0] if data['内容列表'] else ''

            new_row = f"| {data['序号']} | {filename} | {merged_content} |"
            result_lines.append(new_row)

            if len(data['内容列表']) > 1:
                print(f"[DEBUG MERGE STREAM] 合并文件: {filename}, 内容块数: {len(data['内容列表'])}", file=sys.stderr)

        return '\n'.join(result_lines)

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
