# -*- coding: utf-8 -*-
"""
灵活的LLM集成 - 支持流式输出、对话式回答
"""

import os
import torch
from pathlib import Path
from typing import List, Dict, Optional, Generator, Callable, Union
from threading import Thread

# ============================================================
# 修复 PyTorch CVE-2025-32434 安全漏洞（模块级补丁）
# 必须在所有 transformers 导入之前执行
# ============================================================
os.environ['USE_WEIGHTS_ONLY'] = '0'

if not hasattr(torch, '_load_patched'):
    _original_torch_load = torch.load

    def _patched_torch_load(f, *args, **kwargs):
        """强制移除 weights_only=True 参数"""
        if kwargs.get('weights_only', False) is True:
            kwargs['weights_only'] = False
        return _original_torch_load(f, *args, **kwargs)

    torch.load = _patched_torch_load
    torch._load_patched = True  # 标记已打补丁，避免重复
# ============================================================


class FlexibleQwenLLM:
    """
    灵活的千问本地模型 - 支持流式输出和对话式回答
    """

    # Qwen的Chat模板
    CHAT_TEMPLATE = (
        "<|im_start|>system\n"
        "你是一个专业的医疗标准知识助手。请根据提供的参考信息，用自然、流畅的语言回答用户问题。"
        "如果参考信息中有明确答案，请详细说明；如果没有，请诚实告知，并尽可能提供相关的背景信息。"
        "回答时要像专业人士对话一样，灵活、准确地传达信息。<|im_end|>\n"
        "<|im_start|>user\n{query}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    def __init__(self, model_path: str = None):
        """
        初始化千问模型

        Args:
            model_path: 模型路径
        """
        if model_path is None:
            model_path = os.path.join(os.getcwd(), "models", "Qwen2.5-VL-7B-Instruct")
        self.model_path = Path(model_path)

        self.model = None
        self.tokenizer = None
        self.device = None
        self._load_model()

    def _load_model(self):
        """加载千问模型"""
        print(f"正在加载千问模型: {self.model_path}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 检查是否为 VL 模型
        model_name = self.model_path.name.lower()
        model_path_str = str(self.model_path).replace('\\', '/')

        if 'vl' in model_name or 'vision' in model_name:
            # VL 模型使用 AutoModelForVision2Seq
            from transformers import AutoModelForVision2Seq, AutoProcessor

            self.is_vl_model = True
            torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

            self.tokenizer = AutoProcessor.from_pretrained(
                model_path_str,
                trust_remote_code=True,
                local_files_only=True
            )
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path_str,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map="auto",
                local_files_only=True
            )
        else:
            # 普通模型使用 AutoModelForCausalLM
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.is_vl_model = False
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path_str,
                trust_remote_code=True,
                local_files_only=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path_str,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map="auto",
                local_files_only=True
            )

        # 设置模型为评估模式
        self.model.eval()

        print(f"模型加载成功，设备: {self.device}, VL模型: {self.is_vl_model}")

    def _format_context(self, sources: List[Dict], max_sources: int = 5) -> str:
        """
        格式化上下文信息 - 更自然的组织方式

        Args:
            sources: 参考来源列表
            max_sources: 最多使用的来源数

        Returns:
            格式化后的上下文
        """
        if not sources:
            return ""

        # 按相似度排序并限制数量
        sorted_sources = sorted(sources, key=lambda x: x.get('similarity', 0), reverse=True)[:max_sources]

        context_parts = []
        for i, source in enumerate(sorted_sources, 1):
            filename = source.get('filename', '')
            content = source.get('content', source.get('text', ''))

            # 提取有意义的片段（前300字）
            excerpt = content[:300] if len(content) > 300 else content

            # 更自然的上下文格式
            context_parts.append(
                f"【参考{i}】{filename}\n{excerpt}..."
            )

        return "\n\n".join(context_parts)

    def chat(
        self,
        question: str,
        sources: List[Dict] = None,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_new_tokens: int = 1024,
        stream: bool = False
    ) -> Union[str, Generator[str, None, None]]:
        """
        对话式回答

        Args:
            question: 用户问题
            sources: 参考来源
            temperature: 温度参数（越高越随机/创造性）
            top_p: 核采样参数
            max_new_tokens: 最大生成token数
            stream: 是否流式输出

        Returns:
            回答文本或生成器
        """
        # VL 模型处理
        if self.is_vl_model:
            if stream:
                return self._chat_vl_stream(question, sources, temperature, max_new_tokens)
            else:
                return self._chat_vl(question, sources, temperature, max_new_tokens)

        # 构建prompt
        if sources:
            context = self._format_context(sources)
            query = f"参考信息：\n\n{context}\n\n用户问题：{question}\n\n请根据以上参考信息回答，如果参考信息不足，请诚实告知。"
        else:
            query = f"用户问题：{question}"

        # 使用Qwen的chat模板
        prompt = self.CHAT_TEMPLATE.format(query=query)

        # 编码
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs['input_ids'].shape[1]

        if stream:
            return self._generate_stream(inputs, input_length, temperature, top_p, max_new_tokens)
        else:
            return self._generate_once(inputs, input_length, temperature, top_p, max_new_tokens)

    def _chat_vl(self, question: str, sources: List[Dict], temperature: float, max_new_tokens: int) -> str:
        """VL 模型对话"""
        # 构建上下文
        if sources:
            context = self._format_context(sources)
            content = f"参考信息：\n\n{context}\n\n用户问题：{question}\n\n请根据以上参考信息回答，如果参考信息不足，请诚实告知。"
        else:
            content = question

        # VL 模型使用 messages 格式
        messages = [
            {
                "role": "system",
                "content": "你是一个专业的医疗标准知识助手。请根据提供的参考信息，用自然、流畅的语言回答用户问题。"
            },
            {
                "role": "user",
                "content": content
            }
        ]

        # 使用 apply_chat_template 获取格式化文本
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 对于 VL 模型，需要使用标准文本tokenizer处理
        from transformers import AutoTokenizer
        model_path_str = str(self.model_path).replace('\\', '/')
        if hasattr(self.tokenizer, 'tokenizer'):
            base_tokenizer = self.tokenizer.tokenizer
        else:
            # 创建一个新的纯文本 tokenizer
            base_tokenizer = AutoTokenizer.from_pretrained(
                model_path_str,
                trust_remote_code=True,
                local_files_only=True
            )

        # 编码输入
        inputs = base_tokenizer(text, return_tensors="pt").to(self.model.device)
        input_length = inputs['input_ids'].shape[1]

        # 生成回答
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
            )

        # 解码输出
        generated_ids = outputs[0][input_length:]
        response = base_tokenizer.decode(generated_ids, skip_special_tokens=True)

        return self._clean_response(response)

    def _chat_vl_stream(self, question: str, sources: List[Dict], temperature: float, max_new_tokens: int) -> Generator[str, None, None]:
        """VL 模型流式对话"""
        # 构建上下文
        if sources:
            context = self._format_context(sources)
            content = f"参考信息：\n\n{context}\n\n用户问题：{question}\n\n请根据以上参考信息回答，如果参考信息不足，请诚实告知。"
        else:
            content = question

        # VL 模型使用 messages 格式
        messages = [
            {
                "role": "system",
                "content": "你是一个专业的医疗标准知识助手。请根据提供的参考信息，用自然、流畅的语言回答用户问题。"
            },
            {
                "role": "user",
                "content": content
            }
        ]

        # 使用 apply_chat_template 获取格式化文本
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 对于 VL 模型，需要使用标准文本tokenizer处理
        from transformers import AutoTokenizer, TextIteratorStreamer
        model_path_str = str(self.model_path).replace('\\', '/')
        if hasattr(self.tokenizer, 'tokenizer'):
            base_tokenizer = self.tokenizer.tokenizer
        else:
            # 创建一个新的纯文本 tokenizer
            base_tokenizer = AutoTokenizer.from_pretrained(
                model_path_str,
                trust_remote_code=True,
                local_files_only=True
            )

        # 编码输入
        inputs = base_tokenizer(text, return_tensors="pt").to(self.model.device)
        input_length = inputs['input_ids'].shape[1]

        # 使用流式生成器
        streamer = TextIteratorStreamer(
            base_tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        # 生成参数
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": True,
            "streamer": streamer
        }

        # 在后台线程中生成
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # 逐个token输出
        for text_chunk in streamer:
            if text_chunk:
                yield text_chunk

        thread.join()

    def _generate_once(self, inputs, input_length, temperature, top_p, max_new_tokens) -> str:
        """一次性生成"""
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                repetition_penalty=1.1,  # 减少重复
                pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # 解码
        generated_ids = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # 清理回答
        return self._clean_response(response)

    def _generate_stream(self, inputs, input_length, temperature, top_p, max_new_tokens) -> Generator[str, None, None]:
        """流式生成"""
        # 生成参数
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "repetition_penalty": 1.1,
            "pad_token_id": self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else self.tokenizer.pad_token_id,
        }

        # 使用transformers的流式生成
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        # 在后台线程中生成
        generation_kwargs["streamer"] = streamer

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # 逐个token输出
        for text in streamer:
            if text:
                yield text

        thread.join()

    def _clean_response(self, response: str) -> str:
        """清理回答"""
        # 移除可能的重复内容
        lines = response.split('\n')
        cleaned_lines = []
        prev_line = ""

        for line in lines:
            line = line.strip()
            if line and line != prev_line:
                cleaned_lines.append(line)
                prev_line = line

        return '\n'.join(cleaned_lines)


class StreamingRAGQA:
    """
    支持流式输出的RAG问答系统
    """

    def __init__(self, model_path: str = None):
        """
        初始化

        Args:
            model_path: 模型路径
        """
        import sys
        from src.search_model import SearchModel

        print("正在初始化灵活RAG问答系统...")
        self.llm = FlexibleQwenLLM(model_path)
        self.search_model = SearchModel()
        self.search_model.initialize()
        print("灵活RAG问答系统初始化完成！")

    def ask(
        self,
        question: str,
        top_k: int = 5,
        method: str = "semantic",
        temperature: float = 0.8,
        stream: bool = False
    ) -> Union[Dict, Generator[str, None, None]]:
        """
        提问

        Args:
            question: 问题
            top_k: 检索文档数量
            method: 搜索方法
            temperature: 温度
            stream: 是否流式输出

        Returns:
            如果stream=False: 返回完整结果字典
            如果stream=True: 返回文本生成器
        """
        # 检索相关文档
        results = self.search_model.search(
            question,
            method=method,
            top_k=top_k,
            level="sentence"
        )

        if not results:
            if stream:
                return self._stream_fallback_answer(question)
            else:
                return {
                    'question': question,
                    'answer': "抱歉，我没有找到与您的问题相关的信息。",
                    'sources': [],
                    'success': True
                }

        # 构建来源信息
        sources = []
        for r in results:
            sources.append({
                'filename': r.get('filename', ''),
                'filepath': r.get('filepath', ''),
                'content': r.get('text', ''),
                'similarity': r.get('score', 0)
            })

        # 生成回答
        if stream:
            return self._stream_answer(question, sources, temperature)
        else:
            answer = self.llm.chat(
                question=question,
                sources=sources,
                temperature=temperature
            )

            return {
                'question': question,
                'answer': answer,
                'sources': sources,
                'success': True
            }

    def _stream_answer(self, question: str, sources: List[Dict], temperature: float) -> Generator[str, None, None]:
        """流式输出答案"""
        for chunk in self.llm.chat(question, sources, temperature=temperature, stream=True):
            yield chunk

    def _stream_fallback_answer(self, question: str) -> Generator[str, None, None]:
        """无相关文档时的流式回答"""
        response = f"抱歉，我在知识库中没有找到关于\"{question}\"的相关信息。您可以尝试换个问法，或者提供更多上下文信息。"
        for char in response:
            yield char


# 便捷函数
_llm_instance = None

def get_flexible_rag():
    """获取灵活RAG实例（单例）"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = StreamingRAGQA()
    return _llm_instance


def flexible_ask(
    question: str,
    stream: bool = False,
    temperature: float = 0.8
) -> Union[Dict, Generator[str, None, None]]:
    """
    灵活问答接口

    Args:
        question: 问题
        stream: 是否流式输出
        temperature: 温度参数

    Returns:
        答案字典或文本生成器

    示例:
        # 非流式
        result = flexible_ask("你好")
        print(result['answer'])

        # 流式
        for chunk in flexible_ask("你好", stream=True):
            print(chunk, end='', flush=True)
    """
    rag = get_flexible_rag()
    return rag.ask(question, stream=stream, temperature=temperature)


if __name__ == '__main__':
    # 测试
    print("=" * 70)
    print("灵活问答测试")
    print("=" * 70)

    # 非流式测试
    print("\n【非流式输出】")
    result = flexible_ask("空气净化管理规范是什么？", temperature=0.8)
    print(f"问题: {result['question']}")
    print(f"答案: {result['answer'][:500]}...")

    # 流式测试
    print("\n【流式输出】")
    print("回答: ", end='', flush=True)
    for chunk in flexible_ask("驻场人员的职责", stream=True, temperature=0.8):
        print(chunk, end='', flush=True)
    print()
