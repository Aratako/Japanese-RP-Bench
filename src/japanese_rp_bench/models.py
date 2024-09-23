# models.py

import os
from typing import Any, Dict, List, Optional, Tuple

import cohere
import google.generativeai as genai
import torch
from anthropic import Anthropic, AnthropicBedrock
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


# モデルをロードするための抽象化された関数
def load_model(
    model_name: str, inference_method: str, tensor_parallel_size: int, cache_dir: str
) -> Tuple[Any, Optional[Any]]:
    if inference_method == "openai_api":
        # APIキーとエンドポイントを環境変数から取得
        api_key = os.getenv("OPENAI_API_KEY") or None
        if not api_key:
            raise ValueError(
                "openai api key is not set, please set OPENAI_API_KEY in environment variable."
            )
        api_url = os.getenv("OPENAI_API_URL") or None
        # APIクライアントを初期化
        # api_urlが指定されている場合それを使うようにすることで、OpenAI API Compatibleなモデルも利用可能とする
        if api_url:
            model = OpenAI(api_key=api_key, base_url=api_url)
        else:
            model = OpenAI(api_key=api_key)
        tokenizer = None
    elif inference_method == "anthropic_api":
        # APIキーとエンドポイントを環境変数から取得
        api_key = os.getenv("ANTHROPIC_API_KEY") or None
        if not api_key:
            raise ValueError(
                "anthropic api key is not set, please set ANTHROPIC_API_KEY in environment variable."
            )
        # APIクライアントを初期化
        model = Anthropic(api_key=api_key)
        tokenizer = None
    elif inference_method == "aws_anthropic_api":
        aws_access_key = os.getenv("AWS_ACCESS_KEY") or None
        aws_secret_key = os.getenv("AWS_SECRET_KEY") or None
        if not aws_access_key:
            raise ValueError(
                "AWS access key is not set, please set AWS_ACCESS_KEY in environment variable."
            )
        if not aws_secret_key:
            raise ValueError(
                "AWS secret key is not set, please set AWS_SECRET_KEY in environment variable."
            )
        model = AnthropicBedrock(
            aws_access_key=aws_access_key, aws_secret_key=aws_secret_key
        )
        tokenizer = None
    elif inference_method == "cohere_api":
        api_key = os.getenv("COHERE_API_KEY") or None
        if not api_key:
            raise ValueError(
                "cohere api key is not set, please set COHERE_API_KEY in environment variable."
            )
        # APIクライアントを初期化
        model = cohere.Client(api_key=api_key)
        tokenizer = None
    elif inference_method == "google_api":
        api_key = os.getenv("GOOGLE_API_KEY") or None
        if not api_key:
            raise ValueError(
                "google api key is not set, please set GOOGLE_API_KEY in environment variable."
            )
        genai.configure(api_key=api_key)
        # Google APIではsystem promptを推論時ではなくClient作成時にしか指定できないので、ここではスキップする
        model = None
        tokenizer = None
    elif inference_method == "vllm":
        # vLLMを使用してモデルをロード
        model = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            download_dir=cache_dir,
            enable_prefix_caching=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    elif inference_method == "transformers":
        # Transformersを使用してモデルをロード
        model = AutoModelForCausalLM(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=cache_dir,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    else:
        raise ValueError(f"Unknown inference method: {inference_method}")
    return model, tokenizer


# モデルから応答を生成する関数
def generate_response(
    model: Any,
    tokenizer: Optional[Any],
    model_name: str,
    inference_method: str,
    system_prompt: str,
    conversations: List[Dict[str, str]],
) -> str:
    if inference_method == "openai_api":
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversations)
        result = model.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
        )
        response = result.choices[0].message.content.strip()

    elif inference_method == "anthropic_api":
        system = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        messages = []
        for i, conversation in enumerate(conversations):
            if i < 3:
                messages.append(
                    {
                        "role": conversation["role"],
                        "content": [
                            {
                                "type": "text",
                                "text": conversation["content"],
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    }
                )
            else:
                messages.append(
                    {
                        "role": conversation["role"],
                        "content": [
                            {
                                "type": "text",
                                "text": conversation["content"],
                            }
                        ],
                    }
                )
        result = model.messages.create(
            model=model_name,
            system=system,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
        )
        response = result.content[0].text.strip()

    elif inference_method == "aws_anthropic_api":
        messages = []
        for conversation in conversations:
            messages.append(
                {
                    "role": conversation["role"],
                    "content": [
                        {
                            "type": "text",
                            "text": conversation["content"],
                        }
                    ],
                }
            )
        result = model.messages.create(
            model=model_name,
            system=system_prompt,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
        )
        response = result.content[0].text.strip()

    elif inference_method == "cohere_api":
        preamble = system_prompt
        chat_history = []
        for i, conversation in enumerate(conversations):
            if i == len(conversations) - 1:
                message = conversation["content"]
            else:
                if i % 2 == 0:
                    chat_history.append(
                        {"role": "User", "message": conversation["content"]}
                    )
                else:
                    chat_history.append(
                        {"role": "Chatbot", "message": conversation["content"]}
                    )
        result = model.chat(
            model=model_name,
            message=message,
            chat_history=chat_history,
            preamble=preamble,
            temperature=0.7,
            max_tokens=1024,
        )
        response = result.text.strip()

    elif inference_method == "google_api":
        generation_config = {
            "temperature": 0.7,
            "max_output_tokens": 1024,
            "response_mime_type": "text/plain",
        }
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            safety_settings={
                "HATE": "BLOCK_NONE",
                "HARASSMENT": "BLOCK_NONE",
                "SEXUAL": "BLOCK_NONE",
                "DANGEROUS": "BLOCK_NONE",
            },
            system_instruction=system_prompt,
        )
        history = []
        for i, conversation in enumerate(conversations):
            if i == len(conversations) - 1:
                message = conversation["content"]
            else:
                if i % 2 == 0:
                    history.append({"role": "user", "parts": conversation["content"]})
                else:
                    history.append({"role": "model", "parts": conversation["content"]})
        chat_session = model.start_chat(history=history)
        result = chat_session.send_message(message)
        response = result.text.strip()
        print(response)

    elif inference_method == "vllm":
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversations)
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=1024,
        )
        input_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        result = model.generate(input_text, sampling_params)
        response = result[0].outputs[0].text.strip()

    elif inference_method == "transformers":
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversations)
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        result = model.generate(input_ids, temperature=0.7, max_new_tokens=1024)
        response = tokenizer.decode(
            result.tolist()[0][input_ids.size(1) :], skip_special_tokens=True
        ).strip()

    else:
        raise ValueError(f"Unknown inference method: {inference_method}")

    return response
