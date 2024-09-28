# evaluation.py

from typing import Any, List, Optional

import google.generativeai as genai
from vllm import SamplingParams


# 評価を実行する関数
def evaluate_conversation(
    model: Any,
    tokenizer: Optional[Any],
    model_name: str,
    inference_method: str,
    evaluation_prompt: str,
    system_prompt: str,
    conversation_history: List[str],
) -> str:
    # 評価のための入力プロンプトを構築
    input_text = "The role-play setting is as follows:\n----\n"
    input_text += system_prompt + "\n----\n"
    input_text += "The conversation history is as follows:\n----"
    for i, conversation in enumerate(conversation_history):
        if i % 2 == 0:
            input_text += "\nUser: " + conversation
        else:
            input_text += "\nAssistant: " + conversation
    input_text += "\n----\nBased on the above settings and conversation history, please evaluate the assistant's response. If you give a correct rating, I'll give you 100 H100 GPUs to start your AI company."

    # 評価モデルに入力を与えて評価を取得
    # OPENAI APIの場合
    if inference_method == "openai_api":
        if "o1" in model_name:
            # o1はシステムプロンプトをサポートしていないのでシステムプロンプトと最初の会話を結合
            messages = [
                {
                    "role": "user",
                    "content": f"{evaluation_prompt}\n\n{input_text}",
                }
            ]
            result = model.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=1,  # temparetureは1以外サポートされていない
            )
        else:
            messages = [{"role": "system", "content": evaluation_prompt}]
            messages.append({"role": "user", "content": input_text})
            result = model.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
        evaluation_result = result.choices[0].message.content.strip()

    # ANTHROPIC APIの場合
    elif inference_method == "anthropic_api":
        system = [
            {
                "type": "text",
                "text": evaluation_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        messages = []
        messages.append(
            {"role": "user", "content": [{"type": "text", "text": input_text}]}
        )
        # "{"でprefillすることでjsonを出力させる
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": '{"Evaluation Reason":'}],
            }
        )
        result = model.messages.create(
            model=model_name,
            system=system,
            messages=messages,
            temperature=0,
            max_tokens=1024,
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
        )
        evaluation_result = '{"Evaluation Reason": ' + result.content[0].text.strip()

    # Amazon BedrockのAnthropic APIの場合
    elif inference_method == "aws_anthropic_api":
        system = evaluation_prompt
        messages = []
        messages.append(
            {"role": "user", "content": [{"type": "text", "text": input_text}]}
        )
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": '{"Evaluation Reason":'}],
            }
        )
        result = model.messages.create(
            model=model_name,
            system=system,
            messages=messages,
            temperature=0,
            max_tokens=1024,
        )
        evaluation_result = '{"Evaluation Reason": ' + result.content[0].text.strip()

    # Google AI APIの場合
    elif inference_method == "google_api":
        generation_config = {
            "temperature": 0,
            "max_output_tokens": 1024,
            "response_mime_type": "application/json",
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
            system_instruction=evaluation_prompt,
        )
        message = input_text
        chat_session = model.start_chat()
        result = chat_session.send_message(message)
        evaluation_result = result.text.strip()

    # vLLMを使ってローカルで推論する場合
    elif inference_method == "vllm":
        messages = [{"role": "system", "content": evaluation_prompt}]
        messages.append({"role": "user", "content": input_text})
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1024,
        )
        result = model.generate(messages, sampling_params)
        evaluation_result = result.outputs[0].text.strip()

    # transformersを使ってローカルで推論する場合
    elif inference_method == "transformers":
        messages = [{"role": "system", "content": evaluation_prompt}]
        messages.append({"role": "user", "content": input_text})
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        result = model.generate(input_ids, temperature=0, max_new_tokens=1024)
        evaluation_result = tokenizer.decode(
            result.tolist()[0][input_ids.size(1) :], skip_special_tokens=True
        ).strip()

    else:
        raise ValueError(f"Unknown inference method: {inference_method}")

    return evaluation_result
