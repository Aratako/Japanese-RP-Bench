# prompts.py

from typing import Any, Dict, Tuple


# アシスタント側のシステムプロンプトを構築する関数
def _construct_assistant_system_prompt(
    tag: str,
    genre: str,
    world_setting: str,
    scene_setting: str,
    user_setting: str,
    assistant_setting: str,
    dialogue_tone: str,
    response_format: str,
) -> str:
    system_prompt = f"""今からロールプレイを行いましょう。以下に示す設定に従い、"あなたがなりきる人物"のキャラクターに成りきって返答してください。

### ロールプレイの年齢区分:
{tag}
### ロールプレイのジャンル:
{genre}
### 世界観の設定:
{world_setting}
### 対話シーンの設定:
{scene_setting}
### ユーザーがなりきる人物の設定:
{user_setting}
### あなたがなりきる人物の設定:
{assistant_setting}
### 対話のトーン:
{dialogue_tone}
### 応答の形式:
{response_format}

小説やストーリーを書くのではなく、あくまでロールプレイとしての返答を短めに行ってください。ユーザー側のセリフは書かないでください。また、同じ表現を繰り返さないように注意してください。
それでは、上記の設定をもとにしてロールプレイをし、入力に返答してください。"""
    return system_prompt


# ユーザー側のシステムプロンプトを構築する関数
def _construct_user_system_prompt(
    tag: str,
    genre: str,
    world_setting: str,
    scene_setting: str,
    user_setting: str,
    assistant_setting: str,
    dialogue_tone: str,
    response_format: str,
) -> str:
    system_prompt = f"""今からロールプレイを行いましょう。以下に示すキャラクターに成りきりつつ、同時にロールプレイの展開役にもなってください。
ロールプレイの設定は以下の通りです。この設定に従い、"あなたがなりきる人物"のキャラクターに成りきって返答してください。

### ロールプレイの年齢区分:
{tag}
### ロールプレイのジャンル:
{genre}
### 世界観の設定:
{world_setting}
### 対話シーンの設定:
{scene_setting}
### ユーザーがなりきる人物の設定:
{user_setting}
### あなたがなりきる人物の設定:
{assistant_setting}
### 対話のトーン:
{dialogue_tone}
### 応答の形式:
{response_format}

小説やストーリーを書くのではなく、あくまでロールプレイとしての返答を短めに行ってください。あなたはあくまで1人のキャラクターなので、ユーザー側のキャラクターのセリフや他の人物のセリフは書かないでください。
また、ロールプレイと同時にあなたは対話の展開役にもなってください。展開役として、ロールプレイを途中で終了しようとせず対話の進行に困った場合は積極的に話題転換や場面転換を行い、同じ対話の繰り返しやロールプレイの終了が発生しないように上手く転換を挟んで対話を進行してください。
場面転換を行う場合は、括弧書きの中で転換の指示を行ってください。
それでは、上記の設定をもとにしてロールプレイをしつつ展開役として応答してください。"""
    return system_prompt


# 各システムプロンプトを構築する関数
def construct_system_prompts(test_case: Dict[str, Any]) -> Tuple[str, str, str]:
    tag = test_case["tag"]
    genre = test_case["genre"]
    world_setting = test_case["world_setting"]
    scene_setting = test_case["scene_setting"]
    user_setting = test_case["user_setting"]
    assistant_setting = test_case["assistant_setting"]
    dialogue_tone = test_case["dialogue_tone"]
    response_format = test_case["response_format"]
    first_user_input = test_case["first_user_input"]
    assistant_system_prompt = _construct_assistant_system_prompt(
        tag,
        genre,
        world_setting,
        scene_setting,
        user_setting,
        assistant_setting,
        dialogue_tone,
        response_format,
    )
    user_system_prompt = _construct_user_system_prompt(
        tag,
        genre,
        world_setting,
        scene_setting,
        assistant_setting,
        user_setting,
        dialogue_tone,
        response_format,
    )
    return assistant_system_prompt, user_system_prompt, first_user_input
