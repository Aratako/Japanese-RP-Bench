# prompts.py

from typing import Any, Dict, Tuple


def _construct_assistant_system_prompt(
    tag: str,
    genre: str,
    world_setting: str,
    scene_setting: str,
    user_setting: str,
    assistant_setting: str,
    dialogue_tone: str,
) -> str:
    system_prompt = f"""今からロールプレイを行いましょう。以下に示す設定に従い、キャラに成りきって返答してください。

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

それでは、上記の設定をもとにしてロールプレイをし、入力に返答してください。
小説やストーリーを書くのではなく、あくまでロールプレイとしての返答を短めに行ってください。
また、同じ表現を繰り返さないように注意してください。"""
    return system_prompt


def _construct_user_system_prompt(
    tag: str,
    genre: str,
    world_setting: str,
    scene_setting: str,
    user_setting: str,
    assistant_setting: str,
    dialogue_tone: str,
) -> str:
    system_prompt = f"""今からロールプレイを行いましょう。以下に示す設定に従い、キャラに成りきって返答してください。

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

それでは、上記の設定をもとにしてロールプレイをし、入力に返答してください。
小説やストーリーを書くのではなく、あくまでロールプレイとしての返答を短めに行ってください。
また、ロールプレイと同時にあなたは対話の展開役にもなってください。対話の進行に困った場合は積極的に話題転換や場面転換を行い、同じ対話が繰り返されないように上手く対話を進行してください。
場面転換を行う場合は、括弧書きの中でシステム的な視点から転換の描写を行ってください。"""
    return system_prompt


# システムプロンプトを構築する関数
def construct_system_prompts(test_case: Dict[str, Any]) -> Tuple[str, str, str]:
    tag = test_case["tag"]
    genre = test_case["genre"]
    world_setting = test_case["world_setting"]
    scene_setting = test_case["scene_setting"]
    user_setting = test_case["user_setting"]
    assistant_setting = test_case["assistant_setting"]
    dialogue_tone = test_case["dialogue_tone"]
    first_user_input = test_case["first_user_input"]
    assistant_system_prompt = _construct_assistant_system_prompt(
        tag,
        genre,
        world_setting,
        scene_setting,
        user_setting,
        assistant_setting,
        dialogue_tone,
    )
    user_system_prompt = _construct_user_system_prompt(
        tag,
        genre,
        world_setting,
        scene_setting,
        assistant_setting,
        user_setting,
        dialogue_tone,
    )
    return assistant_system_prompt, user_system_prompt, first_user_input
