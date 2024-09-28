# utils.py

import logging

import regex as re


# loggingを有効化する関数
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


# json stringをescapeする関数
def extract_and_escape_json_string(input_string):
    json_pattern = r"(\{(?:[^{}]|(?R))*\}|\[(?:[^\[\]]|(?R))*\])"
    escape_pattern = r"\"(.*?)\""

    extracted_json = re.findall(json_pattern, input_string, re.DOTALL)[0]

    def replace_newlines(match):
        # 改行をエスケープする
        return '"' + match.group(1).replace("\n", "\\n") + '"'

    # Apply regex replacement
    escaped_string = re.sub(
        escape_pattern, replace_newlines, extracted_json, flags=re.DOTALL
    )

    return escaped_string


# 評価結果jsonの形式が正しいかをチェックする関数
def is_valid_evaluation(evaluation):
    required_keys = [
        "Roleplay Adherence",
        "Consistency",
        "Contextual Understanding",
        "Expressiveness",
        "Creativity",
        "Naturalness of Japanese",
        "Enjoyment of the Dialogue",
        "Appropriateness of Turn-Taking",
    ]
    return all(key in evaluation for key in required_keys)
