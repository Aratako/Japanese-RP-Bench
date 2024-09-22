import argparse
import json
import os

import yaml
from tqdm import tqdm

from japanese_rp_bench.data import load_dataset_wrapper
from japanese_rp_bench.evaluation import evaluate_conversation
from japanese_rp_bench.models import generate_response, load_model
from japanese_rp_bench.prompts import construct_system_prompts
from japanese_rp_bench.utils import setup_logging


def run_eval(config) -> None:
    logger = setup_logging()
    logger.info("処理を開始")
    # ステップ1: データセットの読み込み
    logger.info(
        f"データセットの読み込み: {config['dataset_repo']}, split: {config['dataset_split']}"
    )
    dataset = load_dataset_wrapper(
        config["dataset_repo"],
        split=config["dataset_split"],
        cache_dir=config["cache_dir"],
    )

    # ステップ2: モデルの読み込み
    logger.info("各モデルの読み込み")
    try:
        target_model, target_tokenizer = load_model(
            config["target_model_name"],
            config["target_inference_method"],
            config["tensor_parallel_size"],
            config["cache_dir"],
        )
        user_model, user_tokenizer = load_model(
            config["user_model_name"],
            config["user_inference_method"],
            config["tensor_parallel_size"],
            config["cache_dir"],
        )
        judge_model, judge_tokenizer = load_model(
            config["judge_model_name"],
            config["judge_inference_method"],
            config["tensor_parallel_size"],
            config["cache_dir"],
        )
    except Exception as e:
        logger.exception("モデルの読み込み中にエラーが発生しました")
        raise e

    # 評価プロンプトの読み込み
    logger.info(f"評価プロンプトの読み込み: {config['evaluation_prompt_file']}")
    with open(config["evaluation_prompt_file"], "r", encoding="utf-8") as f:
        evaluation_prompt = f.read()

    all_conversations = []
    all_evaluations = []

    # ステップ3: テストデータセットごとに処理
    logger.info("各データに対する処理を開始（推論+評価）")
    for test_case in tqdm(dataset):
        assistant_system_prompt, user_system_prompt, first_user_input = (
            construct_system_prompts(test_case)
        )
        # conversation_historyはlist of strとして持っておき、推論の際にconversationsとして再構築する
        conversation_history = [first_user_input]

        # ステップ4: 対話ループ
        for turn in range(config["max_turns"]):
            # 最初のターンの場合はアシスタントの応答のみを生成
            if turn == 0:
                conversations = [{"role": "user", "content": first_user_input}]
                assistant_response = generate_response(
                    target_model,
                    target_tokenizer,
                    config["target_model_name"],
                    config["target_inference_method"],
                    assistant_system_prompt,
                    conversations,
                )
                conversation_history.append(assistant_response)
            else:
                # まず、次のユーザーの入力を生成
                conversations = [{"role": "user", "content": "対話開始"}]
                for i, conversation in enumerate(conversation_history):
                    if i % 2 == 0:
                        conversations.append(
                            {"role": "assistant", "content": conversation}
                        )
                    else:
                        conversations.append({"role": "user", "content": conversation})
                user_input = generate_response(
                    user_model,
                    user_tokenizer,
                    config["user_model_name"],
                    config["user_inference_method"],
                    user_system_prompt,
                    conversations,
                )
                conversation_history.append(user_input)
                # 次に、アシスタント側の応答を生成
                conversations = []
                for i, conversation in enumerate(conversation_history):
                    if i % 2 == 0:
                        conversations.append({"role": "user", "content": conversation})
                    else:
                        conversations.append(
                            {"role": "assistant", "content": conversation}
                        )
                assistant_response = generate_response(
                    target_model,
                    target_tokenizer,
                    config["target_model_name"],
                    config["target_inference_method"],
                    assistant_system_prompt,
                    conversations,
                )
                conversation_history.append(assistant_response)

        # ステップ5: 対話データの保存
        conversation_id = test_case["id"]
        all_conversations.append(
            {
                "target_model_name": config["target_model_name"],
                "user_model_name": config["user_model_name"],
                "id": conversation_id,
                "conversation_history": conversation_history,
            }
        )

        # ステップ6: 評価の実行
        evaluation_result = evaluate_conversation(
            judge_model,
            judge_tokenizer,
            config["judge_model_name"],
            config["judge_inference_method"],
            evaluation_prompt,
            assistant_system_prompt,
            conversation_history,
        )
        # ステップ7: 評価結果のパース（jsonを想定）
        try:
            evaluation = json.loads(evaluation_result)
        except json.JSONDecodeError:
            logger.exception(
                "Invalid JSON format in evaluation result for ID {conversation_id}"
            )
            continue
        evaluation["target_model_name"] = config["target_model_name"]
        evaluation["user_model_name"] = config["user_model_name"]
        evaluation["judge_model_name"] = config["judge_model_name"]
        evaluation["id"] = conversation_id
        evaluation["conversation_history"] = conversation_history
        all_evaluations.append(evaluation)

    logger.info("推論と評価が完了")
    logger.info(f"推論成功件数: {len(all_conversations)}")
    logger.info(f"評価成功件数: {len(all_evaluations)}")
    # ステップ7: 最終結果の保存
    conversations_output_file = f"./conversations/{config['target_model_name'].replace('/', '-')}_{config['dataset_repo'].replace('/', '-')}.jsonl"
    with open(
        conversations_output_file,
        "w",
        encoding="utf-8",
    ) as f:
        for conversation in all_conversations:
            json.dump(conversation, f, ensure_ascii=False)
            f.write("\n")
    evaluations_output_file = f"./evaluations/{config['target_model_name'].replace('/', '-')}_{config['dataset_repo'].replace('/', '-')}.jsonl"
    with open(
        evaluations_output_file,
        "w",
        encoding="utf-8",
    ) as f:
        for evaluation in all_evaluations:
            json.dump(evaluation, f, ensure_ascii=False)
            f.write("\n")
    logger.info("全ての処理が完了")
    logger.info(f"推論の結果は{conversations_output_file}に保存されました")
    logger.info(f"評価の結果は{evaluations_output_file}に保存されました。")


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="設定ファイルへのパス (YAML形式)"
    )
    args = parser.parse_args()

    # YAMLファイルから設定を読み込む
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    os.makedirs("./conversations", exist_ok=True)
    os.makedirs("./evaluations", exist_ok=True)
    run_eval(config)
