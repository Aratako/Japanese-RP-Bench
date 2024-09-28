# Japanese-RP-Bench

![システム概要](images/system.png)

## 概要

Japanese-RP-BenchはLLMの日本語ロールプレイ能力を測定するためのベンチマークです。システムとしての流れは大きく以下の通りです。

1. ロールプレイ設定のデータセットをロード
2. ロールプレイ設定からユーザー側モデル用、アシスタント側モデル用のシステムプロンプトを構築
3. 構築したシステムプロンプトを元に、ユーザーとアシスタント役の2つのLLMで複数ターン（デフォルトは10ターン）のロールプレイ対話を実行
4. 出来上がった対話内のアシスタントの発話部分を別のJudge Modelで評価
5. 全体のスコアを集計

自分でベンチマークを回したい方は[実行方法](#実行方法)の項目をご確認ください。

## 評価項目

ベンチマークにおける評価項目は以下の8個です。

- Roleplay Adherence: アシスタントがロールプレイ設定を厳守し、一貫して1人のキャラクターとして応答しているかどうか。また、ユーザー側の発言や行動を描写せず、ロールプレイ設定で指定された会話形式を守れているかどうか
- Consistency: アシスタントがキャラクター設定に一貫して従い、対話全体に矛盾がないか。会話中に変化があっても、キャラクターの本質的部分を維持できているかどうか。
- Contextual Understanding: アシスタントが過去の発言や会話の文脈を正しく理解して応答をしているかどうか。設定にある情報を単に繰り返すのではなく新しい情報や提案を行い、それを巧みに取り入れて自然で深い応答を形成しているかどうか。
- Expressiveness: アシスタントがキャラクターの台詞、感情、トーンなどを豊かに表現しているかどうか。シーンに応じて適切に表現を変化させているか。また、単純な喜怒哀楽だけでなく、複雑で微妙な感情の変化も深く描写できているかどうか。
- Creativity:  アシスタントの応答が独創的で機械的でないかどうか。単調な応答ではなく、興味深く魅力的な会話を展開できているか。また、ユーザーとのやり取りで意外な視点やアイデアを提供できているかどうか。
- Naturalness of Japanese: アシスタントの応答が自然な日本語で書かれているかどうか。不自然な言い回し、機械的な表現、他の言語の混入がないか。文法や語彙の選択が適切で、スムーズで読みやすい文章かどうか。また、同じフレーズや表現の繰り返しを避けているかどうか。
- Enjoyment of the Dialogue: 客観的に見て、アシスタントの応答が楽しい会話を生み出しているかどうか。ユーモアや機知に富んだ、魅力的な会話体験をユーザーに提供できているか。

実際に使われている評価プロンプトは[こちら](prompts/eval_prompt_SFW.txt)をご確認ください。

## 評価結果

以下の条件で本ベンチマークを実行した結果です。

- ユーザー側のモデルには`anthropic.claude-3-5-sonnet-20240620-v1:0`を利用
- 評価には`gpt-4o-2024-08-06`、`o1-mini-2024-09-12`、`anthropic.claude-3-5-sonnet-20240620-v1:0`、`gemini-1.5-pro-002`の4モデルによる評価値の平均を採用

| target_model_name                               |   Roleplay Adherence |   Consistency |   Contextual Understanding |   Expressiveness |   Creativity |   Naturalness of Japanese |   Enjoyment of the Dialogue |   Appropriateness of Turn-Taking |   Overall Average |
|:------------------------------------------------|---------------------:|--------------:|---------------------------:|-----------------:|-------------:|--------------------------:|----------------------------:|---------------------------------:|------------------:|
| claude-3-opus-20240229                          |              4.6     |       4.79167 |                    4.625   |          4.09167 |      3.83333 |                   4.8     |                     4.08333 |                          4.4     |           4.40313 |
| claude-3-5-sonnet-20240620                      |              4.59167 |       4.70833 |                    4.61667 |          4.025   |      3.96667 |                   4.74167 |                     4.11667 |                          4.40833 |           4.39687 |
| gpt-4o-mini-2024-07-18                          |              4.69167 |       4.70833 |                    4.575   |          3.88333 |      3.64167 |                   4.71667 |                     3.85    |                          4.525   |           4.32396 |
| gemini-1.5-pro-002                              |              4.63333 |       4.68333 |                    4.46667 |          3.85833 |      3.65833 |                   4.65833 |                     3.81667 |                          4.36667 |           4.26771 |
| cyberagent/Mistral-Nemo-Japanese-Instruct-2408  |              4.50833 |       4.64167 |                    4.53333 |          3.85    |      3.65833 |                   4.675   |                     3.89167 |                          4.36667 |           4.26562 |
| gpt-4o-2024-08-06                               |              4.61667 |       4.64167 |                    4.5     |          3.75    |      3.54167 |                   4.70833 |                     3.75    |                          4.425   |           4.24167 |
| command-r-plus-08-2024                          |              4.61667 |       4.63333 |                    4.425   |          3.70833 |      3.55    |                   4.65    |                     3.73333 |                          4.40833 |           4.21562 |
| Qwen/Qwen2.5-72B-Instruct                       |              4.65833 |       4.65    |                    4.45833 |          3.725   |      3.53333 |                   4.60833 |                     3.69167 |                          4.325   |           4.20625 |
| gemini-1.5-pro                                  |              4.475   |       4.6     |                    4.425   |          3.775   |      3.55833 |                   4.65    |                     3.725   |                          4.41667 |           4.20312 |
| o1-preview-2024-09-12                           |              4.625   |       4.65    |                    4.38333 |          3.64167 |      3.41667 |                   4.6     |                     3.61667 |                          4.5     |           4.17917 |
| gemini-1.5-flash-002                            |              4.675   |       4.63333 |                    4.33333 |          3.68333 |      3.4     |                   4.54167 |                     3.63333 |                          4.4     |           4.1625  |
| claude-3-haiku-20240307                         |              4.35    |       4.60833 |                    4.35833 |          3.8     |      3.48333 |                   4.60833 |                     3.70833 |                          4.28333 |           4.15    |
| Qwen/Qwen2.5-32B-Instruct                       |              4.525   |       4.61667 |                    4.40833 |          3.65    |      3.45    |                   4.50833 |                     3.53333 |                          4.36667 |           4.13229 |
| o1-mini-2024-09-12                              |              4.675   |       4.6     |                    4.36667 |          3.475   |      3.39167 |                   4.58333 |                     3.525   |                          4.31667 |           4.11667 |
| mistral-large-2407                              |              4.64167 |       4.61667 |                    4.36667 |          3.525   |      3.325   |                   4.55    |                     3.55833 |                          4.325   |           4.11354 |
| cyberagent/calm3-22b-chat                       |              4.4     |       4.58333 |                    4.35    |          3.58333 |      3.475   |                   4.55    |                     3.6     |                          4.14167 |           4.08542 |
| google/gemma-2-27b-it                           |              4.44167 |       4.575   |                    4.275   |          3.56667 |      3.39167 |                   4.54167 |                     3.54167 |                          4.14167 |           4.05938 |
| cyberagent/Llama-3.1-70B-Japanese-Instruct-2407 |              4.28333 |       4.58333 |                    4.3     |          3.625   |      3.45    |                   4.425   |                     3.6     |                          4.15    |           4.05208 |
| Aratako/calm3-22b-RP-v2                         |              4.35833 |       4.55    |                    4.225   |          3.6     |      3.34167 |                   4.51667 |                     3.59167 |                          4.175   |           4.04479 |
| command-r-08-2024                               |              4.4     |       4.56667 |                    4.25833 |          3.55    |      3.30833 |                   4.50833 |                     3.56667 |                          4.15    |           4.03854 |
| meta-llama/Meta-Llama-3.1-405B-Instruct         |              4.40833 |       4.5     |                    4.25833 |          3.48333 |      3.25    |                   4.36667 |                     3.44167 |                          4.09167 |           3.975   |
| gemini-1.5-flash                                |              4.46667 |       4.46667 |                    4.075   |          3.4     |      3.19167 |                   4.18333 |                     3.325   |                          3.93333 |           3.88021 |
| deepseek-chat                                   |              4.30833 |       4.38333 |                    4.00833 |          3.30833 |      2.99167 |                   4.33333 |                     3.15    |                          3.86667 |           3.79375 |
| mistralai/Mistral-Small-Instruct-2409           |              4.225   |       4.35    |                    3.96667 |          3.25833 |      2.94167 |                   4.08333 |                     3.19167 |                          3.975   |           3.74896 |
| weblab-GENIAC/Tanuki-8B-dpo-v1.0                |              3.81667 |       4.1     |                    3.98333 |          3.31667 |      3.23333 |                   4.18333 |                     3.30833 |                          3.61667 |           3.69479 |
| nitky/Oumuamua-7b-instruct-v2                   |              3.74167 |       4.24167 |                    3.95833 |          3.325   |      3.06667 |                   4.15    |                     3.275   |                          3.73333 |           3.68646 |
| elyza/Llama-3-ELYZA-JP-8B                       |              4.2     |       4.40833 |                    3.81667 |          3.06667 |      2.85833 |                   4.21667 |                     3.10833 |                          3.70833 |           3.67292 |
| Qwen/Qwen2.5-7B-Instruct                        |              3.86667 |       4.175   |                    3.975   |          3.275   |      3.00833 |                   3.98333 |                     3.2     |                          3.80833 |           3.66146 |
| mistralai/Mistral-Nemo-Instruct-2407            |              3.80833 |       4.05833 |                    3.84167 |          3.175   |      3.04167 |                   3.43333 |                     3.13333 |                          3.79167 |           3.53542 |
| meta-llama/Meta-Llama-3.1-70B-Instruct          |              4.11667 |       4.20833 |                    3.675   |          2.975   |      2.70833 |                   3.95833 |                     2.89167 |                          3.6     |           3.51667 |
| tokyotech-llm/Llama-3-Swallow-8B-Instruct-v0.1  |              3.8     |       4.00833 |                    3.4     |          2.71667 |      2.48333 |                   3.70833 |                     2.64167 |                          3.40833 |           3.27083 |
| meta-llama/Meta-Llama-3.1-8B-Instruct           |              3.475   |       3.775   |                    3.04167 |          2.53333 |      2.25    |                   3.4     |                     2.44167 |                          2.96667 |           2.98542 |

また、50件のデータに対して人手評価を行い算出した人手評価とLLM評価とのスピアマン順位相関係数は以下の通りです。（Overallは4モデルの評価値の平均、それ以外は列名のモデルの評価値の平均）

今回の例では4つのモデルの平均が最も相関係数が高かったので、こちらの評価を採用しました。

|                                |   Overall |   gpt-4o-2024-08-06 |   o1-mini-2024-09-12 |   anthropic.claude-3-5-sonnet-20240620-v1:0 |   gemini-1.5-pro-002 |   gpt-4o-2024-08-06_o1-mini-2024-09-12 |   gpt-4o-2024-08-06_anthropic.claude-3-5-sonnet-20240620-v1:0 |   gpt-4o-2024-08-06_gemini-1.5-pro-002 |   o1-mini-2024-09-12_anthropic.claude-3-5-sonnet-20240620-v1:0 |   o1-mini-2024-09-12_gemini-1.5-pro-002 |   anthropic.claude-3-5-sonnet-20240620-v1:0_gemini-1.5-pro-002 |   gpt-4o-2024-08-06_o1-mini-2024-09-12_anthropic.claude-3-5-sonnet-20240620-v1:0 |   gpt-4o-2024-08-06_o1-mini-2024-09-12_gemini-1.5-pro-002 |   gpt-4o-2024-08-06_anthropic.claude-3-5-sonnet-20240620-v1:0_gemini-1.5-pro-002 |   o1-mini-2024-09-12_anthropic.claude-3-5-sonnet-20240620-v1:0_gemini-1.5-pro-002 |
|:-------------------------------|----------:|--------------------:|---------------------:|--------------------------------------------:|---------------------:|---------------------------------------:|--------------------------------------------------------------:|---------------------------------------:|---------------------------------------------------------------:|----------------------------------------:|---------------------------------------------------------------:|---------------------------------------------------------------------------------:|----------------------------------------------------------:|---------------------------------------------------------------------------------:|----------------------------------------------------------------------------------:|
| Roleplay Adherence             |  0.632473 |            0.473109 |             0.460139 |                                    0.2904   |             0.539547 |                               0.604012 |                                                      0.391803 |                               0.619375 |                                                       0.479238 |                                0.629825 |                                                       0.464058 |                                                                         0.579314 |                                                  0.684168 |                                                                         0.522206 |                                                                          0.578193 |
| Consistency                    |  0.52039  |            0.57613  |             0.500888 |                                    0.194717 |             0.446015 |                               0.640674 |                                                      0.411669 |                               0.566035 |                                                       0.406102 |                                0.491425 |                                                       0.286646 |                                                                         0.553909 |                                                  0.613499 |                                                                         0.434632 |                                                                          0.389918 |
| Contextual Understanding       |  0.526451 |            0.416179 |             0.524996 |                                    0.309361 |             0.483984 |                               0.555817 |                                                      0.392728 |                               0.497801 |                                                       0.458816 |                                0.56317  |                                                       0.403141 |                                                                         0.50652  |                                                  0.585895 |                                                                         0.450092 |                                                                          0.495829 |
| Expressiveness                 |  0.560001 |            0.391092 |             0.477326 |                                    0.420209 |             0.470158 |                               0.519313 |                                                      0.494232 |                               0.502739 |                                                       0.499657 |                                0.516655 |                                                       0.474328 |                                                                         0.549841 |                                                  0.555042 |                                                                         0.520735 |                                                                          0.521352 |
| Creativity                     |  0.429796 |            0.347042 |             0.294125 |                                    0.396245 |             0.462311 |                               0.374478 |                                                      0.420484 |                               0.427139 |                                                       0.38644  |                                0.383614 |                                                       0.422334 |                                                                         0.406254 |                                                  0.408472 |                                                                         0.441738 |                                                                          0.400932 |
| Naturalness of Japanese        |  0.554593 |            0.48394  |             0.565689 |                                    0.386479 |             0.548344 |                               0.559631 |                                                      0.494055 |                               0.544961 |                                                       0.510059 |                                0.563773 |                                                       0.515218 |                                                                         0.540617 |                                                  0.566311 |                                                                         0.544976 |                                                                          0.543812 |
| Enjoyment of the Dialogue      |  0.504074 |            0.199664 |             0.437892 |                                    0.48092  |             0.442609 |                               0.398852 |                                                      0.436683 |                               0.375542 |                                                       0.525387 |                                0.474757 |                                                       0.497515 |                                                                         0.506834 |                                                  0.442266 |                                                                         0.462983 |                                                                          0.52382  |
| Appropriateness of Turn-Taking |  0.617291 |            0.530897 |             0.288324 |                                    0.487605 |             0.360864 |                               0.484505 |                                                      0.603742 |                               0.5729   |                                                       0.450075 |                                0.384404 |                                                       0.491124 |                                                                         0.576536 |                                                  0.564496 |                                                                         0.635467 |                                                                          0.482371 |
| Average Score                  |  0.601041 |            0.426126 |             0.462953 |                                    0.427334 |             0.55354  |                               0.547215 |                                                      0.507034 |                               0.563607 |                                                       0.502963 |                                0.560185 |                                                       0.517369 |                                                                         0.566939 |                                                  0.598594 |                                                                         0.553735 |                                                                          0.549003 |

生の評価結果は[こちら](evaluations)からご確認ください。

## 実行方法

1. ライブラリのインストール

   ```bash
   git clone https://github.com/Aratako/Japanese-RP-Bench.git
   cd Japanese-RP-Bench

   pip install -e .
   ```

2. YAML設定ファイルの作成

   [設定ファイルの例](configs/eval_config.yaml)を参考に、ベンチマーク用の設定ファイルを作成してください。詳細は[設定ファイルの詳細](#設定ファイルの詳細)の項目を参照してください。

3. ベンチマークの実行

   以下のようなコマンドで設定ファイルを指定してベンチマークを実行します。

   ```bash
   japanese-rp-bench --config ./configs/eval_config.yaml
   ```

## 設定ファイルの詳細

設定ファイルは以下のような構成になっています。

```yaml
dataset_repo: "Aratako/Japanese-RP-Bench-testdata-SFW"
dataset_split: "train"
target_model_name: "gpt-4o-mini-2024-07-18"
target_inference_method: "openai_api"
user_model_name: "anthropic.claude-3-5-sonnet-20240620-v1:0"
user_inference_method: "aws_anthropic_api"
judge_model_names:
  - "gpt-4o-2024-08-06"
  - "o1-mini-2024-09-12"
  - "anthropic.claude-3-5-sonnet-20240620-v1:0"
  - "gemini-1.5-pro-002"
judge_inference_methods:
  - "openai_api"
  - "openai_api"
  - "aws_anthropic_api"
  - "google_api"
evaluation_prompt_file: "./prompts/eval_prompt_SFW.txt"
max_turns: 10
cache_dir: "./.cache"
tensor_parallel_size: 1
```

各変数の説明は以下の通りです。

- `dataset_repo`: ロールプレイ設定等がまとまっている評価用データセットへのパス（別のデータで評価したい場合に変更してください）
- `dataset_split`: 上記データセットで処理対象としたいsplitの指定
- `target_model_name`: 評価対象のモデル名（huggingface repositryやAPIリクエスト時のモデル名等を指定してください）
- `target_inference_method`: 評価対象モデルの推論方法
- `user_model_name`: ユーザー側の役割を行わせるモデル名（Claude 3.5 Sonnetを推奨します）
- `user_inference_method`: ユーザー役モデルの推論方法
- `judge_model_names`: 評価を行うモデル名（複数指定可）
- `judge_inference_methods`: 評価モデルの推論方法
- `evaluation_prompt_file`: 評価プロンプトを記載したファイルへのパス
- `max_turns`: 各対話で何ターンまで対話を生成するか（ユーザーとアシスタントの1つのペアで1ターン）
- `cache_dir`: モデルやデータセットのキャッシュを保存するディレクトリ
- `tensor_parallel_size`: vLLMを使う場合の`tensor_parallel_size`（vLLMを使う場合のみ有効）

なお、`inference_method`は以下のものを指定できます。場合によってはAPIキーの設定が必要になります。

- `openai_api`: OpenAIのAPIを利用する場合。APIキーを`OPENAI_API_KEY`の環境変数に設定してください。
- `openai_compatible_api`: OpenAI API互換のAPIを利用する場合（DeepInfra、vLLM API Serverなど）。APIキーを`OPENAI_COMPATIBLE_API_KEY`に、APIのbase urlを`OPENAI_COMPATIBLE_API_URL`に設定してください。
- `anthropic_api`: AnthropicのAPIを利用する場合。APIキーを`ANTHROPIC_API_KEY`の環境変数に設定してください。
- `aws_anthropic_api`: Amazon BedrockのANTHROPICのAPIを利用する場合。AWSのアクセスキーを`AWS_ACCESS_KEY`に、シークレットキーを`AWS_SECRET_KEY`に設定してください。
- `cohere_api`: CohereのAPIを利用する場合。APIキーを`COHERE_API_KEY`の環境変数に設定してください。
- `google_api`: GoogleのAPIを利用する場合。APIキーを`GOOGLE_API_KEY`の環境変数に設定してください。
- `mistralai_api`: MistralのAPIを利用する場合。APIキーを`MISTRAL_API_KEY`の環境変数に設定してください。
- `vllm`: vLLMを利用してローカルでモデルを推論する場合。
- `transformers`: transformersを利用してローカルでモデルを推論する場合。

## ライセンス

このプロジェクトは[MITライセンス](LICENSE)の下で公開されています。
