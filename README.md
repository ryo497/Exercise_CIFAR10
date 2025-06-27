# CIFAR-10 CNN実験フレームワーク

CIFAR-10データセットを使用したCNN（畳み込みニューラルネットワーク）の包括的な実験フレームワークです。様々なアーキテクチャとハイパーパラメータの組み合わせを自動的に実験し、最適なモデルを見つけることができます。

## 特徴

- **柔軟なCNNアーキテクチャ**: 層数、フィルタ数、カーネルサイズなどを自由に設定可能
- **多様な実験設定**: 20種類以上の事前定義された実験パターン
- **自動実験実行**: 複数の実験を連続して実行し、結果を自動保存
- **包括的な分析**: 実験結果の可視化と詳細なレポート生成
- **高度な機能**: ResNet風アーキテクチャ、混合精度学習、各種データ拡張など

## ファイル構成

```
ml-test-CIFAR10/
├── main.py                 # メインのCNN実験スクリプト
├── run_experiments.py      # 自動実験実行スクリプト
├── analyze_results.py      # 結果分析・可視化スクリプト
├── README.md              # このファイル
└── outputs/               # 実験結果の保存先
    ├── experiment_summary.csv      # 全実験のサマリー
    ├── experiment_summary.json     # JSON形式のサマリー
    ├── hyperparameter_analysis.png # ハイパーパラメータ分析図
    ├── top_models_comparison.png   # 上位モデルの比較図
    ├── architecture_comparison.png # アーキテクチャ比較図
    ├── experiment_report.md        # 詳細レポート
    └── [timestamp]_[name]/        # 各実験の詳細結果
        ├── config.json            # 実験設定
        ├── experiment_results.json # 詳細結果
        ├── training_curves.png    # 学習曲線
        ├── training_log.csv       # エポック毎のログ
        ├── best_model.h5          # 最良モデル
        └── final_model.h5         # 最終モデル
```

## 使い方

### 1. 単一実験の実行

```bash
# 基本的な実行
python main.py

# カスタム設定での実行
python main.py --conv_layers 4 --filters 32 64 128 256 --epochs 50 --batch_size 64

# 高度な設定
python main.py --architecture resnet --augmentation_strength strong --lr_schedule cosine --use_mixed_precision
```

### 2. 自動実験の実行

```bash
# 事前定義された20種類の実験を自動実行
python run_experiments.py
```

### 3. 結果の分析

```bash
# 実験結果の分析と可視化
python analyze_results.py
```

## 主要なハイパーパラメータ

### アーキテクチャ関連
- `--architecture`: モデルアーキテクチャ（standard, resnet）
- `--conv_layers`: 畳み込み層の数（2-5）
- `--filters`: 各層のフィルタ数（例: 32 64 128）
- `--kernel_size`: カーネルサイズ（3 or 5）
- `--pooling_type`: プーリングタイプ（max, average）
- `--pooling_size`: プーリングサイズ（通常2）
- `--use_global_avg_pool`: Global Average Poolingを使用

### 学習関連
- `--epochs`: エポック数
- `--batch_size`: バッチサイズ（32, 64, 128, 256）
- `--optimizer`: 最適化手法（adam, sgd, rmsprop, adamw）
- `--learning_rate`: 初期学習率
- `--lr_schedule`: 学習率スケジュール（none, exponential, cosine, step, warmup_cosine）

### 正則化関連
- `--dropout_rate`: ドロップアウト率（0.0-0.5）
- `--use_batch_norm`: バッチ正規化を使用
- `--augmentation_strength`: データ拡張強度（none, weak, medium, strong）

### その他
- `--activation`: 活性化関数（relu, elu, selu, swish）
- `--dense_units`: 全結合層のユニット数（例: 128 64）
- `--use_mixed_precision`: 混合精度学習を使用
- `--experiment_name`: 実験名

## 実験パターン

`run_experiments.py`には以下の実験パターンが含まれています：

1. **ベースライン**: シンプルな3層CNN
2. **深いネットワーク**: 5層CNN
3. **幅広ネットワーク**: 多くのフィルタ
4. **VGG風**: 3x3カーネルの積み重ね
5. **大きなカーネル**: 5x5カーネル使用
6. **Average Pooling**: 平均プーリング
7. **Global Average Pooling**: GAP使用
8. **データ拡張**: 強/中/弱の比較
9. **最適化手法**: Adam, SGD, RMSprop, AdamW
10. **活性化関数**: ReLU, ELU比較
11. **学習率スケジュール**: Cosine, Step, Warmup
12. **バッチサイズ**: 32, 64, 128, 256の比較
13. **ResNet風**: スキップ接続あり
14. **混合精度学習**: float16使用
15. **最適化候補**: 総合的な最適設定

## 分析機能

`analyze_results.py`は以下の分析を提供：

- **ハイパーパラメータの影響分析**: 各パラメータと精度の関係
- **学習曲線の比較**: 上位モデルの学習過程
- **アーキテクチャ比較**: BatchNorm、GAP、Dropoutなどの効果
- **詳細レポート**: Markdown形式の包括的なレポート

## 必要なライブラリ

```bash
pip install tensorflow numpy pandas matplotlib seaborn
```

## 推奨事項

1. **GPU使用推奨**: CIFAR-10の学習にはGPUの使用を強く推奨します
2. **メモリ管理**: 大きなバッチサイズや深いモデルではメモリ不足に注意
3. **実験時間**: 全20実験の実行には数時間かかる可能性があります
4. **結果の保存**: 実験結果は自動的に`outputs/`ディレクトリに保存されます

## トラブルシューティング

- **メモリ不足**: バッチサイズを小さくするか、モデルサイズを削減
- **学習が収束しない**: 学習率を調整するか、正則化を追加
- **過学習**: ドロップアウト率を上げるか、データ拡張を強化

## ライセンス

このプロジェクトは教育目的で作成されています。