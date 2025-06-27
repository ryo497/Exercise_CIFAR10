#!/usr/bin/env python3
"""
CIFAR-10 CNN実験自動実行スクリプト
様々なCNNアーキテクチャとハイパーパラメータの組み合わせで実験を実行
"""

import subprocess
import json
from pathlib import Path
import pandas as pd
import time
import sys
import signal


def run_experiment(params):
    """単一の実験を実行"""
    cmd = ["python", "-u", "main.py"]  # -u フラグでバッファリング無効化
    for key, value in params.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        elif isinstance(value, list):
            cmd.append(f"--{key}")
            for v in value:
                cmd.append(str(v))
        else:
            cmd.append(f"--{key}")
            cmd.append(str(value))
    
    print(f"\n実行中: {' '.join(cmd)}")
    print("進行状況をリアルタイム表示中...")
    print("-" * 60)
    
    try:
        # リアルタイム出力のためにPopen使用
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=0  # バッファサイズを0に設定
        )
        
        # 出力をリアルタイムで表示
        output_lines = []
        for line in iter(process.stdout.readline, ''):
            if line:
                print(line.rstrip())  # リアルタイム表示
                sys.stdout.flush()  # 強制的にフラッシュ
                output_lines.append(line)
        
        return_code = process.wait()
        
        if return_code != 0:
            print(f"\nエラー: 実験が失敗しました (終了コード: {return_code})")
            return None
        
        print(f"\n{'-'*60}")
        print("実験完了!")
        print(f"{'-'*60}")
        
        return ''.join(output_lines)
        
    except KeyboardInterrupt:
        print("\n実験を中断しています...")
        process.terminate()
        return None
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        return None


def signal_handler(sig, frame):
    """Ctrl+C による中断処理"""
    print(f"\n\n🛑 実験を中断しています...")
    print("現在までの結果は保存されています。")
    sys.exit(0)


def main():
    # シグナルハンドラーを設定
    signal.signal(signal.SIGINT, signal_handler)
    
    # 実験パラメータの定義
    experiments = [
        # 1. ベースライン（シンプルなCNN）
        {
            "experiment_name": "baseline_simple",
            "epochs": 30,
            "conv_layers": 3,
            "filters": [32, 64, 128],
            "kernel_size": 3,
            "pooling_type": "max",
            "pooling_size": 2,
            "activation": "relu",
            "optimizer": "adam",
            "learning_rate": 0.001,
            "batch_size": 128,
            "dense_units": [128]
        },
        
        # 2. より深いネットワーク（5層）
        {
            "experiment_name": "deep_5layers",
            "epochs": 40,
            "conv_layers": 5,
            "filters": [32, 64, 128, 256, 512],
            "kernel_size": 3,
            "pooling_type": "max",
            "pooling_size": 2,
            "activation": "relu",
            "optimizer": "adam",
            "learning_rate": 0.001,
            "batch_size": 64,
            "dropout_rate": 0.3,
            "use_batch_norm": True,
            "dense_units": [256, 128]
        },
        
        # 3. 幅広ネットワーク（フィルタ数多め）
        {
            "experiment_name": "wide_network",
            "epochs": 35,
            "conv_layers": 3,
            "filters": [64, 128, 256],
            "kernel_size": 3,
            "pooling_type": "max",
            "pooling_size": 2,
            "activation": "relu",
            "optimizer": "adam",
            "learning_rate": 0.001,
            "batch_size": 64,
            "dropout_rate": 0.2,
            "dense_units": [256]
        },
        
        # 4. VGG風（3x3カーネルの積み重ね）
        {
            "experiment_name": "vgg_style",
            "epochs": 40,
            "conv_layers": 4,
            "filters": [64, 64, 128, 128],
            "kernel_size": 3,
            "pooling_type": "max",
            "pooling_size": 2,
            "activation": "relu",
            "optimizer": "adam",
            "learning_rate": 0.001,
            "batch_size": 128,
            "use_batch_norm": True,
            "dropout_rate": 0.25,
            "dense_units": [256, 128]
        },
        
        # 5. 大きなカーネル（5x5）
        {
            "experiment_name": "large_kernel",
            "epochs": 35,
            "conv_layers": 3,
            "filters": [32, 64, 128],
            "kernel_size": 5,
            "pooling_type": "max",
            "pooling_size": 2,
            "activation": "relu",
            "optimizer": "adam",
            "learning_rate": 0.001,
            "batch_size": 128,
            "dropout_rate": 0.2,
            "dense_units": [128]
        },
        
        # 6. Average Pooling使用
        {
            "experiment_name": "avg_pooling",
            "epochs": 35,
            "conv_layers": 4,
            "filters": [32, 64, 128, 256],
            "kernel_size": 3,
            "pooling_type": "average",
            "pooling_size": 2,
            "activation": "relu",
            "optimizer": "adam",
            "learning_rate": 0.001,
            "batch_size": 128,
            "use_batch_norm": True,
            "dense_units": [128]
        },
        
        # 7. Global Average Pooling使用
        {
            "experiment_name": "global_avg_pool",
            "epochs": 35,
            "conv_layers": 4,
            "filters": [32, 64, 128, 256],
            "kernel_size": 3,
            "pooling_type": "max",
            "pooling_size": 2,
            "activation": "relu",
            "optimizer": "adam",
            "learning_rate": 0.001,
            "batch_size": 128,
            "use_global_avg_pool": True,
            "use_batch_norm": True,
            "dense_units": [128]
        },
        
        # 8. 強いデータ拡張
        {
            "experiment_name": "strong_augmentation",
            "epochs": 50,
            "conv_layers": 4,
            "filters": [32, 64, 128, 256],
            "kernel_size": 3,
            "pooling_type": "max",
            "pooling_size": 2,
            "activation": "relu",
            "optimizer": "adam",
            "learning_rate": 0.001,
            "batch_size": 128,
            "augmentation_strength": "strong",
            "dropout_rate": 0.3,
            "use_batch_norm": True,
            "dense_units": [256, 128]
        },
        
        # 9. 中程度のデータ拡張
        {
            "experiment_name": "medium_augmentation",
            "epochs": 40,
            "conv_layers": 4,
            "filters": [32, 64, 128, 256],
            "kernel_size": 3,
            "pooling_type": "max",
            "pooling_size": 2,
            "activation": "relu",
            "optimizer": "adam",
            "learning_rate": 0.001,
            "batch_size": 128,
            "augmentation_strength": "medium",
            "dropout_rate": 0.2,
            "dense_units": [128]
        },
        
        # 10. SGD with momentum
        {
            "experiment_name": "sgd_momentum",
            "epochs": 60,
            "conv_layers": 4,
            "filters": [32, 64, 128, 256],
            "kernel_size": 3,
            "pooling_type": "max",
            "pooling_size": 2,
            "activation": "relu",
            "optimizer": "sgd",
            "learning_rate": 0.01,
            "batch_size": 64,
            "lr_schedule": "step",
            "use_batch_norm": True,
            "dropout_rate": 0.2,
            "dense_units": [128]
        },
        
        # 11. RMSprop optimizer
        {
            "experiment_name": "rmsprop",
            "epochs": 40,
            "conv_layers": 4,
            "filters": [32, 64, 128, 256],
            "kernel_size": 3,
            "pooling_type": "max",
            "pooling_size": 2,
            "activation": "relu",
            "optimizer": "rmsprop",
            "learning_rate": 0.001,
            "batch_size": 128,
            "use_batch_norm": True,
            "dense_units": [128]
        },
        
        # 12. AdamW (with weight decay)
        {
            "experiment_name": "adamw",
            "epochs": 40,
            "conv_layers": 4,
            "filters": [32, 64, 128, 256],
            "kernel_size": 3,
            "pooling_type": "max",
            "pooling_size": 2,
            "activation": "relu",
            "optimizer": "adamw",
            "learning_rate": 0.001,
            "batch_size": 128,
            "use_batch_norm": True,
            "dropout_rate": 0.2,
            "dense_units": [128]
        },
        
        # 13. ELU activation
        {
            "experiment_name": "elu_activation",
            "epochs": 35,
            "conv_layers": 4,
            "filters": [32, 64, 128, 256],
            "kernel_size": 3,
            "pooling_type": "max",
            "pooling_size": 2,
            "activation": "elu",
            "optimizer": "adam",
            "learning_rate": 0.001,
            "batch_size": 128,
            "dropout_rate": 0.2,
            "dense_units": [128]
        },
        
        # 14. Cosine learning rate schedule
        {
            "experiment_name": "cosine_schedule",
            "epochs": 50,
            "conv_layers": 4,
            "filters": [32, 64, 128, 256],
            "kernel_size": 3,
            "pooling_type": "max",
            "pooling_size": 2,
            "activation": "relu",
            "optimizer": "adam",
            "learning_rate": 0.003,
            "batch_size": 128,
            "lr_schedule": "cosine",
            "use_batch_norm": True,
            "dropout_rate": 0.2,
            "dense_units": [128]
        },
        
        # 15. Warmup + Cosine schedule
        {
            "experiment_name": "warmup_cosine",
            "epochs": 50,
            "conv_layers": 4,
            "filters": [32, 64, 128, 256],
            "kernel_size": 3,
            "pooling_type": "max",
            "pooling_size": 2,
            "activation": "relu",
            "optimizer": "adam",
            "learning_rate": 0.003,
            "batch_size": 128,
            "lr_schedule": "warmup_cosine",
            "use_batch_norm": True,
            "augmentation_strength": "medium",
            "dropout_rate": 0.25,
            "dense_units": [256, 128]
        },
        
        # 16. 小バッチサイズ
        {
            "experiment_name": "small_batch",
            "epochs": 40,
            "conv_layers": 4,
            "filters": [32, 64, 128, 256],
            "kernel_size": 3,
            "pooling_type": "max",
            "pooling_size": 2,
            "activation": "relu",
            "optimizer": "adam",
            "learning_rate": 0.001,
            "batch_size": 32,
            "use_batch_norm": True,
            "dropout_rate": 0.2,
            "dense_units": [128]
        },
        
        # 17. 大バッチサイズ
        {
            "experiment_name": "large_batch",
            "epochs": 40,
            "conv_layers": 4,
            "filters": [32, 64, 128, 256],
            "kernel_size": 3,
            "pooling_type": "max",
            "pooling_size": 2,
            "activation": "relu",
            "optimizer": "adam",
            "learning_rate": 0.002,
            "batch_size": 256,
            "use_batch_norm": True,
            "dense_units": [128]
        },
        
        # 18. ResNet風アーキテクチャ
        {
            "experiment_name": "resnet_style",
            "architecture": "resnet",
            "epochs": 50,
            "conv_layers": 4,  # blocks数として使用
            "filters": [64],
            "optimizer": "adam",
            "learning_rate": 0.001,
            "batch_size": 128,
            "dropout_rate": 0.2,
            "lr_schedule": "cosine",
            "augmentation_strength": "medium"
        },
        
        # 19. 混合精度学習
        {
            "experiment_name": "mixed_precision",
            "epochs": 40,
            "conv_layers": 4,
            "filters": [32, 64, 128, 256],
            "kernel_size": 3,
            "pooling_type": "max",
            "pooling_size": 2,
            "activation": "relu",
            "optimizer": "adam",
            "learning_rate": 0.001,
            "batch_size": 128,
            "use_mixed_precision": True,
            "use_batch_norm": True,
            "dropout_rate": 0.2,
            "dense_units": [128]
        },
        
        # 20. 最適化候補（総合的な設定）
        {
            "experiment_name": "optimized_final",
            "epochs": 60,
            "conv_layers": 4,
            "filters": [64, 128, 256, 512],
            "kernel_size": 3,
            "pooling_type": "max",
            "pooling_size": 2,
            "activation": "relu",
            "optimizer": "adam",
            "learning_rate": 0.002,
            "batch_size": 64,
            "lr_schedule": "warmup_cosine",
            "use_batch_norm": True,
            "use_global_avg_pool": True,
            "augmentation_strength": "medium",
            "dropout_rate": 0.25,
            "dense_units": [256, 128],
            "use_mixed_precision": True
        }
    ]
    
    print(f"実行予定の実験数: {len(experiments)}")
    print("注意: CIFAR-10の学習には時間がかかります。全実験完了まで数時間かかる可能性があります。")
    
    # 各実験を実行
    successful_experiments = 0
    failed_experiments = []
    start_time = time.time()
    
    for i, params in enumerate(experiments, 1):
        print(f"\n{'='*80}")
        print(f"実験 {i}/{len(experiments)}: {params['experiment_name']}")
        print(f"{'='*80}")
        
        # 実験の主要パラメータを表示
        print("主要設定:")
        if 'architecture' in params:
            print(f"  アーキテクチャ: {params['architecture']}")
        print(f"  畳み込み層数: {params.get('conv_layers', 'N/A')}")
        print(f"  フィルタ数: {params.get('filters', 'N/A')}")
        print(f"  エポック数: {params.get('epochs', 'N/A')}")
        print(f"  バッチサイズ: {params.get('batch_size', 'N/A')}")
        if params.get('augmentation_strength', 'none') != 'none':
            print(f"  データ拡張: {params['augmentation_strength']}")
        print()
        
        experiment_start = time.time()
        try:
            output = run_experiment(params)
            if output:
                successful_experiments += 1
                experiment_time = time.time() - experiment_start
                print(f"\n✅ 実験 {i} 成功 (実行時間: {experiment_time/60:.1f}分)")
            else:
                failed_experiments.append(params['experiment_name'])
                print(f"\n❌ 実験 {i} 失敗")
        except Exception as e:
            print(f"\n❌ エラーが発生しました: {e}")
            failed_experiments.append(params['experiment_name'])
        
        # 経過時間と残り予想時間を表示
        elapsed = time.time() - start_time
        if i > 0:
            avg_time_per_exp = elapsed / i
            remaining_time = avg_time_per_exp * (len(experiments) - i)
            print(f"\n📊 進行状況: {i}/{len(experiments)} 完了")
            print(f"⏱️  経過時間: {elapsed/60:.1f}分")
            print(f"⏳ 残り予想時間: {remaining_time/60:.1f}分")
        
        # GPU/メモリ解放のため短い待機
        time.sleep(3)
    
    # 結果の集計
    print(f"\n{'='*80}")
    print("全実験完了！結果を集計中...")
    print(f"{'='*80}")
    
    print(f"\n成功した実験: {successful_experiments}/{len(experiments)}")
    if failed_experiments:
        print(f"失敗した実験: {', '.join(failed_experiments)}")
    
    # CSVファイルを読み込んで結果を表示
    summary_file = Path("outputs/experiment_summary.csv")
    if summary_file.exists():
        df = pd.read_csv(summary_file)
        
        # テスト精度でソート
        df_sorted = df.sort_values("test_accuracy", ascending=False)
        
        print("\n=== Top 10 実験結果（テスト精度順）===")
        cols_to_show = ["experiment_name", "test_accuracy", "conv_layers", 
                       "filters", "activation", "optimizer", "batch_size", 
                       "augmentation_strength", "training_time_seconds"]
        
        # 表示する列を調整（存在しない列を除外）
        cols_to_show = [col for col in cols_to_show if col in df.columns]
        
        print(df_sorted[cols_to_show].head(10).to_string(index=False))
        
        # 最良の結果を詳細表示
        if len(df_sorted) > 0:
            best = df_sorted.iloc[0]
            print(f"\n=== 最良の結果 ===")
            print(f"実験名: {best['experiment_name']}")
            print(f"テスト精度: {best['test_accuracy']:.4f}")
            print(f"設定:")
            print(f"  - アーキテクチャ: {best.get('architecture', 'standard')}")
            print(f"  - 畳み込み層数: {best['conv_layers']}")
            print(f"  - フィルタ数: {best['filters']}")
            print(f"  - カーネルサイズ: {best.get('kernel_size', 'N/A')}")
            print(f"  - プーリング: {best.get('pooling_type', 'N/A')}")
            print(f"  - 活性化関数: {best['activation']}")
            print(f"  - 最適化手法: {best['optimizer']}")
            print(f"  - 学習率: {best['learning_rate']}")
            print(f"  - バッチサイズ: {best['batch_size']}")
            print(f"  - データ拡張: {best.get('augmentation_strength', 'none')}")
            print(f"  - エポック数: {best['epochs']}")
            print(f"  - 学習時間: {best['training_time_seconds']:.2f}秒")
            
            # 分析用サマリーの生成
            analysis_summary = {
                "total_experiments": len(df),
                "successful_experiments": successful_experiments,
                "best_accuracy": float(best['test_accuracy']),
                "average_accuracy": float(df['test_accuracy'].mean()),
                "std_accuracy": float(df['test_accuracy'].std()),
                "best_configuration": {
                    "name": best['experiment_name'],
                    "accuracy": float(best['test_accuracy']),
                    "architecture": best.get('architecture', 'standard'),
                    "conv_layers": int(best['conv_layers']),
                    "filters": best['filters'],
                    "optimizer": best['optimizer'],
                    "learning_rate": float(best['learning_rate']),
                    "batch_size": int(best['batch_size']),
                    "augmentation": best.get('augmentation_strength', 'none')
                }
            }
            
            with open("outputs/experiment_summary.json", "w") as f:
                json.dump(analysis_summary, f, indent=2, ensure_ascii=False)
            
            print("\n実験サマリーを outputs/experiment_summary.json に保存しました")


if __name__ == "__main__":
    main()