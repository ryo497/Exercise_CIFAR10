#!/usr/bin/env python3
"""
CIFAR-10 CNN実験結果の分析・可視化スクリプト
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import numpy as np
from datetime import datetime


def load_experiment_history(experiment_dir):
    """実験ディレクトリから履歴データを読み込む"""
    results_file = experiment_dir / "experiment_results.json"
    if results_file.exists():
        with open(results_file, "r") as f:
            return json.load(f)
    return None


def plot_hyperparameter_impact():
    """ハイパーパラメータの影響を可視化"""
    summary_file = Path("outputs/experiment_summary.csv")
    if not summary_file.exists():
        print("experiment_summary.csv が見つかりません")
        return
    
    df = pd.read_csv(summary_file)
    
    # フィルタ数を文字列として扱う（リスト形式のため）
    df['filters_str'] = df['filters'].astype(str)
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    # 1. 畳み込み層数 vs 精度
    ax = axes[0]
    df_grouped = df.groupby('conv_layers')['test_accuracy'].agg(['mean', 'std', 'count'])
    ax.bar(df_grouped.index, df_grouped['mean'], 
           yerr=df_grouped['std'], capsize=5, alpha=0.7)
    ax.set_xlabel('Number of Conv Layers')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Impact of Number of Conv Layers')
    # サンプル数を表示
    for i, (idx, row) in enumerate(df_grouped.iterrows()):
        ax.text(i, row['mean'] + row['std'] + 0.01, f'n={row["count"]}', 
                ha='center', va='bottom')
    
    # 2. アーキテクチャ vs 精度
    ax = axes[1]
    if 'architecture' in df.columns:
        df_grouped = df.groupby('architecture')['test_accuracy'].agg(['mean', 'std', 'count'])
        ax.bar(df_grouped.index, df_grouped['mean'], 
               yerr=df_grouped['std'], capsize=5, alpha=0.7)
        ax.set_xlabel('Architecture')
        ax.set_ylabel('Test Accuracy')
        ax.set_title('Impact of Architecture')
    
    # 3. 活性化関数 vs 精度
    ax = axes[2]
    df_grouped = df.groupby('activation')['test_accuracy'].agg(['mean', 'std', 'count'])
    ax.bar(df_grouped.index, df_grouped['mean'], 
           yerr=df_grouped['std'], capsize=5, alpha=0.7)
    ax.set_xlabel('Activation Function')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Impact of Activation Function')
    
    # 4. 最適化手法 vs 精度
    ax = axes[3]
    df_grouped = df.groupby('optimizer')['test_accuracy'].agg(['mean', 'std', 'count'])
    ax.bar(df_grouped.index, df_grouped['mean'], 
           yerr=df_grouped['std'], capsize=5, alpha=0.7)
    ax.set_xlabel('Optimizer')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Impact of Optimizer')
    
    # 5. バッチサイズ vs 精度
    ax = axes[4]
    df_grouped = df.groupby('batch_size')['test_accuracy'].agg(['mean', 'std', 'count'])
    ax.bar(df_grouped.index.astype(str), df_grouped['mean'], 
           yerr=df_grouped['std'], capsize=5, alpha=0.7)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Impact of Batch Size')
    
    # 6. データ拡張 vs 精度
    ax = axes[5]
    if 'augmentation_strength' in df.columns:
        df_grouped = df.groupby('augmentation_strength')['test_accuracy'].agg(['mean', 'std', 'count'])
        ax.bar(df_grouped.index, df_grouped['mean'], 
               yerr=df_grouped['std'], capsize=5, alpha=0.7)
        ax.set_xlabel('Data Augmentation Strength')
        ax.set_ylabel('Test Accuracy')
        ax.set_title('Impact of Data Augmentation')
    
    # 7. 学習率 vs 精度（散布図）
    ax = axes[6]
    ax.scatter(df['learning_rate'], df['test_accuracy'], alpha=0.6, s=50)
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Impact of Learning Rate')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # 8. プーリングタイプ vs 精度
    ax = axes[7]
    if 'pooling_type' in df.columns:
        df_grouped = df.groupby('pooling_type')['test_accuracy'].agg(['mean', 'std', 'count'])
        ax.bar(df_grouped.index, df_grouped['mean'], 
               yerr=df_grouped['std'], capsize=5, alpha=0.7)
        ax.set_xlabel('Pooling Type')
        ax.set_ylabel('Test Accuracy')
        ax.set_title('Impact of Pooling Type')
    
    # 9. 学習時間 vs 精度
    ax = axes[8]
    ax.scatter(df['training_time_seconds']/60, df['test_accuracy'], alpha=0.6, s=50)
    ax.set_xlabel('Training Time (minutes)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Training Time vs Accuracy Trade-off')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/hyperparameter_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("ハイパーパラメータ分析を outputs/hyperparameter_analysis.png に保存しました")


def plot_training_curves_comparison():
    """上位モデルの学習曲線を比較"""
    summary_file = Path("outputs/experiment_summary.csv")
    if not summary_file.exists():
        return
    
    df = pd.read_csv(summary_file)
    df_sorted = df.sort_values('test_accuracy', ascending=False)
    
    # 上位5つの実験の学習曲線を取得
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    output_dirs = list(Path("outputs").glob("*_*"))
    output_dirs = [d for d in output_dirs if d.is_dir() and not d.name.endswith('.png')]
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # 上位10個の実験を処理
    processed = 0
    for i, (_, row) in enumerate(df_sorted.head(10).iterrows()):
        # 該当する実験ディレクトリを探す
        for output_dir in output_dirs:
            results = load_experiment_history(output_dir)
            if results and results['experiment_id'] == row['experiment_id']:
                label = f"{row['experiment_name']} ({row['test_accuracy']:.3f})"
                
                # 訓練精度
                ax1.plot(results['history']['train_accuracy'], 
                        label=label, color=colors[i], linewidth=2, alpha=0.8)
                
                # 検証精度
                ax2.plot(results['history']['val_accuracy'], 
                        label=label, color=colors[i], linewidth=2, alpha=0.8)
                
                # 訓練損失
                ax3.plot(results['history']['train_loss'], 
                        label=label, color=colors[i], linewidth=2, alpha=0.8)
                
                # 検証損失
                ax4.plot(results['history']['val_loss'], 
                        label=label, color=colors[i], linewidth=2, alpha=0.8)
                
                processed += 1
                break
        
        if processed >= 5:  # 最大5つまで表示
            break
    
    # グラフの設定
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Accuracy')
    ax1.set_title('Training Accuracy of Top Models')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.4, 1.0)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_title('Validation Accuracy of Top Models')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.4, 1.0)
    
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Training Loss')
    ax3.set_title('Training Loss of Top Models')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Validation Loss')
    ax4.set_title('Validation Loss of Top Models')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/top_models_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("上位モデルの比較を outputs/top_models_comparison.png に保存しました")


def plot_architecture_comparison():
    """アーキテクチャごとの詳細比較"""
    summary_file = Path("outputs/experiment_summary.csv")
    if not summary_file.exists():
        return
    
    df = pd.read_csv(summary_file)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. バッチ正規化の効果
    ax = axes[0, 0]
    if 'use_batch_norm' in df.columns:
        df_bn = df.groupby('use_batch_norm')['test_accuracy'].agg(['mean', 'std', 'count'])
        df_bn.index = ['Without BN', 'With BN']
        ax.bar(df_bn.index, df_bn['mean'], yerr=df_bn['std'], capsize=5, alpha=0.7)
        ax.set_ylabel('Test Accuracy')
        ax.set_title('Effect of Batch Normalization')
        for i, (idx, row) in enumerate(df_bn.iterrows()):
            ax.text(i, row['mean'] + row['std'] + 0.01, f'n={row["count"]}', 
                    ha='center', va='bottom')
    
    # 2. Global Average Poolingの効果
    ax = axes[0, 1]
    if 'use_global_avg_pool' in df.columns:
        df_gap = df.groupby('use_global_avg_pool')['test_accuracy'].agg(['mean', 'std', 'count'])
        df_gap.index = ['Flatten', 'Global Avg Pool']
        ax.bar(df_gap.index, df_gap['mean'], yerr=df_gap['std'], capsize=5, alpha=0.7)
        ax.set_ylabel('Test Accuracy')
        ax.set_title('Effect of Global Average Pooling')
    
    # 3. ドロップアウト率の影響
    ax = axes[1, 0]
    df_dropout = df[df['dropout_rate'] > 0].copy()
    if len(df_dropout) > 0:
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        df_dropout['dropout_bin'] = pd.cut(df_dropout['dropout_rate'], bins=bins)
        df_grouped = df_dropout.groupby('dropout_bin')['test_accuracy'].agg(['mean', 'std', 'count'])
        
        x_labels = [f"{b.left:.1f}-{b.right:.1f}" for b in df_grouped.index]
        ax.bar(x_labels, df_grouped['mean'], yerr=df_grouped['std'], capsize=5, alpha=0.7)
        ax.set_xlabel('Dropout Rate')
        ax.set_ylabel('Test Accuracy')
        ax.set_title('Impact of Dropout Rate')
    
    # 4. 学習率スケジュールの効果
    ax = axes[1, 1]
    if 'lr_schedule' in df.columns:
        df_lr = df.groupby('lr_schedule')['test_accuracy'].agg(['mean', 'std', 'count'])
        ax.bar(df_lr.index, df_lr['mean'], yerr=df_lr['std'], capsize=5, alpha=0.7)
        ax.set_xlabel('Learning Rate Schedule')
        ax.set_ylabel('Test Accuracy')
        ax.set_title('Effect of Learning Rate Schedule')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('outputs/architecture_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("アーキテクチャ比較を outputs/architecture_comparison.png に保存しました")


def generate_report():
    """詳細なレポートを生成"""
    summary_file = Path("outputs/experiment_summary.csv")
    if not summary_file.exists():
        print("実験結果が見つかりません")
        return
    
    df = pd.read_csv(summary_file)
    
    report = []
    report.append("# CIFAR-10 CNN実験結果レポート")
    report.append(f"\n生成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n")
    
    report.append("## 実験概要")
    report.append(f"- 総実験数: {len(df)}")
    report.append(f"- 最高精度: {df['test_accuracy'].max():.4f}")
    report.append(f"- 平均精度: {df['test_accuracy'].mean():.4f} (標準偏差: {df['test_accuracy'].std():.4f})")
    report.append(f"- 中央値精度: {df['test_accuracy'].median():.4f}")
    
    if 'training_time_seconds' in df.columns:
        report.append(f"- 平均学習時間: {df['training_time_seconds'].mean()/60:.1f}分")
        report.append(f"- 最短学習時間: {df['training_time_seconds'].min()/60:.1f}分")
        report.append(f"- 最長学習時間: {df['training_time_seconds'].max()/60:.1f}分\n")
    
    # トップ5の詳細
    report.append("## トップ5モデルの詳細\n")
    df_sorted = df.sort_values('test_accuracy', ascending=False)
    
    for i, (_, row) in enumerate(df_sorted.head(5).iterrows(), 1):
        report.append(f"### {i}位: {row['experiment_name']}")
        report.append(f"- **テスト精度**: {row['test_accuracy']:.4f}")
        report.append(f"- **検証精度（最高）**: {row.get('best_val_accuracy', 'N/A')}")
        report.append(f"- **アーキテクチャ**: {row.get('architecture', 'standard')}")
        report.append(f"- **構造**: {row['conv_layers']}層CNN")
        report.append(f"- **フィルタ数**: {row['filters']}")
        
        if pd.notna(row.get('kernel_size')):
            report.append(f"- **カーネルサイズ**: {int(row['kernel_size'])}x{int(row['kernel_size'])}")
        if pd.notna(row.get('pooling_type')):
            report.append(f"- **プーリング**: {row['pooling_type']} (サイズ: {int(row.get('pooling_size', 2))})")
        
        report.append(f"- **活性化関数**: {row['activation']}")
        report.append(f"- **最適化手法**: {row['optimizer']} (学習率: {row['learning_rate']})")
        report.append(f"- **バッチサイズ**: {int(row['batch_size'])}")
        
        # 追加の設定
        additional = []
        if row.get('dropout_rate', 0) > 0:
            additional.append(f"Dropout: {row['dropout_rate']}")
        if row.get('use_batch_norm', False):
            additional.append("BatchNorm使用")
        if row.get('use_global_avg_pool', False):
            additional.append("Global Avg Pool使用")
        if row.get('augmentation_strength', 'none') != 'none':
            additional.append(f"データ拡張: {row['augmentation_strength']}")
        if row.get('lr_schedule', 'none') != 'none':
            additional.append(f"学習率スケジュール: {row['lr_schedule']}")
        if row.get('use_mixed_precision', False):
            additional.append("混合精度学習")
        
        if additional:
            report.append(f"- **追加設定**: {', '.join(additional)}")
        
        report.append(f"- **エポック数**: {int(row['epochs'])}")
        if 'training_time_seconds' in row:
            report.append(f"- **学習時間**: {row['training_time_seconds']/60:.1f}分")
        report.append("")
    
    # ハイパーパラメータの分析
    report.append("## ハイパーパラメータ分析\n")
    
    # 各パラメータの最適値
    report.append("### 最も効果的な設定（平均精度ベース）")
    
    categorical_params = ['conv_layers', 'activation', 'optimizer', 'pooling_type', 
                         'augmentation_strength', 'lr_schedule', 'architecture']
    
    for param in categorical_params:
        if param in df.columns:
            param_groups = df.groupby(param)['test_accuracy'].agg(['mean', 'count', 'std'])
            # サンプル数が3以上のものだけを考慮
            param_groups = param_groups[param_groups['count'] >= 2]
            if len(param_groups) > 0:
                best_value = param_groups['mean'].idxmax()
                best_score = param_groups.loc[best_value, 'mean']
                best_count = param_groups.loc[best_value, 'count']
                report.append(f"- **{param}**: {best_value} (平均精度: {best_score:.4f}, n={best_count})")
    
    # 特定の組み合わせの効果
    report.append("\n### 重要な発見")
    
    # データ拡張の効果
    if 'augmentation_strength' in df.columns:
        with_aug = df[df['augmentation_strength'] != 'none']['test_accuracy'].mean()
        without_aug = df[df['augmentation_strength'] == 'none']['test_accuracy'].mean()
        if not pd.isna(with_aug) and not pd.isna(without_aug):
            improvement = (with_aug - without_aug) / without_aug * 100
            report.append(f"- データ拡張使用時の平均精度: {with_aug:.4f}")
            report.append(f"- データ拡張未使用時の平均精度: {without_aug:.4f}")
            report.append(f"- 改善率: {improvement:.1f}%")
    
    # バッチ正規化の効果
    if 'use_batch_norm' in df.columns:
        with_bn = df[df['use_batch_norm'] == True]['test_accuracy'].mean()
        without_bn = df[df['use_batch_norm'] == False]['test_accuracy'].mean()
        if not pd.isna(with_bn) and not pd.isna(without_bn):
            improvement = (with_bn - without_bn) / without_bn * 100
            report.append(f"\n- バッチ正規化使用時の平均精度: {with_bn:.4f}")
            report.append(f"- バッチ正規化未使用時の平均精度: {without_bn:.4f}")
            report.append(f"- 改善率: {improvement:.1f}%")
    
    # 深さと精度の関係
    if 'conv_layers' in df.columns:
        correlation = df['conv_layers'].corr(df['test_accuracy'])
        report.append(f"\n- 畳み込み層数と精度の相関: {correlation:.3f}")
    
    # 推奨事項
    report.append("\n## 推奨設定")
    report.append("\n実験結果に基づく推奨設定：")
    
    # 精度重視
    best_acc = df_sorted.iloc[0]
    report.append("\n### 精度重視の場合")
    report.append(f"- アーキテクチャ: {best_acc.get('architecture', 'standard')}")
    report.append(f"- 畳み込み層数: {best_acc['conv_layers']}")
    report.append(f"- フィルタ数: {best_acc['filters']}")
    report.append(f"- 活性化関数: {best_acc['activation']}")
    report.append(f"- 最適化手法: {best_acc['optimizer']}")
    report.append(f"- データ拡張: {best_acc.get('augmentation_strength', 'なし')}")
    
    # 高速学習重視
    if 'training_time_seconds' in df.columns:
        # 精度0.7以上で最も学習時間が短いモデル
        good_models = df[df['test_accuracy'] >= 0.7]
        if len(good_models) > 0:
            fastest = good_models.loc[good_models['training_time_seconds'].idxmin()]
            report.append("\n### 高速学習重視の場合（精度0.7以上）")
            report.append(f"- 実験名: {fastest['experiment_name']}")
            report.append(f"- テスト精度: {fastest['test_accuracy']:.4f}")
            report.append(f"- 学習時間: {fastest['training_time_seconds']/60:.1f}分")
            report.append(f"- 畳み込み層数: {fastest['conv_layers']}")
            report.append(f"- バッチサイズ: {int(fastest['batch_size'])}")
    
    # レポートを保存
    with open("outputs/experiment_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    
    print("実験レポートを outputs/experiment_report.md に保存しました")


def plot_confusion_matrix_sample():
    """最良モデルの混同行列を生成（サンプル）"""
    # 注: 実際の混同行列を生成するには、保存されたモデルを読み込んで
    # テストデータで予測を行う必要があります
    print("混同行列の生成には、保存されたモデルの読み込みが必要です")


def main():
    print("CIFAR-10 CNN実験結果を分析中...")
    
    # 各種分析を実行
    try:
        plot_hyperparameter_impact()
    except Exception as e:
        print(f"ハイパーパラメータ分析でエラー: {e}")
    
    try:
        plot_training_curves_comparison()
    except Exception as e:
        print(f"学習曲線比較でエラー: {e}")
    
    try:
        plot_architecture_comparison()
    except Exception as e:
        print(f"アーキテクチャ比較でエラー: {e}")
    
    try:
        generate_report()
    except Exception as e:
        print(f"レポート生成でエラー: {e}")
    
    print("\n分析完了！")
    print("生成されたファイル:")
    print("- outputs/hyperparameter_analysis.png: ハイパーパラメータの影響分析")
    print("- outputs/top_models_comparison.png: 上位モデルの学習曲線比較")
    print("- outputs/architecture_comparison.png: アーキテクチャ要素の比較")
    print("- outputs/experiment_report.md: 実験レポート（Markdown形式）")
    
    # 簡易的な統計情報を表示
    summary_file = Path("outputs/experiment_summary.csv")
    if summary_file.exists():
        df = pd.read_csv(summary_file)
        print(f"\n=== 実験統計 ===")
        print(f"実験数: {len(df)}")
        print(f"最高精度: {df['test_accuracy'].max():.4f}")
        print(f"平均精度: {df['test_accuracy'].mean():.4f}")
        print(f"標準偏差: {df['test_accuracy'].std():.4f}")


if __name__ == "__main__":
    main()