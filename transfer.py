import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121, ResNet50, MobileNetV2
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from datetime import datetime
import os
import json

# 実験結果を保存するディレクトリ
RESULTS_DIR = "transfer_learning_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# CIFAR-10のクラス名
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

class TransferLearningExperiment:
    def __init__(self):
        self.img_size = (224, 224)
        self.batch_size = 32
        self.epochs = 30
        self.results = []
        
        # データセットの準備
        print("Loading CIFAR-10 dataset...")
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar10.load_data()
        self.num_classes = 10
        
        # データの正規化（0-255の範囲に）
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        
    def get_model_config(self):
        """使用可能なモデルと前処理関数の設定"""
        return {
            'DenseNet121': {
                'model_fn': DenseNet121,
                'preprocess_fn': densenet_preprocess,
                'last_conv_layer': 'conv5_block16_concat'
            },
            'ResNet50': {
                'model_fn': ResNet50,
                'preprocess_fn': resnet_preprocess,
                'last_conv_layer': 'conv5_block3_out'
            },
            'MobileNetV2': {
                'model_fn': MobileNetV2,
                'preprocess_fn': mobilenet_preprocess,
                'last_conv_layer': 'out_relu'
            }
        }
    
    def create_data_generators(self, preprocess_fn):
        """データジェネレータの作成"""
        # 32x32を224x224にリサイズするためのデータ拡張
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_fn,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            validation_split=0.2
        )
        
        test_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_fn
        )
        
        # リサイズして224x224にする
        x_train_resized = tf.image.resize(self.x_train, self.img_size)
        x_test_resized = tf.image.resize(self.x_test, self.img_size)
        
        train_gen = train_datagen.flow(
            x_train_resized, self.y_train,
            batch_size=self.batch_size,
            subset='training',
            shuffle=True
        )
        
        val_gen = train_datagen.flow(
            x_train_resized, self.y_train,
            batch_size=self.batch_size,
            subset='validation',
            shuffle=False
        )
        
        test_gen = test_datagen.flow(
            x_test_resized, self.y_test,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        return train_gen, val_gen, test_gen
    
    def create_model(self, base_model_name, freeze_pattern='all'):
        """転移学習モデルの作成
        
        Args:
            base_model_name: 'DenseNet121', 'ResNet50', 'MobileNetV2'
            freeze_pattern: 'all' (全層凍結), 'partial' (部分凍結), 'none' (凍結なし)
        """
        model_config = self.get_model_config()[base_model_name]
        
        # ベースモデルの作成
        base_model = model_config['model_fn'](
            include_top=False,
            weights='imagenet',
            input_shape=self.img_size + (3,)
        )
        
        # 凍結パターンの適用
        if freeze_pattern == 'all':
            # 全層凍結
            base_model.trainable = False
        elif freeze_pattern == 'partial':
            # 最後の20%の層のみ学習可能にする
            total_layers = len(base_model.layers)
            freeze_until = int(total_layers * 0.8)
            for layer in base_model.layers[:freeze_until]:
                layer.trainable = False
            for layer in base_model.layers[freeze_until:]:
                layer.trainable = True
        else:  # 'none'
            # 全層学習可能
            base_model.trainable = True
        
        # 分類ヘッドの追加
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model, model_config['preprocess_fn']
    
    def compile_model(self, model, learning_rate=0.001, freeze_pattern='all'):
        """モデルのコンパイル"""
        # 凍結パターンに応じて学習率を調整
        if freeze_pattern == 'none':
            # 全層学習の場合は学習率を下げる
            learning_rate = learning_rate * 0.1
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, model, train_gen, val_gen, experiment_name):
        """モデルの訓練"""
        # コールバックの設定
        checkpoint_path = os.path.join(RESULTS_DIR, f"{experiment_name}_best.h5")
        
        callbacks = [
            ModelCheckpoint(
                checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # 訓練
        history = model.fit(
            train_gen,
            epochs=self.epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, model, test_gen, experiment_name):
        """モデルの評価"""
        # テストデータでの評価
        test_loss, test_accuracy = model.evaluate(test_gen, verbose=0)
        
        # 予測
        y_pred_probs = model.predict(test_gen, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = self.y_test.flatten()[:len(y_pred)]
        
        # 各種メトリクスの計算
        precision = metrics.precision_score(y_true, y_pred, average='weighted')
        recall = metrics.recall_score(y_true, y_pred, average='weighted')
        f1 = metrics.f1_score(y_true, y_pred, average='weighted')
        
        # 混同行列
        cm = metrics.confusion_matrix(y_true, y_pred)
        
        results = {
            'experiment_name': experiment_name,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist()
        }
        
        return results, cm
    
    def plot_training_history(self, history, experiment_name):
        """学習曲線のプロット"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 精度のプロット
        ax1.plot(history.history['accuracy'], label='Train')
        ax1.plot(history.history['val_accuracy'], label='Validation')
        ax1.set_title(f'{experiment_name} - Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # 損失のプロット
        ax2.plot(history.history['loss'], label='Train')
        ax2.plot(history.history['val_loss'], label='Validation')
        ax2.set_title(f'{experiment_name} - Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'{experiment_name}_training_curves.png'), dpi=300)
        plt.close()
    
    def plot_confusion_matrix(self, cm, experiment_name):
        """混同行列のプロット"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=CIFAR10_CLASSES, yticklabels=CIFAR10_CLASSES)
        plt.title(f'{experiment_name} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'{experiment_name}_confusion_matrix.png'), dpi=300)
        plt.close()
    
    def run_experiment(self, model_name, freeze_pattern, learning_rate=0.001):
        """単一の実験を実行"""
        experiment_name = f"{model_name}_{freeze_pattern}_lr{learning_rate}"
        print(f"\n{'='*50}")
        print(f"Running experiment: {experiment_name}")
        print(f"{'='*50}")
        
        # モデルの作成
        model, preprocess_fn = self.create_model(model_name, freeze_pattern)
        model = self.compile_model(model, learning_rate, freeze_pattern)
        
        # データジェネレータの作成
        train_gen, val_gen, test_gen = self.create_data_generators(preprocess_fn)
        
        # 訓練
        print("Training model...")
        history = self.train_model(model, train_gen, val_gen, experiment_name)
        
        # 評価
        print("Evaluating model...")
        results, cm = self.evaluate_model(model, test_gen, experiment_name)
        results['learning_rate'] = learning_rate
        results['freeze_pattern'] = freeze_pattern
        results['model_name'] = model_name
        
        # 可視化
        self.plot_training_history(history, experiment_name)
        self.plot_confusion_matrix(cm, experiment_name)
        
        # 結果の保存
        self.results.append(results)
        
        # メモリの解放
        del model
        tf.keras.backend.clear_session()
        
        return results
    
    def run_all_experiments(self):
        """全ての実験を実行"""
        models = ['DenseNet121', 'ResNet50', 'MobileNetV2']
        freeze_patterns = ['all', 'partial', 'none']
        learning_rates = [0.001, 0.0001]
        
        total_experiments = len(models) * len(freeze_patterns) * len(learning_rates)
        experiment_count = 0
        
        for model_name in models:
            for freeze_pattern in freeze_patterns:
                for lr in learning_rates:
                    experiment_count += 1
                    print(f"\nExperiment {experiment_count}/{total_experiments}")
                    self.run_experiment(model_name, freeze_pattern, lr)
        
        # 結果の保存
        self.save_results()
        self.create_comparison_plots()
        self.generate_report()
    
    def save_results(self):
        """実験結果をCSVとJSONで保存"""
        # DataFrameに変換
        df = pd.DataFrame(self.results)
        
        # confusion_matrixを除外してCSV保存
        df_csv = df.drop('confusion_matrix', axis=1)
        df_csv.to_csv(os.path.join(RESULTS_DIR, 'experiment_results.csv'), index=False)
        
        # 完全な結果をJSON保存
        with open(os.path.join(RESULTS_DIR, 'experiment_results.json'), 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def create_comparison_plots(self):
        """比較プロットの作成"""
        df = pd.DataFrame(self.results)
        
        # モデル別の精度比較
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. モデル別・凍結パターン別の精度
        ax = axes[0, 0]
        pivot_accuracy = df.pivot_table(values='test_accuracy', 
                                       index='model_name', 
                                       columns='freeze_pattern', 
                                       aggfunc='max')
        pivot_accuracy.plot(kind='bar', ax=ax)
        ax.set_title('Test Accuracy by Model and Freeze Pattern')
        ax.set_ylabel('Test Accuracy')
        ax.set_xlabel('Model')
        ax.legend(title='Freeze Pattern')
        ax.grid(True, alpha=0.3)
        
        # 2. 学習率の影響
        ax = axes[0, 1]
        for model in df['model_name'].unique():
            model_data = df[df['model_name'] == model]
            model_data.plot(x='learning_rate', y='test_accuracy', 
                          label=model, marker='o', ax=ax)
        ax.set_xscale('log')
        ax.set_title('Test Accuracy vs Learning Rate')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Test Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. F1スコアの比較
        ax = axes[1, 0]
        pivot_f1 = df.pivot_table(values='f1_score', 
                                 index='model_name', 
                                 columns='freeze_pattern', 
                                 aggfunc='max')
        pivot_f1.plot(kind='bar', ax=ax)
        ax.set_title('F1 Score by Model and Freeze Pattern')
        ax.set_ylabel('F1 Score')
        ax.set_xlabel('Model')
        ax.legend(title='Freeze Pattern')
        ax.grid(True, alpha=0.3)
        
        # 4. 全実験の精度ランキング
        ax = axes[1, 1]
        df_sorted = df.sort_values('test_accuracy', ascending=True)
        df_sorted['experiment'] = df_sorted['model_name'] + '_' + df_sorted['freeze_pattern']
        df_sorted.plot(x='experiment', y='test_accuracy', kind='barh', ax=ax, legend=False)
        ax.set_title('Test Accuracy Ranking')
        ax.set_xlabel('Test Accuracy')
        ax.set_ylabel('Experiment')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'comparison_plots.png'), dpi=300)
        plt.close()
    
    def generate_report(self):
        """Markdownレポートの生成"""
        df = pd.DataFrame(self.results)
        
        report = f"""# Transfer Learning Experiment Report

## 実験概要
- **実験日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **データセット**: CIFAR-10
- **画像サイズ**: {self.img_size[0]}x{self.img_size[1]}
- **バッチサイズ**: {self.batch_size}
- **エポック数**: {self.epochs}

## 実験条件
### モデル
- DenseNet121
- ResNet50
- MobileNetV2

### 凍結パターン
- **all**: 全層凍結（分類ヘッドのみ学習）
- **partial**: 部分凍結（最後の20%の層を学習）
- **none**: 凍結なし（全層学習）

### 学習率
- 0.001
- 0.0001

## 実験結果サマリー

### 最高精度トップ5
"""
        # トップ5の結果
        top5 = df.nlargest(5, 'test_accuracy')
        report += "| 順位 | モデル | 凍結パターン | 学習率 | テスト精度 | F1スコア |\n"
        report += "|------|--------|--------------|--------|------------|----------|\n"
        for i, row in enumerate(top5.itertuples(), 1):
            report += f"| {i} | {row.model_name} | {row.freeze_pattern} | {row.learning_rate} | {row.test_accuracy:.4f} | {row.f1_score:.4f} |\n"
        
        report += "\n### モデル別最高精度\n"
        report += "| モデル | 最高精度 | 最適凍結パターン | 最適学習率 |\n"
        report += "|--------|----------|------------------|------------|\n"
        
        for model in df['model_name'].unique():
            model_df = df[df['model_name'] == model]
            best = model_df.loc[model_df['test_accuracy'].idxmax()]
            report += f"| {model} | {best['test_accuracy']:.4f} | {best['freeze_pattern']} | {best['learning_rate']} |\n"
        
        report += """
## 考察

### 1. 凍結パターンの影響
"""
        # 凍結パターン別の平均精度
        freeze_stats = df.groupby('freeze_pattern')['test_accuracy'].agg(['mean', 'std'])
        for pattern in freeze_stats.index:
            mean_acc = freeze_stats.loc[pattern, 'mean']
            std_acc = freeze_stats.loc[pattern, 'std']
            report += f"- **{pattern}**: 平均精度 {mean_acc:.4f} (±{std_acc:.4f})\n"
        
        report += """
### 2. モデルアーキテクチャの比較
"""
        # モデル別の統計
        model_stats = df.groupby('model_name')['test_accuracy'].agg(['mean', 'std', 'max'])
        for model in model_stats.index:
            mean_acc = model_stats.loc[model, 'mean']
            std_acc = model_stats.loc[model, 'std']
            max_acc = model_stats.loc[model, 'max']
            report += f"- **{model}**: 平均精度 {mean_acc:.4f} (±{std_acc:.4f}), 最高精度 {max_acc:.4f}\n"
        
        report += """
### 3. 学習率の影響
"""
        # 学習率別の統計
        lr_stats = df.groupby('learning_rate')['test_accuracy'].agg(['mean', 'std'])
        for lr in lr_stats.index:
            mean_acc = lr_stats.loc[lr, 'mean']
            std_acc = lr_stats.loc[lr, 'std']
            report += f"- **{lr}**: 平均精度 {mean_acc:.4f} (±{std_acc:.4f})\n"
        
        report += """
## 結論

1. **最適な組み合わせ**: """ + f"{top5.iloc[0]['model_name']}モデルで{top5.iloc[0]['freeze_pattern']}凍結パターン、学習率{top5.iloc[0]['learning_rate']}が最高精度{top5.iloc[0]['test_accuracy']:.4f}を達成"
        
        report += """
2. **凍結パターンの選択**: データセットと事前学習モデルのドメインの違いにより、部分凍結または全層学習が有効な場合がある
3. **モデル選択**: 各モデルには異なる特性があり、タスクに応じて選択する必要がある
4. **学習率の調整**: 凍結パターンに応じて適切な学習率を選択することが重要

## 生成されたファイル
- `experiment_results.csv`: 実験結果のサマリー
- `experiment_results.json`: 詳細な実験結果（混同行列を含む）
- `comparison_plots.png`: 比較グラフ
- 各実験の学習曲線と混同行列

## 参考情報
- 実験コード: `transfer.py`
- 結果ディレクトリ: `transfer_learning_results/`
"""
        
        # レポートの保存
        with open(os.path.join(RESULTS_DIR, 'experiment_report.md'), 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\nReport saved to {os.path.join(RESULTS_DIR, 'experiment_report.md')}")

def main():
    """メイン実行関数"""
    # GPUメモリの成長を許可
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    # 実験の実行
    experiment = TransferLearningExperiment()
    
    # 単一実験のテスト（デバッグ用）
    # experiment.run_experiment('DenseNet121', 'all', 0.001)
    
    # 全実験の実行
    experiment.run_all_experiments()
    
    print("\n" + "="*50)
    print("All experiments completed!")
    print(f"Results saved in '{RESULTS_DIR}' directory")
    print("="*50)

if __name__ == "__main__":
    main()