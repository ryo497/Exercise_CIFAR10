import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import pandas as pd


class ProgressCallback(callbacks.Callback):
    """エポック進行状況を表示するカスタムコールバック"""
    
    def __init__(self):
        super().__init__()
        self.batch_count = 0
        self.total_batches = 0
        self.current_epoch = 0
        
    def on_train_begin(self, logs=None):
        print("\n" + "="*60)
        print("学習開始")
        print("="*60)
        
    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch
        self.batch_count = 0
        self.total_batches = self.params['steps']
        print(f"\nエポック {epoch + 1}/{self.params['epochs']} 開始...")
        
    def on_batch_end(self, batch, logs=None):
        self.batch_count = batch + 1
        if self.batch_count % 20 == 0 or self.batch_count == self.total_batches:
            logs = logs or {}
            acc = logs.get('accuracy', 0)
            loss = logs.get('loss', 0)
            progress = self.batch_count / self.total_batches * 100
            print(f"  バッチ {self.batch_count}/{self.total_batches} ({progress:.1f}%) - "
                  f"損失: {loss:.4f}, 精度: {acc:.4f}")
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_acc = logs.get('accuracy', 0)
        train_loss = logs.get('loss', 0)
        val_acc = logs.get('val_accuracy', 0)
        val_loss = logs.get('val_loss', 0)
        
        print(f"\nエポック {epoch + 1}/{self.params['epochs']} 完了")
        print(f"  訓練 - 損失: {train_loss:.4f}, 精度: {train_acc:.4f}")
        print(f"  検証 - 損失: {val_loss:.4f}, 精度: {val_acc:.4f}")
        
        # 改善があった場合の表示
        if hasattr(self, 'best_val_acc'):
            if val_acc > self.best_val_acc:
                print(f"  ✓ 検証精度が改善しました! ({self.best_val_acc:.4f} → {val_acc:.4f})")
                self.best_val_acc = val_acc
        else:
            self.best_val_acc = val_acc
            
        print("-" * 60)
    
    def on_train_end(self, logs=None):
        print("\n" + "="*60)
        print("学習完了")
        print("="*60)


def create_data_augmentation(strength="medium"):
    """データ拡張のレイヤーを作成
    
    Args:
        strength: "weak", "medium", "strong" のいずれか
    """
    if strength == "weak":
        return tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.05),
        ])
    elif strength == "medium":
        return tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.1, 0.1),
        ])
    elif strength == "strong":
        return tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.2),
            layers.RandomTranslation(0.15, 0.15),
            layers.RandomContrast(0.2),
        ])
    else:
        return None


def build_cnn_model(conv_layers: int, 
                    filters: list,
                    kernel_size: int,
                    pooling_type: str,
                    pooling_size: int,
                    dropout_rate: float,
                    use_batch_norm: bool,
                    use_global_avg_pool: bool,
                    dense_units: list,
                    activation: str = "relu") -> models.Sequential:
    """柔軟なCNNアーキテクチャを構築
    
    Args:
        conv_layers: 畳み込み層の数
        filters: 各層のフィルタ数のリスト（例: [32, 64, 128]）
        kernel_size: カーネルサイズ（3 or 5）
        pooling_type: "max" or "average"
        pooling_size: プーリングサイズ（通常2）
        dropout_rate: ドロップアウト率
        use_batch_norm: バッチ正規化を使用するか
        use_global_avg_pool: Global Average Poolingを使用するか
        dense_units: 全結合層のユニット数リスト
        activation: 活性化関数
    """
    model = models.Sequential(name="cifar10_cnn")
    
    # 入力層
    model.add(layers.Input(shape=(32, 32, 3)))
    
    # 畳み込み層の構築
    for i in range(conv_layers):
        # フィルタ数の決定
        if i < len(filters):
            num_filters = filters[i]
        else:
            # リストが足りない場合は最後の値を使用
            num_filters = filters[-1]
        
        # 畳み込み層
        model.add(layers.Conv2D(
            num_filters, 
            kernel_size=(kernel_size, kernel_size),
            padding='same',
            name=f"conv_{i+1}"
        ))
        
        # バッチ正規化
        if use_batch_norm:
            model.add(layers.BatchNormalization(name=f"batch_norm_{i+1}"))
        
        # 活性化関数
        model.add(layers.Activation(activation, name=f"activation_{i+1}"))
        
        # 2層ごとにプーリング（またはストライドで調整）
        if (i + 1) % 2 == 0 or i == conv_layers - 1:
            pool_idx = len([layer for layer in model.layers if 'pool' in layer.name]) + 1
            if pooling_type == "max":
                model.add(layers.MaxPooling2D(
                    pool_size=(pooling_size, pooling_size),
                    name=f"max_pool_{pool_idx}"
                ))
            else:
                model.add(layers.AveragePooling2D(
                    pool_size=(pooling_size, pooling_size),
                    name=f"avg_pool_{pool_idx}"
                ))
            
            # プーリング後のドロップアウト
            if dropout_rate > 0:
                model.add(layers.Dropout(dropout_rate, name=f"dropout_conv_{pool_idx}"))
    
    # 全結合層への移行
    if use_global_avg_pool:
        model.add(layers.GlobalAveragePooling2D(name="global_avg_pool"))
    else:
        model.add(layers.Flatten(name="flatten"))
    
    # 全結合層
    for i, units in enumerate(dense_units):
        model.add(layers.Dense(units, name=f"dense_{i+1}"))
        if use_batch_norm:
            model.add(layers.BatchNormalization(name=f"dense_batch_norm_{i+1}"))
        model.add(layers.Activation(activation, name=f"dense_activation_{i+1}"))
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate, name=f"dense_dropout_{i+1}"))
    
    # 出力層
    model.add(layers.Dense(10, activation="softmax", name="output"))
    
    return model


def build_resnet_style_model(blocks: int, filters: int, dropout_rate: float = 0.0) -> models.Model:
    """簡易的なResNet風モデル（スキップ接続あり）"""
    inputs = layers.Input(shape=(32, 32, 3))
    
    # 初期畳み込み
    x = layers.Conv2D(filters, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Residual blocks
    for i in range(blocks):
        # スキップ接続の保存
        shortcut = x
        
        # メインパス
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # スキップ接続を追加
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        
        # 2ブロックごとにダウンサンプリング
        if (i + 1) % 2 == 0 and i < blocks - 1:
            x = layers.MaxPooling2D(2)(x)
            filters *= 2
            # フィルタ数を合わせるための1x1畳み込み
            x = layers.Conv2D(filters, 1, padding='same')(x)
            
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)
    
    # 出力層
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    return models.Model(inputs, outputs, name="cifar10_resnet")


def create_lr_scheduler(schedule_type: str, initial_lr: float, epochs: int):
    """学習率スケジューラーを作成"""
    if schedule_type == "exponential":
        decay_rate = 0.96
        return callbacks.LearningRateScheduler(
            lambda epoch: initial_lr * (decay_rate ** (epoch / 10))
        )
    elif schedule_type == "cosine":
        return callbacks.LearningRateScheduler(
            lambda epoch: initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / epochs))
        )
    elif schedule_type == "step":
        return callbacks.LearningRateScheduler(
            lambda epoch: initial_lr * (0.5 ** (epoch // 30))
        )
    elif schedule_type == "warmup_cosine":
        warmup_epochs = 5
        def schedule(epoch):
            if epoch < warmup_epochs:
                return initial_lr * (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
                return initial_lr * 0.5 * (1 + np.cos(np.pi * progress))
        return callbacks.LearningRateScheduler(schedule)
    else:
        return None


def plot_history(history, out_path: Path, experiment_name: str) -> None:
    """学習履歴をプロット"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 精度プロット
    ax1.plot(history.history["accuracy"], label="Train Accuracy")
    ax1.plot(history.history["val_accuracy"], label="Validation Accuracy")
    ax1.set_title(f"Accuracy - {experiment_name}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 損失プロット
    ax2.plot(history.history["loss"], label="Train Loss")
    ax2.plot(history.history["val_loss"], label="Validation Loss")
    ax2.set_title(f"Loss - {experiment_name}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_experiment_results(args, history, test_acc, test_loss, training_time, out_dir):
    """実験結果を構造化して保存"""
    results = {
        "experiment_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "hyperparameters": {
            "epochs": args.epochs,
            "conv_layers": args.conv_layers,
            "filters": args.filters,
            "kernel_size": args.kernel_size,
            "pooling_type": args.pooling_type,
            "pooling_size": args.pooling_size,
            "architecture": args.architecture,
            "activation": args.activation,
            "optimizer": args.optimizer,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "dropout_rate": args.dropout_rate,
            "use_batch_norm": args.use_batch_norm,
            "use_global_avg_pool": args.use_global_avg_pool,
            "augmentation_strength": args.augmentation_strength,
            "lr_schedule": args.lr_schedule,
            "dense_units": args.dense_units,
            "use_mixed_precision": args.use_mixed_precision
        },
        "results": {
            "test_accuracy": float(test_acc),
            "test_loss": float(test_loss),
            "best_val_accuracy": float(max(history.history["val_accuracy"])),
            "best_val_loss": float(min(history.history["val_loss"])),
            "final_train_accuracy": float(history.history["accuracy"][-1]),
            "final_train_loss": float(history.history["loss"][-1]),
            "training_time_seconds": training_time,
            "total_epochs": len(history.history["accuracy"])
        },
        "history": {
            "train_accuracy": [float(x) for x in history.history["accuracy"]],
            "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
            "train_loss": [float(x) for x in history.history["loss"]],
            "val_loss": [float(x) for x in history.history["val_loss"]]
        }
    }
    
    # JSON結果を保存
    with open(out_dir / "experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # CSVサマリーを保存/更新
    summary_file = out_dir.parent / "experiment_summary.csv"
    summary_data = {
        "experiment_id": results["experiment_id"],
        **results["hyperparameters"],
        **results["results"]
    }
    
    if summary_file.exists():
        df = pd.read_csv(summary_file)
        df = pd.concat([df, pd.DataFrame([summary_data])], ignore_index=True)
    else:
        df = pd.DataFrame([summary_data])
    
    df.to_csv(summary_file, index=False)
    
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CIFAR-10用CNN実験スクリプト"
    )
    
    # アーキテクチャパラメータ
    parser.add_argument("--architecture", type=str, default="standard",
                      choices=["standard", "resnet"],
                      help="モデルアーキテクチャ (default: standard)")
    parser.add_argument("--conv_layers", type=int, default=4,
                      help="畳み込み層の数 (default: 4)")
    parser.add_argument("--filters", type=int, nargs='+', default=[32, 64, 128],
                      help="各層のフィルタ数 (default: 32 64 128)")
    parser.add_argument("--kernel_size", type=int, default=3,
                      choices=[3, 5],
                      help="カーネルサイズ (default: 3)")
    parser.add_argument("--pooling_type", type=str, default="max",
                      choices=["max", "average"],
                      help="プーリングタイプ (default: max)")
    parser.add_argument("--pooling_size", type=int, default=2,
                      help="プーリングサイズ (default: 2)")
    parser.add_argument("--dense_units", type=int, nargs='+', default=[128],
                      help="全結合層のユニット数 (default: 128)")
    
    # 学習パラメータ
    parser.add_argument("--epochs", type=int, default=50,
                      help="学習エポック数 (default: 50)")
    parser.add_argument("--batch_size", type=int, default=128,
                      help="バッチサイズ (default: 128)")
    parser.add_argument("--optimizer", type=str, default="adam",
                      choices=["adam", "sgd", "rmsprop", "adamw"],
                      help="最適化アルゴリズム (default: adam)")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                      help="初期学習率 (default: 0.001)")
    parser.add_argument("--activation", type=str, default="relu",
                      choices=["relu", "elu", "selu", "swish"],
                      help="活性化関数 (default: relu)")
    
    # 正則化パラメータ
    parser.add_argument("--dropout_rate", type=float, default=0.0,
                      help="ドロップアウト率 (default: 0.0)")
    parser.add_argument("--use_batch_norm", action="store_true",
                      help="バッチ正規化を使用")
    parser.add_argument("--use_global_avg_pool", action="store_true",
                      help="Global Average Poolingを使用")
    
    # 拡張機能
    parser.add_argument("--augmentation_strength", type=str, default="none",
                      choices=["none", "weak", "medium", "strong"],
                      help="データ拡張の強度 (default: none)")
    parser.add_argument("--lr_schedule", type=str, default="none",
                      choices=["none", "exponential", "cosine", "step", "warmup_cosine"],
                      help="学習率スケジューリング (default: none)")
    parser.add_argument("--use_mixed_precision", action="store_true",
                      help="混合精度学習を使用")
    
    # 出力パラメータ
    parser.add_argument("--output_dir", type=str, default="outputs",
                      help="結果出力ディレクトリ (default: ./outputs)")
    parser.add_argument("--experiment_name", type=str, default="",
                      help="実験名")
    
    args = parser.parse_args()
    
    # 混合精度学習の設定
    if args.use_mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
    
    # 実験名の生成
    if not args.experiment_name:
        args.experiment_name = f"{args.architecture}_{args.conv_layers}layers_{args.filters[0]}filters"
    
    # 出力ディレクトリの作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"{timestamp}_{args.experiment_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 設定の保存
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"\n=== 実験開始: {args.experiment_name} ===")
    print(f"出力ディレクトリ: {out_dir.resolve()}")
    
    # データの読み込みと前処理
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    # クラス名
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # データ拡張の設定
    augmentation = None
    if args.augmentation_strength != "none":
        augmentation = create_data_augmentation(args.augmentation_strength)
    
    # モデルの構築
    if args.architecture == "standard":
        model = build_cnn_model(
            conv_layers=args.conv_layers,
            filters=args.filters,
            kernel_size=args.kernel_size,
            pooling_type=args.pooling_type,
            pooling_size=args.pooling_size,
            dropout_rate=args.dropout_rate,
            use_batch_norm=args.use_batch_norm,
            use_global_avg_pool=args.use_global_avg_pool,
            dense_units=args.dense_units,
            activation=args.activation
        )
    else:  # resnet
        model = build_resnet_style_model(
            blocks=args.conv_layers,
            filters=args.filters[0],
            dropout_rate=args.dropout_rate
        )
    
    # データ拡張をモデルに組み込む
    if augmentation is not None:
        augmented_model = models.Sequential([
            augmentation,
            model
        ])
        model = augmented_model
    
    model.summary()
    
    # 最適化手法の選択
    optimizers = {
        "adam": tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        "sgd": tf.keras.optimizers.SGD(learning_rate=args.learning_rate, momentum=0.9, nesterov=True),
        "rmsprop": tf.keras.optimizers.RMSprop(learning_rate=args.learning_rate),
        "adamw": tf.keras.optimizers.Adam(learning_rate=args.learning_rate)  # AdamWの代替としてAdamを使用
    }
    
    # モデルのコンパイル
    model.compile(
        optimizer=optimizers[args.optimizer],
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # コールバックの設定
    callback_list = [
        ProgressCallback(),  # カスタム進行状況表示
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=0  # verboseを0にしてProgressCallbackの表示と重複を避ける
        ),
        callbacks.ModelCheckpoint(
            str(out_dir / "best_model.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=0  # verboseを0にしてProgressCallbackの表示と重複を避ける
        ),
        callbacks.CSVLogger(str(out_dir / "training_log.csv")),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0  # verboseを0にしてProgressCallbackの表示と重複を避ける
        )
    ]
    
    # 学習率スケジューラーの追加
    lr_scheduler = create_lr_scheduler(args.lr_schedule, args.learning_rate, args.epochs)
    if lr_scheduler:
        callback_list.append(lr_scheduler)
    
    # 学習の実行
    start_time = time.time()
    history = model.fit(
        x_train, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=0.1,
        callbacks=callback_list,
        verbose=0  # 詳細なログはProgressCallbackで表示
    )
    training_time = time.time() - start_time
    
    # テストセットでの評価
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n[RESULT] Test accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")
    print(f"Training time: {training_time:.2f} seconds")
    
    # 結果の保存
    model.save(str(out_dir / "final_model.h5"))
    plot_history(history, out_dir / "training_curves.png", args.experiment_name)
    
    # 構造化結果の保存
    results = save_experiment_results(args, history, test_acc, test_loss, training_time, out_dir)
    
    # サマリーの出力
    print(f"\n=== 実験完了: {args.experiment_name} ===")
    print(f"Test Accuracy: {results['results']['test_accuracy']:.4f}")
    print(f"Best Val Accuracy: {results['results']['best_val_accuracy']:.4f}")
    print(f"Training Time: {results['results']['training_time_seconds']:.2f}s")
    print(f"Total Epochs: {results['results']['total_epochs']}")
    print(f"\n結果保存先: {out_dir.resolve()}")


if __name__ == "__main__":
    main()