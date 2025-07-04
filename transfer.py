import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121, ResNet50
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
from datetime import datetime

# 結果保存用ディレクトリ
RESULTS_DIR = "transfer_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# CIFAR-10のクラス名
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

def prepare_data():
    """CIFAR-10データの準備"""
    print("Loading CIFAR-10...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # float32に変換
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    # 32x32を224x224にリサイズ
    print("Resizing images to 224x224...")
    x_train_resized = tf.image.resize(x_train, (224, 224))
    x_test_resized = tf.image.resize(x_test, (224, 224))
    
    return x_train_resized, y_train, x_test_resized, y_test

def create_model(model_name='DenseNet121', freeze_base=True):
    """転移学習モデルの作成"""
    if model_name == 'DenseNet121':
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        preprocess_fn = densenet_preprocess
    else:  # ResNet50
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        preprocess_fn = resnet_preprocess
    
    # ベースモデルの凍結設定
    base_model.trainable = not freeze_base
    
    # モデルの構築
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])
    
    return model, preprocess_fn

def create_generators(x_train, y_train, x_test, y_test, preprocess_fn):
    """データジェネレータの作成"""
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn
    )
    
    train_gen = train_datagen.flow(
        x_train, y_train,
        batch_size=32,
        subset='training',
        shuffle=True
    )
    
    val_gen = train_datagen.flow(
        x_train, y_train,
        batch_size=32,
        subset='validation',
        shuffle=False
    )
    
    test_gen = test_datagen.flow(
        x_test, y_test,
        batch_size=32,
        shuffle=False
    )
    
    return train_gen, val_gen, test_gen

def train_model(model, train_gen, val_gen, epochs=15):
    """モデルの訓練"""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Early Stoppingのみ使用（シンプルに）
    early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
    
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=[early_stop],
        verbose=1
    )
    
    return history

def evaluate_and_plot(model, test_gen, history, experiment_name):
    """評価と可視化"""
    # テストデータで評価
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"\nTest accuracy: {test_acc:.4f}")
    
    # 予測
    y_pred = model.predict(test_gen)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_gen.labels[:len(y_pred_classes)]
    
    # 1. 学習曲線
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title(f'{experiment_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title(f'{experiment_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{experiment_name}_curves.png'))
    plt.close()
    
    # 2. 混同行列
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'{experiment_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{experiment_name}_confusion.png'))
    plt.close()
    
    # 3. 分類レポート
    report = classification_report(y_true, y_pred_classes, target_names=CLASS_NAMES)
    with open(os.path.join(RESULTS_DIR, f'{experiment_name}_report.txt'), 'w') as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
        f.write(report)
    
    return test_acc

def run_experiments():
    """メイン実験の実行"""
    # データの準備
    x_train, y_train, x_test, y_test = prepare_data()
    
    # 実験設定（シンプルに3つの実験のみ）
    experiments = [
        {"name": "ResNet50_frozen", "model": "ResNet50", "freeze": True},
    ]
    
    results = []
    
    for exp in experiments:
        print(f"\n{'='*50}")
        print(f"実験: {exp['name']}")
        print(f"{'='*50}")
        
        # モデル作成
        model, preprocess_fn = create_model(exp['model'], exp['freeze'])
        
        # データジェネレータ作成
        train_gen, val_gen, test_gen = create_generators(
            x_train, y_train, x_test, y_test, preprocess_fn
        )
        
        # 訓練
        history = train_model(model, train_gen, val_gen, epochs=5)
        
        # 評価と可視化
        test_acc = evaluate_and_plot(model, test_gen, history, exp['name'])
        
        results.append({
            'experiment': exp['name'],
            'model': exp['model'],
            'frozen': exp['freeze'],
            'test_accuracy': test_acc
        })
        
        # メモリ解放
        del model
        tf.keras.backend.clear_session()
    
    # 結果の比較プロット
    plt.figure(figsize=(10, 6))
    names = [r['experiment'] for r in results]
    accs = [r['test_accuracy'] for r in results]
    
    plt.bar(names, accs, color=['blue', 'orange', 'green'])
    plt.title('Transfer Learning Results Comparison')
    plt.xlabel('Experiment')
    plt.ylabel('Test Accuracy')
    plt.ylim(0, 1)
    
    for i, acc in enumerate(accs):
        plt.text(i, acc + 0.01, f'{acc:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'comparison.png'))
    plt.close()
    
    # 簡易レポート生成
    report = f"""# Transfer Learning 実験結果

実験日時: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## 実験内容
1. ResNet50（凍結）: ImageNetで事前学習済み、ベースモデル凍結

## 結果
"""
    
    for r in results:
        report += f"- **{r['experiment']}**: {r['test_accuracy']:.4f}\n"
    
    report += f"""
## 結論
最高精度: {max(results, key=lambda x: x['test_accuracy'])['experiment']} ({max(accs):.4f})

## 生成ファイル
- 各実験の学習曲線: *_curves.png
- 各実験の混同行列: *_confusion.png
- 各実験の詳細レポート: *_report.txt
- 比較グラフ: comparison.png
"""
    
    with open(os.path.join(RESULTS_DIR, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n実験完了！結果は '{RESULTS_DIR}' に保存されました。")

if __name__ == "__main__":
    # GPU設定
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    run_experiments()