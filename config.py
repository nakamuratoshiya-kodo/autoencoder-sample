# config.py

# モデル構成
MODEL_CONFIG = {
    "input_shape": (3, 128, 128),  # (channels, height, width), grayscale時は1, カラー時は3に設定
    "hidden_dims": [64, 32, 16],  # 潜在空間の次元数
    "latent_dim": 32,
    "activation": "ReLU"
}

# 学習条件
TRAINING_CONFIG = {
    "batch_size": 200,
    "epochs": 50,
    "learning_rate": 1e-3,
    "optimizer": "Adam",
    "weight_decay": 1e-5,
}

# データ設定
DATA_CONFIG = {
    "train_dir": "./data/train",
    "test_dir": "./data/test",
    "normalize": True,  # True: 正規化, False: 正規化なし
    "image_size": (128, 128),
    "grayscale": False  # True: モノクロ画像, False: カラー画像
}

# MLflow設定
MLFLOW_CONFIG = {
    "use_mlflow": True,
    "experiment_name": "Autoencoder_Anomaly_PoC",
    "tracking_uri": "http://localhost:8080"
}

# 異常検知設定
ANOMALY_CONFIG = {
    "threshold": None,
    "threshold_method": "mean+3std"
}

# 乱数シード設定
SEED_CONFIG = {
    "seed": 42,
    "deterministic": True,  # 完全な再現性を優先（速度低下あり）
    "benchmark": False      # 高速化したい場合はTrue（再現性は落ちる）
}