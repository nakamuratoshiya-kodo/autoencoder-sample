# train.py
# オートエンコーダの学習スクリプト（MLflow切替・進捗バー・カラー/グレースケール対応）

import torch
import torch.nn as nn
import torch.optim as optim
from models.autoencoder import ConvAutoencoder
from utils import get_image_dataloader
from config import MODEL_CONFIG, TRAINING_CONFIG, DATA_CONFIG, MLFLOW_CONFIG
from tqdm import tqdm

if MLFLOW_CONFIG["use_mlflow"]:
    import mlflow
    import mlflow.pytorch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# configからチャンネル数を動的に設定
if DATA_CONFIG["grayscale"]:
    MODEL_CONFIG["input_shape"] = (1, *DATA_CONFIG["image_size"])
else:
    MODEL_CONFIG["input_shape"] = (3, *DATA_CONFIG["image_size"])

def train():
    if MLFLOW_CONFIG["use_mlflow"]:
        mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])
        mlflow.set_experiment(MLFLOW_CONFIG["experiment_name"])
        mlflow_context = mlflow.start_run()
    else:
        mlflow_context = None

    with mlflow_context if mlflow_context else dummy_context():
        train_loader = get_image_dataloader(
            DATA_CONFIG["train_dir"],
            TRAINING_CONFIG["batch_size"],
            DATA_CONFIG["image_size"],
            DATA_CONFIG["grayscale"],
            DATA_CONFIG["normalize"]
        )

        model = ConvAutoencoder(
            input_shape=MODEL_CONFIG["input_shape"],
            latent_dim=MODEL_CONFIG["latent_dim"],
            hidden_dims=MODEL_CONFIG["hidden_dims"],
            activation=MODEL_CONFIG["activation"]
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = getattr(optim, TRAINING_CONFIG["optimizer"])(
            model.parameters(),
            lr=TRAINING_CONFIG["learning_rate"],
            weight_decay=TRAINING_CONFIG["weight_decay"]
        )

        if MLFLOW_CONFIG["use_mlflow"]:
            mlflow.log_params({**MODEL_CONFIG, **TRAINING_CONFIG})

        print("==== 学習開始 ====")
        for epoch in range(1, TRAINING_CONFIG["epochs"] + 1):
            total_loss = 0
            for batch, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{TRAINING_CONFIG['epochs']}"):
                batch = batch.to(device)
                optimizer.zero_grad()
                outputs = model(batch)
                loss = criterion(outputs, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch}/{TRAINING_CONFIG['epochs']}], 平均損失: {avg_loss:.6f}")

            if MLFLOW_CONFIG["use_mlflow"]:
                mlflow.log_metric("train_loss", avg_loss, step=epoch)

        print("==== 学習完了 ====")
        torch.save(model.state_dict(), "autoencoder_model.pt")
        if MLFLOW_CONFIG["use_mlflow"]:
            mlflow.pytorch.log_model(model, "autoencoder_model")

from contextlib import contextmanager
@contextmanager
def dummy_context():
    yield

if __name__ == "__main__":
    train()
