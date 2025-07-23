# evaluate.py
# 評価スクリプト（MLflow切替・進捗バー・カラー/グレースケール対応）

import torch
from models.autoencoder import ConvAutoencoder
from utils import get_image_dataloader
from config import MODEL_CONFIG, DATA_CONFIG, ANOMALY_CONFIG, MLFLOW_CONFIG
import matplotlib.pyplot as plt
import numpy as np
import os
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

def evaluate():
    if MLFLOW_CONFIG["use_mlflow"]:
        mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])
        mlflow.set_experiment(MLFLOW_CONFIG["experiment_name"])
        mlflow_context = mlflow.start_run(run_name="Evaluation")
    else:
        mlflow_context = None

    with mlflow_context if mlflow_context else dummy_context():
        os.makedirs("output/anomaly_samples", exist_ok=True)

        model = ConvAutoencoder(
            input_shape=MODEL_CONFIG["input_shape"],
            latent_dim=MODEL_CONFIG["latent_dim"],
            activation=MODEL_CONFIG["activation"]
        ).to(device)
        model.load_state_dict(torch.load("autoencoder_model.pt"))
        model.eval()

        test_loader = get_image_dataloader(
            DATA_CONFIG["test_dir"],
            batch_size=1,
            image_size=DATA_CONFIG["image_size"],
            grayscale=DATA_CONFIG["grayscale"],
            normalize=DATA_CONFIG["normalize"]
        )

        class_names = test_loader.dataset.classes
        print(f"テストデータクラス: {class_names}")

        errors, images, reconstructions, labels = [], [], [], []
        with torch.no_grad():
            for batch, label in tqdm(test_loader, desc="テストデータ評価中"):
                batch = batch.to(device)
                output = model(batch)
                diff = (output - batch).pow(2).squeeze().cpu().numpy()
                error = np.mean(diff)
                errors.append(error)
                images.append(batch.squeeze().cpu().numpy())
                reconstructions.append(output.squeeze().cpu().numpy())
                labels.append(label.item())

        errors, labels = np.array(errors), np.array(labels)
        if ANOMALY_CONFIG["threshold"] is None:
            good_errors = errors[labels == 0]
            threshold = np.mean(good_errors) + 3 * np.std(good_errors)
            print(f"自動閾値 (goodのみ): {threshold:.6f}")
        else:
            threshold = ANOMALY_CONFIG["threshold"]
        if MLFLOW_CONFIG["use_mlflow"]:
            mlflow.log_param("threshold", threshold)

        predicted_anomalies = errors > threshold
        num_anomalies = np.sum(predicted_anomalies)
        print(f"異常検知: {num_anomalies} / {len(errors)} サンプル")
        if MLFLOW_CONFIG["use_mlflow"]:
            mlflow.log_metric("num_anomalies", int(num_anomalies))

        plt.figure(figsize=(8,6))
        plt.hist(errors[labels == 0], bins=50, alpha=0.7, label="good")
        for class_idx in range(1, len(class_names)):
            plt.hist(errors[labels == class_idx], bins=50, alpha=0.7, label=class_names[class_idx])
        plt.axvline(threshold, color="r", linestyle="--", label=f"Threshold={threshold:.4f}")
        plt.title("再構成誤差分布")
        plt.xlabel("再構成誤差")
        plt.ylabel("件数")
        plt.legend()
        plt.savefig("output/reconstruction_error.png")
        plt.close()
        if MLFLOW_CONFIG["use_mlflow"]:
            mlflow.log_artifact("output/reconstruction_error.png")

        for idx, is_predicted_anomaly in enumerate(predicted_anomalies):
            label_name = class_names[labels[idx]]
            orig, recon, diff_map = images[idx], reconstructions[idx], (images[idx] - reconstructions[idx])**2
            save_dir = f"output/anomaly_samples/{label_name}"
            os.makedirs(save_dir, exist_ok=True)

            fig, ax = plt.subplots(1,3,figsize=(10,4))
            cmap = "gray" if DATA_CONFIG["grayscale"] else None
            ax[0].imshow(orig.transpose(1,2,0) if not DATA_CONFIG["grayscale"] else orig, cmap=cmap)
            ax[0].set_title("元画像")
            ax[0].axis("off")
            ax[1].imshow(recon.transpose(1,2,0) if not DATA_CONFIG["grayscale"] else recon, cmap=cmap)
            ax[1].set_title("再構成画像")
            ax[1].axis("off")
            ax[2].imshow(orig.transpose(1,2,0) if not DATA_CONFIG["grayscale"] else orig, cmap=cmap)
            ax[2].imshow(diff_map.transpose(1,2,0) if not DATA_CONFIG["grayscale"] else diff_map, cmap="jet", alpha=0.5)
            ax[2].set_title("誤差ヒートマップ")
            ax[2].axis("off")
            plt.tight_layout()
            img_path = f"{save_dir}/sample_{idx}.png"
            plt.savefig(img_path)
            plt.close()
            if MLFLOW_CONFIG["use_mlflow"]:
                mlflow.log_artifact(img_path)

from contextlib import contextmanager
@contextmanager
def dummy_context():
    yield

if __name__ == "__main__":
    evaluate()
