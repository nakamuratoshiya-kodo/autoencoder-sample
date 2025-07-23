import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tqdm import tqdm


# ===== データ拡張関数 =====
def horizontal_flip(image):
    return cv2.flip(image, 1)

def vertical_flip(image):
    return cv2.flip(image, 0)

def rotate_with_white_background(image, angle, max_angle=30):
    """背景を白で回転"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # 新しい画像サイズを計算
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # 回転行列の平行移動を修正
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]

    # 背景色を白に指定
    rotated = cv2.warpAffine(image, M, (nW, nH), borderValue=(255, 255, 255))
    return rotated

def random_brightness_contrast(image, alpha_range=(0.8, 1.2), beta_range=(-30, 30)):
    alpha = np.random.uniform(*alpha_range)  # コントラスト
    beta = np.random.randint(*beta_range)    # 明度
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def add_gaussian_noise(image, mean=0, sigma=10):
    noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    return cv2.add(image, noise)


# ===== GUIアプリ =====
class AugmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("データセット水増しアプリ（白背景回転対応）")
        self.root.geometry("500x300")

        # 元フォルダと保存先フォルダ
        self.source_dir = tk.StringVar()
        self.output_dir = tk.StringVar()

        # オプション
        self.option_flip_horizontal = tk.BooleanVar(value=True)
        self.option_flip_vertical = tk.BooleanVar()
        self.option_rotate = tk.BooleanVar(value=True)
        self.option_brightness_contrast = tk.BooleanVar()
        self.option_noise = tk.BooleanVar()

        self.build_ui()

    def build_ui(self):
        tk.Label(self.root, text="データセットフォルダ:").pack(anchor="w", padx=10, pady=(10, 0))
        tk.Entry(self.root, textvariable=self.source_dir, width=60).pack(padx=10, fill="x")
        tk.Button(self.root, text="フォルダ選択", command=self.select_source_dir).pack(pady=5)

        tk.Label(self.root, text="保存先フォルダ:").pack(anchor="w", padx=10, pady=(10, 0))
        tk.Entry(self.root, textvariable=self.output_dir, width=60).pack(padx=10, fill="x")
        tk.Button(self.root, text="フォルダ選択", command=self.select_output_dir).pack(pady=5)

        # オプション
        tk.Label(self.root, text="水増しオプション:").pack(anchor="w", padx=10, pady=(10, 0))
        tk.Checkbutton(self.root, text="左右反転", variable=self.option_flip_horizontal).pack(anchor="w", padx=20)
        tk.Checkbutton(self.root, text="上下反転", variable=self.option_flip_vertical).pack(anchor="w", padx=20)
        tk.Checkbutton(self.root, text="ランダム回転 (白背景)", variable=self.option_rotate).pack(anchor="w", padx=20)
        tk.Checkbutton(self.root, text="明度・コントラスト補正", variable=self.option_brightness_contrast).pack(anchor="w", padx=20)
        tk.Checkbutton(self.root, text="ノイズ追加", variable=self.option_noise).pack(anchor="w", padx=20)

        tk.Button(self.root, text="水増し開始", command=self.start_augmentation, bg="lightgreen").pack(pady=15)

    def select_source_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.source_dir.set(path)

    def select_output_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.output_dir.set(path)

    def start_augmentation(self):
        source = self.source_dir.get()
        output = self.output_dir.get()

        if not source or not output:
            messagebox.showwarning("警告", "元フォルダと保存先フォルダを指定してください。")
            return

        if not os.path.exists(source):
            messagebox.showerror("エラー", "指定された元フォルダが存在しません。")
            return

        # 処理開始
        total_images = []
        for root_dir, _, files in os.walk(source):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                    total_images.append(os.path.join(root_dir, file))

        if not total_images:
            messagebox.showinfo("情報", "元フォルダに画像が見つかりませんでした。")
            return

        os.makedirs(output, exist_ok=True)

        for img_path in tqdm(total_images, desc="水増し中"):
            img = cv2.imread(img_path)
            base_name = os.path.splitext(os.path.basename(img_path))[0]

            save_dir = os.path.join(output, os.path.relpath(os.path.dirname(img_path), source))
            os.makedirs(save_dir, exist_ok=True)

            # 元画像を保存
            cv2.imwrite(os.path.join(save_dir, base_name + "_orig.jpg"), img)

            # 水増し
            if self.option_flip_horizontal.get():
                cv2.imwrite(os.path.join(save_dir, base_name + "_hflip.jpg"), horizontal_flip(img))
            if self.option_flip_vertical.get():
                cv2.imwrite(os.path.join(save_dir, base_name + "_vflip.jpg"), vertical_flip(img))
            if self.option_rotate.get():
                rotated_img = rotate_with_white_background(img, np.random.uniform(-30, 30))
                cv2.imwrite(os.path.join(save_dir, base_name + "_rotate.jpg"), rotated_img)
            if self.option_brightness_contrast.get():
                cv2.imwrite(os.path.join(save_dir, base_name + "_brightcont.jpg"), random_brightness_contrast(img))
            if self.option_noise.get():
                cv2.imwrite(os.path.join(save_dir, base_name + "_noise.jpg"), add_gaussian_noise(img))

        messagebox.showinfo("完了", "水増し処理が完了しました。")


if __name__ == "__main__":
    root = tk.Tk()
    app = AugmentationApp(root)
    root.mainloop()
