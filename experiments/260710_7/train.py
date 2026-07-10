import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.models as models
from pathlib import Path
import torchvision.transforms.functional as TF

# 既存の共通ユーティリティ関数
from expt_thu_eact_50_chl.utils import (
    measure_model_complexity,
    measure_inference_latency,
    calculate_topk_accuracy,
    save_best_model,
)

# 自身の実験フォルダ内の前処理データを動的に参照
CURRENT_DIR = Path(__file__).parent.resolve()
PROCESSED_DIR = CURRENT_DIR / "processed_data"

MODEL_SAVE_PATH = CURRENT_DIR / "best_model_augmented.pth"
LOG_FILENAME = CURRENT_DIR / "result_augmented.txt"


# ====================================================
# 1. 専用データセットクラス（origもaugも自動で両方読み込みます）
# ====================================================
class HoloEvDataset(Dataset):

    def __init__(self, mode="train"):
        self.dir_path = PROCESSED_DIR / mode
        # _orig_ と _aug_ の両方の形式を一括キャッチ
        self.file_paths = glob.glob(os.path.join(self.dir_path, "*.npy"))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label_str = file_path.split("_label_")[-1].split(".npy")[0]
        label = int(label_str.replace("A", ""))

        features = np.load(file_path)

        # 最大値による正規化
        max_val = np.max(np.abs(features))
        if max_val > 0:
            features = features / max_val

        # NumPyからPyTorchのテンソルに変換 (形状はおそらく [4, 224, 260])
        features_tensor = torch.tensor(features, dtype=torch.float32)

        # ==========================================
        # ★ここからが「オンザフライ（リアルタイム）拡張」
        # ==========================================
        # 訓練モード（train）の時だけ、ランダムに変形をかけます（テスト時はスルー）
        if self.mode == "train":
            
            # 1. 確率50%で左右反転
            if random.random() < 0.5:
                features_tensor = TF.hflip(features_tensor)

            # 2. 確率50%で0.8~1.2倍のズーム
            if random.random() < 0.5:
                zoom_factor = random.uniform(0.8, 1.2)
                orig_h, orig_w = features_tensor.shape[1], features_tensor.shape[2]
                
                # 新しいサイズを計算
                new_h = int(orig_h * zoom_factor)
                new_w = int(orig_w * zoom_factor)
                
                # 一旦リサイズ（拡大または縮小）
                features_resized = TF.resize(features_tensor, [new_h, new_w], antialias=True)
                
                if zoom_factor > 1.0:
                    # 拡大された場合は、中心部分を元のサイズに切り抜く (Crop)
                    features_tensor = TF.center_crop(features_resized, [orig_h, orig_w])
                else:
                    # 縮小された場合は、周りを0でパディングして元のサイズに戻す (Pad)
                    pad_h = (orig_h - new_h) // 2
                    pad_w = (orig_w - new_w) // 2
                    # 上下左右のパディング量を指定
                    features_tensor = TF.pad(features_resized, [pad_w, pad_h, orig_w - new_w - pad_w, orig_h - new_h - pad_h], fill=0)

            # 3. 確率50%で10%~30%のイベント（要素）をランダムに間引く（ゼロにする）
            if random.random() < 0.5:
                drop_rate = random.uniform(0.1, 0.3)
                # 特徴量と同じ形のランダムなマスク（0か1）を作成
                mask = torch.rand_like(features_tensor) > drop_rate
                features_tensor = features_tensor * mask

        return features_tensor, torch.tensor(label, dtype=torch.long)

# ====================================================
# 2. 空間アテンションのみのモジュール
# ====================================================
class SpatialAttentionOnly(nn.Module):

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        max_avg = torch.cat([avg_out, max_out], dim=1)

        attention_map = self.sigmoid(self.conv1(max_avg))
        return x * attention_map


# ====================================================
# 3. Global Spectral Gating (GSG) モジュール
# ====================================================
class GlobalSpectralGating(nn.Module):

    def __init__(self, channels, T_prime, H_prime):
        super().__init__()
        self.dw_conv = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )
        self.ln = nn.LayerNorm([channels, T_prime, H_prime])

        self.weight_real = nn.Parameter(
            torch.randn(channels, T_prime, H_prime // 2 + 1, dtype=torch.float32)
            * 0.02
        )
        self.weight_imag = nn.Parameter(
            torch.randn(channels, T_prime, H_prime // 2 + 1, dtype=torch.float32)
            * 0.02
        )
        self.gate_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        x_local = self.dw_conv(x)

        orig_dtype = x_local.dtype
        x_local_fp32 = x_local.float()

        x_freq = torch.fft.rfft2(x_local_fp32, dim=(2, 3), norm="ortho")

        complex_weight = torch.complex(self.weight_real, self.weight_imag)
        x_freq = x_freq * complex_weight

        z_tilde = torch.fft.irfft2(
            x_freq, s=(x.size(2), x.size(3)), dim=(2, 3), norm="ortho"
        )
        z_tilde = z_tilde.to(orig_dtype)

        carrier = F.silu(self.ln(z_tilde))
        gate = torch.sigmoid(self.gate_conv(z_tilde))
        return carrier * gate


# ====================================================
# 4. HoloEv-Net V4.1 (MobileNetV2ベース・エッジ最適化版)
# ====================================================
class HoloEvNetBaseV4_MobileNetV2(nn.Module):

    def __init__(self, num_classes=50, use_pretrained=True):
        super().__init__()
        
        # MobileNetV2をベースとして用意（use_pretrained=Trueで事前学習重みを使用）
        weights = models.MobileNet_V2_Weights.DEFAULT if use_pretrained else None
        mb2 = models.mobilenet_v2(weights=weights)

        # 【カスタム1】入力層をイベントカメラの4チャンネルに対応させる
        # 元のmb2.features[0][0]は3チャンネル(RGB)用なので、4チャンネル用(32出力)に作り直す
        self.conv0 = nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn0 = mb2.features[0][1]
        self.relu0 = mb2.features[0][2]

        # 【カスタム2】解像度が 14x17 になるまでの層をひとまとめにする (featuresの1〜13番目)
        # ※ResNet18の「layer3」に相当するステージ
        self.stage1 = nn.Sequential(*mb2.features[1:14])  # 出力は96チャンネル
        self.spatial3 = SpatialAttentionOnly(kernel_size=7)

        # 【カスタム3】解像度が 7x9 になるまでの層をひとまとめにする (featuresの14〜17番目)
        # ※ResNet18の「layer4」に相当するステージ
        self.stage2 = nn.Sequential(*mb2.features[14:18])  # 出力は320チャンネル
        self.spatial4 = SpatialAttentionOnly(kernel_size=7)

        # 【カスタム4】GSGモジュールのチャンネル数をMobileNetに合わせて「320」に変更
        self.gsg = GlobalSpectralGating(channels=320, T_prime=7, H_prime=9)

        # MobileNetV2の最終1x1 Conv層（特徴量を1280チャンネルに引き上げる層）
        self.conv_last = mb2.features[18]

        # 出口部分（グローバルアベレージプーリングと分類器）
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, v_tensor):
        # 最初の層 (1/2にダウンサンプリング)
        x = self.conv0(v_tensor)
        x = self.bn0(x)
        x = self.relu0(x)

        # ステージ1 (解像度: 14x17, チャンネル数: 96)
        x = self.stage1(x)
        x = self.spatial3(x)

        # ステージ2 (解像度: 7x9, チャンネル数: 320)
        x_in = self.stage2(x)
        x_in = self.spatial4(x_in)

        # 独自モジュール GSG との残差接続
        x_gsg = self.gsg(x_in)
        x_in = x_in + x_gsg

        # 最終特徴抽出 (解像度: 7x9, チャンネル数: 1280)
        x_out = self.conv_last(x_in)

        # 分類
        feat = self.avgpool(x_out).view(x_out.size(0), -1)
        return self.classifier(feat)


# ====================================================
# 5. 学習ループ
# ====================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    INPUT_SIZE = (1, 4, 224, 260)
    NUM_CLASSES = 50
    num_epochs = 100

    model = HoloEvNetBaseV4_MobileNetV2(num_classes=NUM_CLASSES, use_pretrained=True).to(device)

    macs, params = measure_model_complexity(
        model, input_size=INPUT_SIZE, device=device
    )
    latency_ms = measure_inference_latency(
        model, input_size=INPUT_SIZE, device=device
    )

    train_dataset = HoloEvDataset(mode="train")
    test_dataset = HoloEvDataset(mode="test")

    # 拡張でデータ数が2倍になっているため、高効率な並列読み込みを設定
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=0
    )

    scaler = torch.amp.GradScaler()
    best_test_acc1 = 0.0

    with open(LOG_FILENAME, "w", encoding="utf-8") as f:
        f.write("=== THUE-ACT-50 CHL (Data Augmented Pipeline) ===\n")
        if params is not None:
            f.write(f"Model Params: {params/1e6:.2f} M\n")
            f.write(f"Model FLOPs: {macs/1e9:.2f} G\n")
        f.write(f"Inference Latency: {latency_ms:.2f} ms\n")
        f.write("=================================================\n")

        print(f"🚀 ステップ1：データ拡張（生イベント空間）の検証を開始します")

        for epoch in range(num_epochs):
            model.train()
            train_loss, train_total = 0.0, 0
            train_top1, train_top5 = 0.0, 0.0

            current_lr = optimizer.param_groups[0]["lr"]

            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.amp.autocast(device_type=device.type):
                    outputs = model(features)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item() * features.size(0)
                train_total += labels.size(0)
                acc1, acc5 = calculate_topk_accuracy(outputs, labels, topk=(1, 5))
                train_top1 += acc1
                train_top5 += acc5

            scheduler.step()

            model.eval()
            test_total = 0
            test_top1, test_top5 = 0.0, 0.0

            with torch.no_grad():
                for features, labels in test_loader:
                    features, labels = features.to(device), labels.to(device)
                    with torch.amp.autocast(device_type=device.type):
                        outputs = model(features)

                    test_total += labels.size(0)
                    acc1, acc5 = calculate_topk_accuracy(
                        outputs, labels, topk=(1, 5)
                    )
                    test_top1 += acc1
                    test_top5 += acc5

            tr_acc1_pct = (train_top1 / train_total) * 100
            tr_acc5_pct = (train_top5 / train_total) * 100
            te_acc1_pct = (test_top1 / test_total) * 100
            te_acc5_pct = (test_top5 / test_total) * 100

            epoch_status = (
                f"Epoch {epoch+1:03d}/{num_epochs} [LR: {current_lr:.6f}] -> Loss: {train_loss/train_total:.4f} | "
                f"Train Top-1: {tr_acc1_pct:.2f}% (Top-5: {tr_acc5_pct:.2f}%)| "
                f"★Test Top-1: {te_acc1_pct:.2f}% (Top-5: {te_acc5_pct:.2f}%)"
            )
            print(epoch_status)
            f.write(epoch_status + "\n")

            best_test_acc1 = save_best_model(
                model, te_acc1_pct, best_test_acc1, MODEL_SAVE_PATH
            )
            f.flush()

    print(
        f"\n🎉 ステップ1の学習が完了しました！最高の Test Top-1 精度は {best_test_acc1:.2f}% でした。"
    )