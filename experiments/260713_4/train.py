import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.models as models
from pathlib import Path

# 既存の共通ユーティリティ関数
from expt_thu_eact_50_chl.utils import (
    measure_model_complexity,
    measure_inference_latency,
    calculate_topk_accuracy,
    save_best_model,
)

# 自身の実験フォルダ（260713_4）内の前処理データを自動動的参照
CURRENT_DIR = Path(__file__).parent.resolve()
PROCESSED_DIR = CURRENT_DIR / "processed_data"

MODEL_SAVE_PATH = CURRENT_DIR / "best_model_augmented.pth"
LOG_FILENAME = CURRENT_DIR / "result_augmented.txt"


# ====================================================
# 1. 2ストリーム対応データセットクラス
# ====================================================
class HoloEvTwoStreamDataset(Dataset):

    def __init__(self, mode="train"):
        self.dir_path = PROCESSED_DIR / mode
        self.file_paths = glob.glob(os.path.join(str(self.dir_path), "*.npy"))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label_str = file_path.split("_label_")[-1].split(".npy")[0]
        label = int(label_str.replace("A", ""))

        # (8, 224, 260) 形状の統合特徴をロード
        features = np.load(file_path)

        # グローバル(0~3ch)とローカル(4~7ch)に分離して正規化
        feat_global = features[:4]
        feat_local = features[4:]

        max_g = np.max(np.abs(feat_global))
        if max_g > 0: feat_global = feat_global / max_g

        max_l = np.max(np.abs(feat_local))
        if max_l > 0: feat_local = feat_local / max_l

        return (
            torch.tensor(feat_global, dtype=torch.float32),
            torch.tensor(feat_local, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )


# ====================================================
# 2. 空間アテンションモジュール
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
            torch.randn(channels, T_prime, H_prime // 2 + 1, dtype=torch.float32) * 0.02
        )
        self.weight_imag = nn.Parameter(
            torch.randn(channels, T_prime, H_prime // 2 + 1, dtype=torch.float32) * 0.02
        )
        self.gate_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        x_local = self.dw_conv(x)
        orig_dtype = x_local.dtype
        x_local_fp32 = x_local.float()

        x_freq = torch.fft.rfft2(x_local_fp32, dim=(2, 3), norm="ortho")
        complex_weight = torch.complex(self.weight_real, self.weight_imag)
        x_freq = x_freq * complex_weight

        z_tilde = torch.fft.irfft2(x_freq, s=(x.size(2), x.size(3)), dim=(2, 3), norm="ortho")
        z_tilde = z_tilde.to(orig_dtype)

        carrier = F.silu(self.ln(z_tilde))
        gate = torch.sigmoid(self.gate_conv(z_tilde))
        return carrier * gate


# ====================================================
# 4. 脳型二流・動的ゲートネットワーク（HoloEv-Net V5.0）
# ====================================================
class HoloEvNetTwoStreamGated(nn.Module):

    def __init__(self, num_classes=50):
        super().__init__()
        
        # ─── 1. グローバル・ストリーム (大局運動解折) ───
        resnet_g = models.resnet18(weights=None)
        self.g_conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.g_bn1 = resnet_g.bn1
        self.g_relu = resnet_g.relu
        self.g_maxpool = resnet_g.maxpool
        self.g_layer1 = resnet_g.layer1
        self.g_layer2 = resnet_g.layer2
        self.g_layer3 = resnet_g.layer3
        self.g_spatial3 = SpatialAttentionOnly(kernel_size=7)
        self.g_layer4 = resnet_g.layer4
        self.g_spatial4 = SpatialAttentionOnly(kernel_size=7)
        self.g_gsg = GlobalSpectralGating(channels=512, T_prime=7, H_prime=9)
        self.g_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.g_classifier = nn.Linear(512, num_classes)

        # ─── 2. ローカル・ストリーム (手元形状幾何解析) ───
        resnet_l = models.resnet18(weights=None)
        self.l_conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.l_bn1 = resnet_l.bn1
        self.l_relu = resnet_l.relu
        self.l_maxpool = resnet_l.maxpool
        self.l_layer1 = resnet_l.layer1
        self.l_layer2 = resnet_l.layer2
        self.l_layer3 = resnet_l.layer3
        self.l_layer4 = resnet_l.layer4
        self.l_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.l_classifier = nn.Linear(512, num_classes)

        # ─── 3. 動的コンテクスト・ゲートレイヤー ───
        # グローバルな文脈[512次元]を基に、各ストリームの寄与度(2チャネル)を動的生成
        self.gating_layer = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x_global, x_local):
        # --- ① グローバルフォワード ---
        g = self.g_conv1(x_global)
        g = self.g_bn1(g)
        g = self.g_relu(g)
        g = self.g_maxpool(g)
        g = self.g_layer1(g)
        g = self.g_layer2(g)
        g = self.g_layer3(g)
        g = self.g_spatial3(g)
        g_in = self.g_layer4(g)
        g_in = self.g_spatial4(g_in)
        g_gsg = self.g_gsg(g_in)
        g_out = g_in + g_gsg
        
        # ゲートの判断基準となるグローバル特徴
        feat_global = self.g_avgpool(g_out).view(g_out.size(0), -1)
        y_global = self.g_classifier(feat_global)

        # --- ② ローカル（クラスター）フォワード ---
        l = self.l_conv1(x_local)
        l = self.l_bn1(l)
        l = self.l_relu(l)
        l = self.l_maxpool(l)
        l = self.l_layer1(l)
        l = self.l_layer2(l)
        l = self.l_layer3(l)
        l = self.l_layer4(l)
        
        feat_local = self.l_avgpool(l).view(l.size(0), -1)
        y_local = self.l_classifier(feat_local)

        # --- ③ 動的線形結合マージ（Gating Fusion） ---
        # 形状: (batch_size, 2) -> 各サンプルの信頼度の比率
        gate_logits = self.gating_layer(feat_global)
        gate_weights = F.softmax(gate_logits, dim=1)
        
        alpha_global = gate_weights[:, 0:1]
        alpha_local = gate_weights[:, 1:2]

        # 動的重みによるインテリジェント予測値の統合
        y_final = alpha_global * y_global + alpha_local * y_local
        
        return y_final, alpha_global, alpha_local


# ====================================================
# 5. 学習・評価パイプライン
# ====================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    NUM_CLASSES = 50
    num_epochs = 100

    model = HoloEvNetTwoStreamGated(num_classes=NUM_CLASSES).to(device)

    train_dataset = HoloEvTwoStreamDataset(mode="train")
    test_dataset = HoloEvTwoStreamDataset(mode="test")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

    scaler = torch.amp.GradScaler()
    best_test_acc1 = 0.0

    with open(LOG_FILENAME, "w", encoding="utf-8") as f:
        f.write("=== THUE-ACT-50 CHL (Two-Stream Dynamic Gating Network) ===\n")
        f.write("============================================================\n")

        print(f"🚀 粗密二流・動的ゲート融合ネットワーク（V5.0）の学習を開始します")

        for epoch in range(num_epochs):
            model.train()
            train_loss, train_total = 0.0, 0
            train_top1, train_top5 = 0.0, 0.0

            current_lr = optimizer.param_groups[0]["lr"]

            for x_g, x_l, labels in train_loader:
                x_g, x_l, labels = x_g.to(device), x_l.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.amp.autocast(device_type=device.type):
                    outputs, _, _ = model(x_g, x_l)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item() * labels.size(0)
                train_total += labels.size(0)
                acc1, acc5 = calculate_topk_accuracy(outputs, labels, topk=(1, 5))
                train_top1 += acc1
                train_top5 += acc5

            scheduler.step()

            # --- Validation（テスト）フェーズ ---
            model.eval()
            test_total = 0
            test_top1, test_top5 = 0.0, 0.0
            
            # 重要パラメータ alpha_global, alpha_local の追跡用リスト
            epoch_alpha_g_list = []
            epoch_alpha_l_list = []

            with torch.no_grad():
                for x_g, x_l, labels in test_loader:
                    x_g, x_l, labels = x_g.to(device), x_l.to(device), labels.to(device)
                    with torch.amp.autocast(device_type=device.type):
                        outputs, a_g, a_l = model(x_g, x_l)

                    test_total += labels.size(0)
                    acc1, acc5 = calculate_topk_accuracy(outputs, labels, topk=(1, 5))
                    test_top1 += acc1
                    test_top5 += acc5
                    
                    # ゲート重みの推移をバッチごとに保存
                    epoch_alpha_g_list.append(a_g.cpu().numpy().mean())
                    epoch_alpha_l_list.append(a_l.cpu().numpy().mean())

            tr_acc1_pct = (train_top1 / train_total) * 100
            tr_acc5_pct = (train_top5 / train_total) * 100
            te_acc1_pct = (test_top1 / test_total) * 100
            te_acc5_pct = (test_top5 / test_total) * 100
            
            # テストデータ全体を通じた、マクロ重み・ミクロ重みの平均値を算出
            avg_alpha_g = np.mean(epoch_alpha_g_list)
            avg_alpha_l = np.mean(epoch_alpha_l_list)

            epoch_status = (
                f"Epoch {epoch+1:03d}/{num_epochs} [LR: {current_lr:.6f}] -> Loss: {train_loss/train_total:.4f} | "
                f"Train Top-1: {tr_acc1_pct:.2f}% | ★Test Top-1: {te_acc1_pct:.2f}% (Top-5: {te_acc5_pct:.2f}%) | "
                f"[G-Gate: {avg_alpha_g:.3f} / L-Gate: {avg_alpha_l:.3f}]"
            )
            print(epoch_status)
            f.write(epoch_status + "\n")

            best_test_acc1 = save_best_model(model, te_acc1_pct, best_test_acc1, MODEL_SAVE_PATH)
            f.flush()

    print(f"\n🎉 学習完了しました！最高の Test Top-1 精度は {best_test_acc1:.2f}% でした。")