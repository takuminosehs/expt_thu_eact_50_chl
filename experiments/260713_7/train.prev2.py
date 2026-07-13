import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.models as models
from pathlib import Path
import math

from expt_thu_eact_50_chl.utils import (
    calculate_topk_accuracy,
    save_best_model,
)

# ====================================================
# ─── ⚙️ ハイパーパラメータ設定（後から自由調整可能） ───
# ====================================================
STAGE1_EPOCHS = 50  # 1. グローバルCNN単独育成期間 (0〜49)
STAGE2_EPOCHS = 50  # 2. グローバル固定・ローカル＆ゲート連動最適化期間 (50〜99)
STAGE3_EPOCHS = 50  # 3. 全パラメータ完全同期融合期間 (100〜149)
# ====================================================

NUM_EPOCHS = STAGE1_EPOCHS + STAGE2_EPOCHS + STAGE3_EPOCHS
CURRENT_DIR = Path(__file__).parent.resolve()
PROCESSED_DIR = CURRENT_DIR / "processed_data"

MODEL_SAVE_PATH = CURRENT_DIR / "best_model_augmented.pth"
LOG_FILENAME = CURRENT_DIR / "result_augmented.txt"

class HoloEvTwinFolderDataset(Dataset):
    def __init__(self, mode="train"):
        self.global_dir = PROCESSED_DIR / mode / "global"
        self.local_dir = PROCESSED_DIR / mode / "local"
        self.file_names = [os.path.basename(p) for p in glob.glob(os.path.join(str(self.global_dir), "*.npy"))]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        f_name = self.file_names[idx]
        label_str = f_name.split("_label_")[-1].split(".npy")[0]
        label = int(label_str.replace("A", ""))

        feat_g = np.load(str(self.global_dir / f_name))
        feat_l = np.load(str(self.local_dir / f_name))

        # 最大値正規化
        mg = np.max(np.abs(feat_g))
        if mg > 0: feat_g = feat_g / mg
        ml = np.max(np.abs(feat_l))
        if ml > 0: feat_l = feat_l / ml

        return (
            torch.tensor(feat_g, dtype=torch.float32),
            torch.tensor(feat_l, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )

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
        return x * self.sigmoid(self.conv1(max_avg))

class GlobalSpectralGating(nn.Module):
    def __init__(self, channels, T_prime, H_prime):
        super().__init__()
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.ln = nn.LayerNorm([channels, T_prime, H_prime])
        self.weight_real = nn.Parameter(torch.randn(channels, T_prime, H_prime // 2 + 1) * 0.02)
        self.weight_imag = nn.Parameter(torch.randn(channels, T_prime, H_prime // 2 + 1) * 0.02)
        self.gate_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        x_local = self.dw_conv(x)
        x_freq = torch.fft.rfft2(x_local.float(), dim=(2, 3), norm="ortho")
        x_freq = x_freq * torch.complex(self.weight_real, self.weight_imag)
        z_tilde = torch.fft.irfft2(x_freq, s=(x.size(2), x.size(3)), dim=(2, 3), norm="ortho").to(x.dtype)
        return F.silu(self.ln(z_tilde)) * torch.sigmoid(self.gate_conv(z_tilde))

class HoloEvNetThreeStageGated(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        
        # ① グローバル・ストリーム (現行の洗練されたマクロ移動解析構造)
        resnet_g = models.resnet18(weights=None)
        self.g_conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.g_bn1 = resnet_g.bn1
        self.g_relu = resnet_g.relu
        self.g_maxpool = resnet_g.maxpool
        self.g_layer1 = resnet_g.layer1
        self.g_layer2 = resnet_g.layer2
        self.g_layer3 = resnet_g.layer3
        # self.g_spatial3 = SpatialAttentionOnly(kernel_size=7)
        self.g_layer4 = resnet_g.layer4
        self.g_spatial4 = SpatialAttentionOnly(kernel_size=7)
        self.g_gsg = GlobalSpectralGating(channels=512, T_prime=7, H_prime=9)
        self.g_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.g_classifier = nn.Linear(512, num_classes)

        # ② 新・ローカル・ストリーム (ピュア幾何形状2D-CNN)
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

        # ③ 動的コンテクスト・ゲートレイヤー
        self.gating_layer = nn.Sequential(nn.Linear(512, 64), nn.ReLU(), nn.Linear(64, 2))
        self._init_gating_weights()

    def _init_gating_weights(self):
        for m in self.gating_layer:
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x_global, x_local, mode='both'):
        # --- グローバル特徴抽出（ステージ1, 2, 3共通で常に実施） ---
        g = self.g_conv1(x_global)
        g = self.g_bn1(g)
        g = self.g_relu(g)
        g = self.g_maxpool(g)
        g = self.g_layer1(g)
        g = self.g_layer2(g)
        g = self.g_layer3(g)
        # g = self.g_spatial3(g)
        g_in = self.g_layer4(g)
        g_in = self.g_spatial4(g_in)
        g_out = g_in + self.g_gsg(g_in)
        feat_global = self.g_avgpool(g_out).view(g_out.size(0), -1)
        y_global = self.g_classifier(feat_global)

        # 【本質修正①】STAGE-1の時はゲート計算をバイパスし、予測は純粋なグローバルのみで行う
        if mode == 'global_only':
            alpha_g = torch.ones((x_global.size(0), 1), device=x_global.device)
            alpha_l = torch.zeros((x_global.size(0), 1), device=x_global.device)
            return y_global, alpha_g, alpha_l

        # --- ローカル特徴抽出（ステージ2, 3共通） ---
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

        # --- 動的ゲートマージ ---
        gate_logits = self.gating_layer(feat_global)
        gate_weights = F.softmax(gate_logits, dim=1)
        alpha_g, alpha_l = gate_weights[:, 0:1], gate_weights[:, 1:2]
        
        # 最終予測値（グローバルとローカルをゲート比率でマージ）
        y_final = alpha_g * y_global + alpha_l * y_local
        return y_final, alpha_g, alpha_l

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    model = HoloEvNetThreeStageGated(num_classes=50).to(device)
    train_loader = DataLoader(HoloEvTwinFolderDataset(mode="train"), batch_size=16, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(HoloEvTwinFolderDataset(mode="test"), batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    global_params, local_params, gating_params = [], [], []
    for n, p in model.named_parameters():
        if "gating_layer" in n: gating_params.append(p)
        elif n.startswith("g_"): global_params.append(p)
        elif n.startswith("l_"): local_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": global_params, "lr": 0.0, "weight_decay": 0.01},
        {"params": local_params, "lr": 0.0, "weight_decay": 0.01},
        {"params": gating_params, "lr": 0.0, "weight_decay": 0.0}
    ])

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler()
    best_test_acc1 = 0.0

    with open(LOG_FILENAME, "w", encoding="utf-8") as f:
        f.write("=== 3-Stage Dependent Reset Optimization Pipeline ===\\n")

        for epoch in range(NUM_EPOCHS):
            # ─── 🔄 【ユーザー指示の完全具現化】3段階独立LR数式 ───
            if epoch < STAGE1_EPOCHS:
                phase, mode = "STAGE-1 (Global-Train)", "global_only"
                lr = 0.0003 * (1 + math.cos(math.pi * epoch / STAGE1_EPOCHS)) / 2
                # ゲートの値は放置（LR=0）、グローバルのみを更新
                optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"], optimizer.param_groups[2]["lr"] = lr, 0.0, 0.0
            
            elif epoch < STAGE1_EPOCHS + STAGE2_EPOCHS:
                # 【本質修正②】modeを 'both' に設定し、グローバル(固定)とローカル(動的)を結合してフォワード
                phase, mode = "STAGE-2 (Gate&Local-Train)", "both"
                stage_epoch = epoch - STAGE1_EPOCHS
                lr = 0.0003 * (1 + math.cos(math.pi * stage_epoch / STAGE2_EPOCHS)) / 2
                # グローバルの重みはすべて固定 (LR=0.0)。ローカルの重みとGateの値のみを更新
                optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"], optimizer.param_groups[2]["lr"] = 0.0, lr, lr
            
            else:
                # 【本質修正③】リセットを挟まず、ステージ2の最適化を引き継いで一斉微調整へ突入
                phase, mode = "STAGE-3 (All-Synch)", "both"
                stage_epoch = epoch - STAGE1_EPOCHS - STAGE2_EPOCHS
                lr = 0.0003 * (1 + math.cos(math.pi * stage_epoch / STAGE3_EPOCHS)) / 2
                # すべての固定を外して一斉に更新
                optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"], optimizer.param_groups[2]["lr"] = lr, lr, lr

            model.train()
            train_loss, train_total = 0.0, 0
            train_top1 = 0.0

            for x_g, x_l, labels in train_loader:
                x_g, x_l, labels = x_g.to(device), x_l.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast(device_type=device.type):
                    outputs, _, _ = model(x_g, x_l, mode=mode)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item() * labels.size(0)
                train_total += labels.size(0)
                acc1, _ = calculate_topk_accuracy(outputs, labels, topk=(1, 5))
                train_top1 += acc1

            # --- Validation フェーズ ---
            model.eval()
            test_total, test_top1 = 0, 0.0
            epoch_alpha_g, epoch_alpha_l = [], []

            with torch.no_grad():
                for x_g, x_l, labels in test_loader:
                    x_g, x_l, labels = x_g.to(device), x_l.to(device), labels.to(device)
                    with torch.amp.autocast(device_type=device.type):
                        # テスト時も訓練時のモードと完全に同期
                        outputs, a_g, a_l = model(x_g, x_l, mode=mode)
                    test_total += labels.size(0)
                    acc1, _ = calculate_topk_accuracy(outputs, labels, topk=(1, 5))
                    test_top1 += acc1
                    epoch_alpha_g.append(a_g.cpu().numpy().mean())
                    epoch_alpha_l.append(a_l.cpu().numpy().mean())

            tr_acc = (train_top1 / train_total) * 100
            te_acc = (test_top1 / test_total) * 100
            avg_a_g, avg_a_l = np.mean(epoch_alpha_g), np.mean(epoch_alpha_l)

            status = (
                f"Epoch {epoch+1:03d} [{phase} | LR: {lr:.6f}] -> Loss: {train_loss/train_total:.4f} | "
                f"Train: {tr_acc:.2f}% | ★Test: {te_acc:.2f}% | [G-Gate: {avg_a_g:.3f} / L-Gate: {avg_a_l:.3f}]"
            )
            print(status)
            f.write(status + "\\n")
            best_test_acc1 = save_best_model(model, te_acc, best_test_acc1, MODEL_SAVE_PATH)
            f.flush()