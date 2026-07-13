import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.models as models
from pathlib import Path
import math  # 2段階学習率の手動数式制御のために追加

# 既存の共通ユーティリティ関数
from expt_thu_eact_50_chl.utils import (
    measure_model_complexity,
    measure_inference_latency,
    calculate_topk_accuracy,
    save_best_model,
)

# 自身の実験フォルダ（260713_4）内の前処理データを動的参照
CURRENT_DIR = Path(__file__).parent.resolve()
PROCESSED_DIR = CURRENT_DIR.parent / "260713_4" / "processed_data"

# 出力先を CURRENT_DIR 直下に固定
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
# 4. 二流・動的ゲートネットワーク（HoloEv-Net V5.3）
# ====================================================
class HoloEvNetTwoStreamGated(nn.Module):

    def __init__(self, num_classes=50):
        super().__init__()
        
        # ─── ① グローバル・ストリーム ───
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

        # ─── ② ローカル・ストリーム ───
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

        # ─── ③ 動的コンテクスト・ゲートレイヤー ───
        self.gating_layer = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        self._init_gating_weights()

    def _init_gating_weights(self):
        for m in self.gating_layer:
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x_global, x_local, global_only=False):
        # グローバルフォワード
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
        g_gsg = self.g_gsg(g_in)
        g_out = g_in + g_gsg
        
        feat_global = self.g_avgpool(g_out).view(g_out.size(0), -1)
        y_global = self.g_classifier(feat_global)

        # 最初の50エポックはローカル計算を完全スキップし、G-Gate=1.0固定で返す
        if global_only:
            alpha_global = torch.ones((x_global.size(0), 1), device=x_global.device)
            alpha_local = torch.zeros((x_global.size(0), 1), device=x_global.device)
            return y_global, alpha_global, alpha_local

        # ローカルフォワード（50エポック以降に解放）
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

        # 動的ゲートマージ
        gate_logits = self.gating_layer(feat_global)
        gate_weights = F.softmax(gate_logits, dim=1)
        
        alpha_global = gate_weights[:, 0:1]
        alpha_local = gate_weights[:, 1:2]

        y_final = alpha_global * y_global + alpha_local * y_local
        
        return y_final, alpha_global, alpha_local


# ====================================================
# 5. メイン・評価独立スケジュールパイプライン
# ====================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    NUM_CLASSES = 50
    num_epochs = 100
    FREEZE_EPOCHS = 50  # 前半50エポックはグローバルのみを育成

    model = HoloEvNetTwoStreamGated(num_classes=NUM_CLASSES).to(device)

    train_dataset = HoloEvTwoStreamDataset(mode="train")
    test_dataset = HoloEvTwoStreamDataset(mode="test")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # ─── 【重要修正①】名前の先頭一致（排他的選別）により、二重登録のエラーを完全回避 ───
    global_params = []
    local_params = []
    gating_params = []

    for n, p in model.named_parameters():
        if "gating_layer" in n:
            gating_params.append(p)
        elif n.startswith("g_"):
            global_params.append(p)
        elif n.startswith("l_"):
            local_params.append(p)

    # オプティマイザの初期設定
    optimizer = torch.optim.AdamW([
        {"params": global_params, "lr": 0.0003, "weight_decay": 0.01},
        {"params": local_params, "lr": 0.0, "weight_decay": 0.01},
        {"params": gating_params, "lr": 0.0, "weight_decay": 0.0}
    ])

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler()
    best_test_acc1 = 0.0
    
    current_gating_lr = 0.0

    with open(LOG_FILENAME, "w", encoding="utf-8") as f:
        f.write("=== THUE-ACT-50 CHL (2-Stage Global-First Sequential Training) ===\n")
        f.write("===================================================================\n")

        print(f"🚀 【2段階シーケンシャル最適化】最初の {FREEZE_EPOCHS} エポックはグローバルストリームを集中育成します")

        for epoch in range(num_epochs):
            # ─── 【重要修正②】ややこしいPyTorchスケジューラを廃止し、数学的に直接LRを制御 ───
            # 基本のコサインアニーリングカーブを計算
            current_backbone_lr = 0.0003 * (1 + math.cos(math.pi * epoch / num_epochs)) / 2
            
            if epoch < FREEZE_EPOCHS:
                # 【フェーズ1】グローバルのみを更新、他は完全ロック(LR=0)
                is_global_only = True
                optimizer.param_groups[0]["lr"] = current_backbone_lr
                optimizer.param_groups[1]["lr"] = 0.0
                optimizer.param_groups[2]["lr"] = 0.0
                current_gating_lr = 0.0
            else:
                # 【フェーズ2】50エポック以降、ローカルとゲートを一気に解放
                is_global_only = False
                optimizer.param_groups[0]["lr"] = current_backbone_lr
                optimizer.param_groups[1]["lr"] = current_backbone_lr  # ローカルも同期合流
                
                # ゲートは50エポック目を 0.02 とし、以降毎エポック 0.90 倍ずつ急速ブレーキ
                current_gating_lr = 0.02 * (0.90 ** (epoch - FREEZE_EPOCHS))
                optimizer.param_groups[2]["lr"] = current_gating_lr

            model.train()
            train_loss, train_total = 0.0, 0
            train_top1, train_top5 = 0.0, 0.0

            for x_g, x_l, labels in train_loader:
                x_g, x_l, labels = x_g.to(device), x_l.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.amp.autocast(device_type=device.type):
                    outputs, _, _ = model(x_g, x_l, global_only=is_global_only)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item() * labels.size(0)
                train_total += labels.size(0)
                acc1, acc5 = calculate_topk_accuracy(outputs, labels, topk=(1, 5))
                train_top1 += acc1
                train_top5 += acc5

            # --- Validation（テスト）フェーズ ---
            model.eval()
            test_total = 0
            test_top1, test_top5 = 0.0, 0.0
            
            epoch_alpha_g_list = []
            epoch_alpha_l_list = []

            # with torch.no_grad():
            #     for x_g, x_l, labels in test_loader:
            #         x_g, x_l, labels = x_g.to(device), x_l.to(device), labels.to(device)
            #         with torch.amp.autocast(device_type=device.type):
            #             # テスト時は実力検証のため、常に2ストリームのゲートマージで推論
            #             outputs, a_g, a_l = model(x_g, x_l, global_only=False)
            with torch.no_grad():
                for x_g, x_l, labels in test_loader:
                    x_g, x_l, labels = x_g.to(device), x_l.to(device), labels.to(device)
                    with torch.amp.autocast(device_type=device.type):
                        # 訓練時とフラグを同期させ、前半50エポックはテストもグローバル単体で評価する
                        outputs, a_g, a_l = model(x_g, x_l, global_only=is_global_only)

                    test_total += labels.size(0)
                    acc1, acc5 = calculate_topk_accuracy(outputs, labels, topk=(1, 5))
                    test_top1 += acc1
                    test_top5 += acc5
                    
                    epoch_alpha_g_list.append(a_g.cpu().numpy().mean())
                    epoch_alpha_l_list.append(a_l.cpu().numpy().mean())

            tr_acc1_pct = (train_top1 / train_total) * 100
            tr_acc5_pct = (train_top5 / train_total) * 100
            te_acc1_pct = (test_top1 / test_total) * 100
            te_acc5_pct = (test_top5 / test_total) * 100
            
            avg_alpha_g = np.mean(epoch_alpha_g_list)
            avg_alpha_l = np.mean(epoch_alpha_l_list)

            # 各フェーズの状態を明示的にコンソールへ可視化
            phase_str = "PHASE-1 (G-Only)" if is_global_only else "PHASE-2 (Fused)"
            epoch_status = (
                f"Epoch {epoch+1:03d}/{num_epochs} [{phase_str} | Bb-LR: {current_backbone_lr:.6f} / Gt-LR: {current_gating_lr:.5f}] -> "
                f"Loss: {train_loss/train_total:.4f} | Train Top-1: {tr_acc1_pct:.2f}% | "
                f"★Test Top-1: {te_acc1_pct:.2f}% (Top-5: {te_acc5_pct:.2f}%) | "
                f"[G-Gate: {avg_alpha_g:.3f} / L-Gate: {avg_alpha_l:.3f}]"
            )
            print(epoch_status)
            f.write(epoch_status + "\n")

            best_test_acc1 = save_best_model(model, te_acc1_pct, best_test_acc1, MODEL_SAVE_PATH)
            f.flush()

    print(f"\n🎉 2段階シーケンシャル動的最適化の学習が完了しました！最高の Test Top-1 精度は {best_test_acc1:.2f}% でした。")