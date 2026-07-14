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
import random

from expt_thu_eact_50_chl.utils import (
    calculate_topk_accuracy,
    save_best_model,
)

# ====================================================
# ─── ⚙️ 4-Stage ハイパーパラメータ設定 ───
# ====================================================
# 各ステージのエポック数設定
STAGE1_EPOCHS = 50  # ステップ1: Global単体育成 (0〜49)
STAGE2_EPOCHS = 50  # ステップ2: Local単体・ハードマイニング (50〜99)
STAGE3_EPOCHS = 50  # ステップ3: ゲート単体最適化 (100〜149)
STAGE4_EPOCHS = 50  # ステップ4: Local & ゲート同時最適化 (150〜199)

# 各ステージ・各ストリーム別の独立初期学習率（LR）設定
STAGE1_LR_GLOBAL = 0.0003

STAGE2_LR_LOCAL  = 0.0003
HARD_EXAMPLE_WEIGHT = 5.0  # ★ステップ1でGlobalが間違えたデータのサンプリング確率引き上げ倍率
RANDOM_SEED = 42           # ★実験の再現性を完全に担保するための共通シード値

STAGE3_LR_GATING = 0.001

STAGE4_LR_LOCAL  = 0.0001
STAGE4_LR_GATING = 0.0005
# ※Globalの重みはステップ1の最高精度モデルで完全固定（LR=0.0）するため、STAGE2〜4のLR_GLOBALは不要です
# ====================================================

NUM_EPOCHS = STAGE1_EPOCHS + STAGE2_EPOCHS + STAGE3_EPOCHS + STAGE4_EPOCHS
CURRENT_DIR = Path(__file__).parent.resolve()
PROCESSED_DIR = CURRENT_DIR.parent / "260713_7" / "processed_data"

MODEL_SAVE_PATH = CURRENT_DIR / "best_model_augmented.pth"
LOG_FILENAME = CURRENT_DIR / "result_augmented.txt"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


class HoloEvNetClassWiseGated(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        self.num_classes = num_classes
        
        # Global ストリーム
        resnet_g = models.resnet18(weights=None)
        self.g_conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.g_bn1 = resnet_g.bn1
        self.g_relu = resnet_g.relu
        self.g_maxpool = resnet_g.maxpool
        self.g_layer1 = resnet_g.layer1
        self.g_layer2 = resnet_g.layer2
        self.g_layer3 = resnet_g.layer3
        self.g_layer4 = resnet_g.layer4
        self.g_spatial4 = SpatialAttentionOnly(kernel_size=7)
        self.g_gsg = GlobalSpectralGating(channels=512, T_prime=7, H_prime=9)
        self.g_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.g_classifier = nn.Linear(512, num_classes)

        # Local ストリーム
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

        # クラス別融合ゲート層 (50クラス × 2系統 = 100出力)
        self.gating_layer = nn.Sequential(
            nn.Linear(512, 128), 
            nn.ReLU(), 
            nn.Linear(128, num_classes * 2)
        )
        
        self._init_custom_weights()

    def _init_custom_weights(self):
        nn.init.kaiming_normal_(self.g_conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.l_conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.normal_(self.g_classifier.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.g_classifier.bias, 0.0)
        nn.init.normal_(self.l_classifier.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.l_classifier.bias, 0.0)
        for m in self.gating_layer:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x_global, x_local, mode='both'):
        # --- ① グローバル特徴の抽出 ---
        g = self.g_conv1(x_global)
        g = self.g_bn1(g)
        g = self.g_relu(g)
        g = self.g_maxpool(g)
        g = self.g_layer1(g)
        g = self.g_layer2(g)
        g = self.g_layer3(g)
        g_in = self.g_layer4(g)
        g_in = self.g_spatial4(g_in)
        g_out = g_in + self.g_gsg(g_in)
        feat_global = self.g_avgpool(g_out).view(g_out.size(0), -1)
        y_global = self.g_classifier(feat_global)

        if mode == 'global_only':
            alpha_g = torch.ones((x_global.size(0), self.num_classes), device=x_global.device)
            alpha_l = torch.zeros((x_global.size(0), self.num_classes), device=x_global.device)
            return y_global, alpha_g, alpha_l

        # --- ② ローカル特徴の抽出 ---
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

        if mode == 'local_only':
            alpha_g = torch.zeros((x_global.size(0), self.num_classes), device=x_global.device)
            alpha_l = torch.ones((x_global.size(0), self.num_classes), device=x_global.device)
            return y_local, alpha_g, alpha_l

        # --- ③ クラス別適応融合 ---
        gate_logits = self.gating_layer(feat_global).view(-1, self.num_classes, 2)
        gate_weights = F.softmax(gate_logits, dim=2)
        
        alpha_g, alpha_l = gate_weights[:, :, 0], gate_weights[:, :, 1]
        y_final = alpha_g * y_global + alpha_l * y_local
        return y_final, alpha_g, alpha_l


if __name__ == "__main__":
    # シードの初期固定
    set_seed(RANDOM_SEED)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    model = HoloEvNetClassWiseGated(num_classes=50).to(device)
    
    # 初期の通常のデータローダー構築
    train_dataset = HoloEvTwinFolderDataset(mode="train")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(HoloEvTwinFolderDataset(mode="test"), batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # パラメータの分離
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
        f.write("=== 4-Stage Progressive Fusion Hard-Mining Training Log ===\n")

        for epoch in range(NUM_EPOCHS):
            
            # ─── 🔄 【重要】ステップ2開始時にハードエグザンプル抽出とサンプラーの動的構築 ───
            if epoch == STAGE1_EPOCHS:
                print(f"\n[*] [移行処理] ステップ1の最高精度モデルをロードし、ハードエグザンプルを抽出します...")
                # ステップ1で最高精度だった重みを厳密にロード
                model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
                model.eval()
                
                # 順序を固定して正誤判定を行うための評価用DataLoader
                eval_train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
                
                sample_weights = []
                with torch.no_grad():
                    for x_g, _, labels in eval_train_loader:
                        x_g, labels = x_g.to(device), labels.to(device)
                        with torch.amp.autocast(device_type=device.type):
                            # Global単体モードで推論
                            outputs, _, _ = model(x_g, None, mode='global_only')
                            preds = torch.argmax(outputs, dim=1)
                        
                        # 正誤を判定 (正解=1.0, 不正解=指定された変数倍率)
                        is_correct = (preds == labels).cpu().numpy()
                        weights = np.where(is_correct, 1.0, HARD_EXAMPLE_WEIGHT)
                        sample_weights.extend(weights)
                
                # 再現性担保のためのジェネレータ固定
                set_seed(RANDOM_SEED)
                g = torch.Generator()
                g.manual_seed(RANDOM_SEED)
                
                # 加重サンプラーを作成
                sampler = torch.utils.data.WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True,
                    generator=g
                )
                
                # サンプラーを適用した DataLoader に完全切り替え (シード固定対応)
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=16, 
                    sampler=sampler, 
                    num_workers=4, 
                    pin_memory=True, 
                    drop_last=True,
                    generator=g
                )
                print(f"[✓] 抽出完了。不正解データのサンプリング確率を {HARD_EXAMPLE_WEIGHT} 倍に引き上げました。\n")

            # ─── 🔄 【新規追加】ステップ3開始時にサンプリングを通常（全データ均等・シャッフル）に戻す ───
            if epoch == STAGE1_EPOCHS + STAGE2_EPOCHS:
                print(f"\n[*] [移行処理] ステップ3開始に伴い、サンプラーを解除して通常のシャッフルLoaderに戻します...")
                # 再現性（実験の同一性）を完全に担保するため、再度シードを設定
                set_seed(RANDOM_SEED)
                g = torch.Generator()
                g.manual_seed(RANDOM_SEED)
                
                # 通常の shuffle=True な DataLoader にリセット（上書き）
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=16, 
                    shuffle=True, 
                    num_workers=4, 
                    pin_memory=True, 
                    drop_last=True,
                    generator=g
                )
                print(f"[✓] 通常のシャッフルデータローダーに切り替えました。\n")
            # 👆 ★★★ ここまで新規追加 ★★★

            # ─── 📈 ステージ別の学習率＆コサインアニーリング制御 ───
            if epoch < STAGE1_EPOCHS:
                # 【ステップ1】Global単体育成 (Globalのみ更新)
                phase, mode = "STAGE-1 (Global-Train)", "global_only"
                cos_factor = (1 + math.cos(math.pi * epoch / STAGE1_EPOCHS)) / 2
                optimizer.param_groups[0]["lr"] = STAGE1_LR_GLOBAL * cos_factor
                optimizer.param_groups[1]["lr"] = 0.0
                optimizer.param_groups[2]["lr"] = 0.0
            
            elif epoch < STAGE1_EPOCHS + STAGE2_EPOCHS:
                # 【ステップ2】Local単体系・ハードマイニング (Localのみ更新、Globalは固定)
                phase, mode = "STAGE-2 (Local-HardMining)", "local_only"
                stage_epoch = epoch - STAGE1_EPOCHS
                cos_factor = (1 + math.cos(math.pi * stage_epoch / STAGE2_EPOCHS)) / 2
                optimizer.param_groups[0]["lr"] = 0.0
                optimizer.param_groups[1]["lr"] = STAGE2_LR_LOCAL * cos_factor
                optimizer.param_groups[2]["lr"] = 0.0
            
            elif epoch < STAGE1_EPOCHS + STAGE2_EPOCHS + STAGE3_EPOCHS:
                # 【ステップ3】ゲート単体最適化 (ゲートのみ更新、Global/Local共に完全固定)
                phase, mode = "STAGE-3 (Gating-Train)", "both"
                stage_epoch = epoch - (STAGE1_EPOCHS + STAGE2_EPOCHS)
                cos_factor = (1 + math.cos(math.pi * stage_epoch / STAGE3_EPOCHS)) / 2
                optimizer.param_groups[0]["lr"] = 0.0
                optimizer.param_groups[1]["lr"] = 0.0
                optimizer.param_groups[2]["lr"] = STAGE3_LR_GATING * cos_factor
            
            else:
                # 【ステップ4】Local & ゲート同時最適化 (Globalは完全固定のまま)
                phase, mode = "STAGE-4 (Local&Gate-Synch)", "both"
                stage_epoch = epoch - (STAGE1_EPOCHS + STAGE2_EPOCHS + STAGE3_EPOCHS)
                cos_factor = (1 + math.cos(math.pi * stage_epoch / STAGE4_EPOCHS)) / 2
                optimizer.param_groups[0]["lr"] = 0.0
                optimizer.param_groups[1]["lr"] = STAGE4_LR_LOCAL * cos_factor
                optimizer.param_groups[2]["lr"] = STAGE4_LR_GATING * cos_factor

            # 現在の実学習率を取得
            current_g_lr = optimizer.param_groups[0]["lr"]
            current_l_lr = optimizer.param_groups[1]["lr"]
            current_gate_lr = optimizer.param_groups[2]["lr"]

            # --- Training フェーズ ---
            model.train()
            
            # ステップ2〜4の間、Globalを完全フリーズ化（BNやドロップアウトも含む挙動固定）
            if epoch >= STAGE1_EPOCHS:
                model.g_conv1.eval()
                model.g_bn1.eval()
                model.g_layer1.eval()
                model.g_layer2.eval()
                model.g_layer3.eval()
                model.g_layer4.eval()
                model.g_spatial4.eval()
                model.g_gsg.eval()
                model.g_classifier.eval()

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
                f"Epoch {epoch+1:03d} [{phase} | G-LR: {current_g_lr:.6f} | L-LR: {current_l_lr:.6f} | Gt-LR: {current_gate_lr:.6f}] -> "
                f"Loss: {train_loss/train_total:.4f} | Train: {tr_acc:.2f}% | ★Test: {te_acc:.2f}% | [G-Gate_Avg: {avg_a_g:.3f} / L-Gate_Avg: {avg_a_l:.3f}]"
            )
            print(status)
            f.write(status + "\n")
            
            best_test_acc1 = save_best_model(model, te_acc, best_test_acc1, MODEL_SAVE_PATH)

            # 👇 ★★★ ここから新規追加 ★★★
            # ─── 💾 150エポック目および200エポック目のモデルを強制保存 ───
            if epoch + 1 == 150:
                epoch150_path = CURRENT_DIR / "model_epoch150.pth"
                torch.save(model.state_dict(), epoch150_path)
                msg = f"⏰ [強制保存] 150エポック目（Stage-3終了時）のモデルを保存しました: {epoch150_path.name}"
                print(msg)
                f.write(msg + "\n")
                
            if epoch + 1 == 200:
                epoch200_path = CURRENT_DIR / "model_epoch200.pth"
                torch.save(model.state_dict(), epoch200_path)
                msg = f"⏰ [強制保存] 200エポック目（最終ステージ終了時）のモデルを保存しました: {epoch200_path.name}"
                print(msg)
                f.write(msg + "\n")
            # 👆 ★★★ ここまで新規追加 ★★★
            
            f.flush()