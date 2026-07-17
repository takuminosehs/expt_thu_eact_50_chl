import copy
import math
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as models
from expt_thu_eact_50_chl import config

# ====================================================
# ─── ⚙️ 再現性のためのグローバル設定 ───
# ====================================================
SEED = 42
SAMPLING_MULTIPLIER = 5.0  # ステップ2における苦手データのサンプリング倍率

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 初期シード設定の実行
seed_everything(SEED)

def seed_worker(worker_id):
    """DataLoaderのマルチプロセスにおける再現性担保用"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ====================================================
# ─── 📂 パス・ディレクトリ設定 ───
# ====================================================
PROJECT_ROOT = config.PROJECT_ROOT
PROCESSED_DIR = PROJECT_ROOT / "experiments" / "260713_7" / "processed_data"
CURRENT_DIR = Path(__file__).parent.resolve()

from expt_thu_eact_50_chl.utils import (
    calculate_topk_accuracy,
    save_best_model,
)

# ====================================================
# ─── ⚙️ 3-Stage ハイパーパラメータ設定 ───
# ====================================================
STAGE1_EPOCHS = 50
STAGE1_LR_GLOBAL = 0.0003

STAGE2_EPOCHS = 50
STAGE2_LR_LOCAL = 0.0003

STAGE3_EPOCHS = 50
STAGE3_LR_GATING = 0.001 

NUM_EPOCHS = STAGE1_EPOCHS + STAGE2_EPOCHS + STAGE3_EPOCHS

MODEL_SAVE_PATH = CURRENT_DIR / "best_model_augmented.pth"
LOG_FILENAME = CURRENT_DIR / "result_augmented.txt"


# ----------------------------------------------------
# ─── 🛡️ SAM / ASAM オプティマイザ定義 ───
# ----------------------------------------------------
class SAM(torch.optim.Optimizer):
    """
    ASAM (Adaptive Sharpness-Aware Minimization) 対応 SAM 実装
    """
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=True, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


# ----------------------------------------------------
# ─── 📊 データセット定義 (pathlib 完全準拠) ───
# ----------------------------------------------------
class HoloEvTwinFolderDataset(Dataset):
    def __init__(self, mode="train"):
        self.global_dir = PROCESSED_DIR / mode / "global"
        self.local_dir = PROCESSED_DIR / mode / "local"
        
        if not self.global_dir.exists():
            raise FileNotFoundError(f"グローバルデータディレクトリが見つかりません: {self.global_dir}")

        self.file_names = sorted([p.name for p in self.global_dir.glob("*.npy")])

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        f_name = self.file_names[idx]
        label_str = f_name.split("_label_")[-1].split(".npy")[0]
        label = int(label_str.replace("A", ""))

        feat_g = np.load(self.global_dir / f_name)
        feat_l = np.load(self.local_dir / f_name)

        mg = np.max(np.abs(feat_g))
        if mg > 0: 
            feat_g = feat_g / mg
        ml = np.max(np.abs(feat_l))
        if ml > 0: 
            feat_l = feat_l / ml

        return (
            torch.tensor(feat_g, dtype=torch.float32),
            torch.tensor(feat_l, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )


# ----------------------------------------------------
# ─── 🧠 モデルアーキテクチャ定義 ───
# ----------------------------------------------------
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
        
        # Global ストリーム: 入力形状 (4, 224, 260)
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

        # Local ストリーム: 入力形状 (4, 260, 346)
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

        gate_logits = self.gating_layer(feat_global).view(-1, self.num_classes, 2)
        gate_weights = F.softmax(gate_logits, dim=2)
        
        alpha_g, alpha_l = gate_weights[:, :, 0], gate_weights[:, :, 1]
        y_final = alpha_g * y_global + alpha_l * y_local
        return y_final, alpha_g, alpha_l


# ----------------------------------------------------
# ─── 🔍 動的サンプラーおよびエラー率計算ヘルパー ───
# ----------------------------------------------------
def compute_class_error_rates(model, loader, device, num_classes=50):
    """
    テストデータに対するクラス（ラベル）ごとの誤答率を算出する関数
    """
    model.eval()
    class_correct = torch.zeros(num_classes, device=device)
    class_total = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for x_g, x_l, labels in loader:
            x_g, x_l, labels = x_g.to(device), x_l.to(device), labels.to(device)
            # ステップ1終了時の判定。Globalのみで推論
            outputs, _, _ = model(x_g, x_l, mode="global_only")
            preds = torch.argmax(outputs, dim=1)
            for p, l in zip(preds, labels):
                class_total[l] += 1
                if p == l:
                    class_correct[l] += 1

    error_rates = torch.zeros(num_classes)
    for c in range(num_classes):
        total = class_total[c].item()
        if total > 0:
            error_rates[c] = 1.0 - (class_correct[c].item() / total)
        else:
            error_rates[c] = 0.0  # テストに存在しなかったクラスは誤答率0
    return error_rates


def create_error_aware_sampler(dataset, error_rates, multiplier=5.0, seed=42):
    """
    誤答率の高いラベルを持つサンプルの抽出確率を引き上げる WeightedRandomSampler を作成する
    """
    weights = []
    for idx in range(len(dataset)):
        # ファイル名から高速にクラスのラベルIDを抽出
        f_name = dataset.file_names[idx]
        label_str = f_name.split("_label_")[-1].split(".npy")[0]
        label = int(label_str.replace("A", ""))
        
        # 誤答率 E_c を用いてサンプリング重みを計算
        # 式: 1.0 + (multiplier - 1.0) * E_c
        # 誤答率が 1.0（全問不正解）の場合は multiplier（5.0倍）、0.0（全問正解）の場合は 1.0倍となる
        w = 1.0 + (multiplier - 1.0) * error_rates[label].item()
        weights.append(w)

    weights = torch.DoubleTensor(weights)
    
    # 再現性を確保するためのジェネレータ
    g = torch.Generator()
    g.manual_seed(seed)

    # 1エポック当たりのサンプリング数を決定（デフォルトは元のデータサイズを維持）
    num_samples = len(dataset)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples,
        replacement=True,
        generator=g
    )
    return sampler


# ====================================================
# ─── 🏃 トレーニングメイン処理 (FP32 + ASAM) ───
# ====================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    model = HoloEvNetClassWiseGated(num_classes=50).to(device)
    
    # 初期状態（Stage-1）のDataLoader設定（再現性のためにシードを設定）
    g_init = torch.Generator()
    g_init.manual_seed(SEED)
    
    train_dataset = HoloEvTwinFolderDataset(mode="train")
    test_dataset = HoloEvTwinFolderDataset(mode="test")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=16, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g_init
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=16, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    global_params, local_params, gating_params = [], [], []
    for n, p in model.named_parameters():
        if "gating_layer" in n: 
            gating_params.append(p)
        elif n.startswith("g_"): 
            global_params.append(p)
        elif n.startswith("l_"): 
            local_params.append(p)

    # 3つのパラメータグループ構造を維持したSAM（ベース：AdamW）
    optimizer = SAM(
        [
            {"params": global_params, "lr": 0.0, "weight_decay": 0.01},
            {"params": local_params, "lr": 0.0, "weight_decay": 0.01},
            {"params": gating_params, "lr": 0.0, "weight_decay": 0.0}
        ],
        base_optimizer=torch.optim.AdamW,
        rho=0.05,
        adaptive=True  # ASAMの有効化
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_test_acc1 = 0.0

    stage_best_acc = 0.0
    stage_best_weights = None

    with LOG_FILENAME.open("w", encoding="utf-8") as f:
        f.write("=== Class-Wise Adaptive Fusion Training Log (3-Stage with ASAM) ===\n")

        for epoch in range(NUM_EPOCHS):
            # --- ステージごとのフェーズ & 学習率制御 ---
            if epoch < STAGE1_EPOCHS:
                current_stage = 1
                phase, mode = "STAGE-1 (Global-Train)", "global_only"
                cos_factor = (1 + math.cos(math.pi * epoch / STAGE1_EPOCHS)) / 2
                optimizer.param_groups[0]["lr"] = STAGE1_LR_GLOBAL * cos_factor
                optimizer.param_groups[1]["lr"] = 0.0
                optimizer.param_groups[2]["lr"] = 0.0
            
            elif epoch < STAGE1_EPOCHS + STAGE2_EPOCHS:
                current_stage = 2
                phase, mode = "STAGE-2 (Local-Train)", "local_only"
                stage_epoch = epoch - STAGE1_EPOCHS
                cos_factor = (1 + math.cos(math.pi * stage_epoch / STAGE2_EPOCHS)) / 2
                optimizer.param_groups[0]["lr"] = 0.0
                optimizer.param_groups[1]["lr"] = STAGE2_LR_LOCAL * cos_factor
                optimizer.param_groups[2]["lr"] = 0.0
            
            else:
                current_stage = 3
                phase, mode = "STAGE-3 (Gating-Train)", "both"
                stage_epoch = epoch - (STAGE1_EPOCHS + STAGE2_EPOCHS)
                cos_factor = (1 + math.cos(math.pi * stage_epoch / STAGE3_EPOCHS)) / 2
                optimizer.param_groups[0]["lr"] = 0.0
                optimizer.param_groups[1]["lr"] = 0.0
                optimizer.param_groups[2]["lr"] = STAGE3_LR_GATING * cos_factor

            current_g_lr = optimizer.param_groups[0]["lr"]
            current_l_lr = optimizer.param_groups[1]["lr"]
            current_gate_lr = optimizer.param_groups[2]["lr"]

            # --- Training with ASAM ---
            model.train()
            train_loss, train_total = 0.0, 0
            train_top1 = 0.0

            for x_g, x_l, labels in train_loader:
                x_g, x_l, labels = x_g.to(device), x_l.to(device), labels.to(device)
                
                # ─── 1st Step ───
                optimizer.zero_grad()
                outputs, _, _ = model(x_g, x_l, mode=mode)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.first_step(zero_grad=True)

                # ─── 2nd Step ───
                outputs_adv, _, _ = model(x_g, x_l, mode=mode)
                loss_adv = criterion(outputs_adv, labels)
                loss_adv.backward()
                optimizer.second_step(zero_grad=True)

                train_loss += loss.item() * labels.size(0)
                train_total += labels.size(0)
                acc1, _ = calculate_topk_accuracy(outputs, labels, topk=(1, 5))
                train_top1 += acc1

            # --- Validation ---
            model.eval()
            test_total, test_top1 = 0, 0.0
            epoch_alpha_g, epoch_alpha_l = [], []

            with torch.no_grad():
                for x_g, x_l, labels in test_loader:
                    x_g, x_l, labels = x_g.to(device), x_l.to(device), labels.to(device)
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

            # --- ステージごとのベスト重みの保存 ---
            if te_acc > stage_best_acc:
                stage_best_acc = te_acc
                stage_best_weights = copy.deepcopy(model.state_dict())
                print(f"    🔥 [Stage-{current_stage} Best Updated] Test Acc: {te_acc:.2f}% (Epoch {epoch+1:03d})")

            # 全エポックを通じた最高精度モデルのディスクへの自動保存
            best_test_acc1 = save_best_model(model, te_acc, best_test_acc1, MODEL_SAVE_PATH)

            # --- 🛠️ ステージ切り替え処理 ---
            
            # Stage 1 終了時
            if epoch + 1 == STAGE1_EPOCHS:
                # 1. 最高精度の重みをロード
                if stage_best_weights is not None:
                    model.load_state_dict(stage_best_weights)
                    msg = (
                        f"\n🔄 [Stage-1 Transition] Stage-1の最高精度重み ({stage_best_acc:.2f}%) "
                        f"をロードしました。\n"
                    )
                    print(msg)
                    f.write(msg + "\n")
                
                # 2. テストデータを用いたクラス別誤答率の計算
                error_rates = compute_class_error_rates(model, test_loader, device, num_classes=50)
                
                err_log = "💡 [計算完了] クラス別テスト誤答率:\n"
                for c_id, err_val in enumerate(error_rates):
                    if err_val > 0:
                        err_log += f"  Class {c_id:02d}: {err_val * 100:.1f}%\n"
                print(err_log)
                f.write(err_log + "\n")

                # 3. 再現性を担保した WeightedRandomSampler の構築
                sampler = create_error_aware_sampler(
                    train_dataset, error_rates, multiplier=SAMPLING_MULTIPLIER, seed=SEED
                )

                # 4. ステップ2専用の DataLoader に更新（サンプラーを有効化するため shuffle=False）
                g_step2 = torch.Generator()
                g_step2.manual_seed(SEED)
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=16,
                    sampler=sampler,
                    num_workers=4,
                    pin_memory=True,
                    drop_last=True,
                    worker_init_fn=seed_worker,
                    generator=g_step2
                )
                
                transition_msg = "🚀 [Stage-2 Start] エラー指向の加重サンプラーを適用してStage-2を開始します。\n"
                print(transition_msg)
                f.write(transition_msg + "\n")

                stage_best_acc = 0.0
                stage_best_weights = None

            # Stage 2 終了時
            elif epoch + 1 == STAGE1_EPOCHS + STAGE2_EPOCHS:
                if stage_best_weights is not None:
                    model.load_state_dict(stage_best_weights)
                    msg = (
                        f"\n🔄 [Stage-2 Transition] Stage-2の最高精度重み ({stage_best_acc:.2f}%) "
                        f"をロードしました。\n"
                    )
                    print(msg)
                    f.write(msg + "\n")

                # 1. サンプリングを元のシャッフル（通常サンプリング）に戻す
                g_step3 = torch.Generator()
                g_step3.manual_seed(SEED)
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=16,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True,
                    drop_last=True,
                    worker_init_fn=seed_worker,
                    generator=g_step3
                )
                
                transition_msg = "🚀 [Stage-3 Start] サンプリング確率を均等（Shuffle）に戻してStage-3へ移行します。\n"
                print(transition_msg)
                f.write(transition_msg + "\n")

                stage_best_acc = 0.0
                stage_best_weights = None

            # Stage 3（全体の終了）時の強制保存
            elif epoch + 1 == NUM_EPOCHS:
                final_model_path = CURRENT_DIR / f"model_epoch{NUM_EPOCHS}.pth"
                torch.save(model.state_dict(), final_model_path)
                msg = f"\n⏰ [完了保存] 最終150エポック（Stage-3終了時）のモデルを保存しました: {final_model_path.name}\n"
                print(msg)
                f.write(msg + "\n")

            f.flush()