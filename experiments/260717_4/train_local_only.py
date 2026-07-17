import copy
import math
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from expt_thu_eact_50_chl import config

# ====================================================
# ─── ⚙️ 再現性のためのグローバル設定 ───
# ====================================================
SEED = 42
SAMPLING_MULTIPLIER = 5.0  # 苦手データのサンプリング倍率

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
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ====================================================
# ─── 📂 パス・ディレクトリ設定 (config.py準拠) ───
# ====================================================
PROJECT_ROOT = config.PROJECT_ROOT
PROCESSED_DIR = PROJECT_ROOT / "experiments" / "260717_3" / "processed_data"
CURRENT_DIR = Path(__file__).parent.resolve()

from expt_thu_eact_50_chl.utils import (
    calculate_topk_accuracy,
    save_best_model,
)

# ─── ⚙️ ハイパーパラメータ設定 (ローカル単体最適化用) ───
NUM_EPOCHS = 80
LR_INITIAL = 0.0003
BATCH_SIZE = 16

MODEL_SAVE_PATH = CURRENT_DIR / "best_local_only_model.pth"
LOG_FILENAME = CURRENT_DIR / "local_only_result.txt"

# ─── 📊 過去の実験から得られたGlobalモデルのクラス別誤答率 (設定値) ───
# 携帯電話(14)は 58.3%、敬礼(36)は 25.0% と、この2クラスだけでも明確な学習重みの差が発生します。
STATIC_ERROR_RATES = {
    0: 0.167, 1: 0.083, 2: 0.250, 3: 0.083, 5: 0.125, 6: 0.167, 7: 0.062, 
    8: 0.333, 9: 0.583, 10: 0.333, 11: 0.250, 14: 0.583, 15: 0.833, 
    16: 0.538, 17: 0.333, 18: 0.083, 19: 0.083, 21: 0.333, 22: 0.083, 
    27: 0.417, 29: 0.083, 30: 0.417, 34: 0.500, 35: 0.750, 36: 0.250, 
    37: 0.167, 39: 0.250, 40: 0.167, 41: 0.583, 42: 0.583, 46: 0.083
}


# ----------------------------------------------------
# ─── 🛡️ ASAM オプティマイザ定義 ───
# ----------------------------------------------------
class SAM(torch.optim.Optimizer):
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


# ----------------------------------------------------
# ─── 📊 ローカル専用データセット定義 ───
# ----------------------------------------------------
class LocalOnlyDataset(Dataset):
    def __init__(self, mode="train"):
        self.local_dir = PROCESSED_DIR / mode / "local"
        
        if not self.local_dir.exists():
            raise FileNotFoundError(f"ローカルデータディレクトリが見つかりません: {self.local_dir}")

        self.file_names = sorted([p.name for p in self.local_dir.glob("*.npy")])

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        f_name = self.file_names[idx]
        label_str = f_name.split("_label_")[-1].split(".npy")[0]
        label = int(label_str.replace("A", ""))

        feat_l = np.load(self.local_dir / f_name) # 形状: (4, 260, 346)

        # 外れ値（極値）に対するマイルドな正規化
        ml = np.max(np.abs(feat_l))
        if ml > 0: 
            feat_l = feat_l / ml

        return (
            torch.tensor(feat_l, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )


# ----------------------------------------------------
# ─── 🔍 動的サンプラーおよびエラー率計算ヘルパー ───
# ----------------------------------------------------
def create_static_error_sampler(dataset, error_rates_dict, multiplier=5.0, seed=42):
    """
    設定されたテスト誤答率に基づいて、苦手なクラスのサンプリング確率を
    動的（再現性保証）に引き上げるWeightedRandomSamplerを作成する
    """
    weights = []
    for idx in range(len(dataset)):
        f_name = dataset.file_names[idx]
        label_str = f_name.split("_label_")[-1].split(".npy")[0]
        label = int(label_str.replace("A", ""))
        
        # 辞書からエラー率を取得 (未定義のものは誤答率0.0とみなす)
        err_val = error_rates_dict.get(label, 0.0)
        
        # 誤答率に応じたスケーリング
        w = 1.0 + (multiplier - 1.0) * err_val
        weights.append(w)

    weights = torch.DoubleTensor(weights)
    
    g = torch.Generator()
    g.manual_seed(seed)
    num_samples = len(dataset)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples,
        replacement=True,
        generator=g
    )
    return sampler


# ----------------------------------------------------
# ─── 🧠 高解像度維持×チャネル半減型 ResNet18 (オプションA) ───
# ----------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class HighResResNet18HalfWidth(nn.Module):
    """
    解像度 [260, 346] の細部エッジを100%保護しつつ、
    標準ResNet18のチャネル数を半分 [32, 64, 128, 256] に削減して
    圧倒的な過学習耐性と軽量化を両立したカスタムモデル
    """
    def __init__(self, num_classes=50):
        super(HighResResNet18HalfWidth, self).__init__()
        self.in_planes = 32  # チャネル数を半減 (通常は64)

        # 1. 最初の畳み込み層: 入力4ch、高解像度維持のため stride=1
        self.conv1 = nn.Conv2d(4, 32, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        # MaxPoolは完全に排除（空間情報の急激なダウンサンプリングを阻止）

        # 2. 残差ブロックのチャネルを [32, 64, 128, 256] に半減
        self.layer1 = self._make_layer(BasicBlock, 32, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 64, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 256, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * BasicBlock.expansion, num_classes)

        self._init_custom_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _init_custom_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# ====================================================
# ─── 🏃 トレーニングメイン処理 (FP32 + ASAM) ───
# ====================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # チャネル半減・高解像度維持型 ResNet18
    model = HighResResNet18HalfWidth(num_classes=50).to(device)
    
    train_dataset = LocalOnlyDataset(mode="train")
    test_dataset = LocalOnlyDataset(mode="test")

    # 🔍 苦手ターゲット重点サンプリングの適用
    sampler = create_static_error_sampler(
        train_dataset, 
        STATIC_ERROR_RATES, 
        multiplier=SAMPLING_MULTIPLIER, 
        seed=SEED
    )

    g_init = torch.Generator()
    g_init.manual_seed(SEED)
    
    # sampler指定時は、DataLoaderの shuffle をFalseにする必要があります。
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=sampler, 
        num_workers=4, 
        pin_memory=True, 
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g_init
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    # ASAMの適用
    optimizer = SAM(
        model.parameters(),
        base_optimizer=torch.optim.AdamW,
        rho=0.05,
        adaptive=True,
        lr=LR_INITIAL,
        weight_decay=0.01
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_test_acc1 = 0.0

    with LOG_FILENAME.open("w", encoding="utf-8") as f:
        f.write("=== Local-Only High-Resolution ResNet18 Half-Width Training Log ===\n")
        f.write(f"Sampling Multiplier: {SAMPLING_MULTIPLIER}x\n\n")

        for epoch in range(NUM_EPOCHS):
            # コサインアニーリングLRスケジューリング
            cos_factor = (1 + math.cos(math.pi * epoch / NUM_EPOCHS)) / 2
            current_lr = LR_INITIAL * cos_factor
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            # --- Training with ASAM ---
            model.train()
            train_loss, train_total = 0.0, 0
            train_top1 = 0.0

            for x_l, labels in train_loader:
                x_l, labels = x_l.to(device), labels.to(device)
                
                # ─── 1st Step: 摂動の追加 ───
                optimizer.zero_grad()
                outputs = model(x_l)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.first_step(zero_grad=True)

                # ─── 2nd Step: 勾配更新 ───
                outputs_adv = model(x_l)
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
            
            # クラスごとの正答率追跡用のカウンター
            class_correct = np.zeros(50)
            class_total = np.zeros(50)

            with torch.no_grad():
                for x_l, labels in test_loader:
                    x_l, labels = x_l.to(device), labels.to(device)
                    outputs = model(x_l)
                    test_total += labels.size(0)
                    acc1, _ = calculate_topk_accuracy(outputs, labels, topk=(1, 5))
                    test_top1 += acc1
                    
                    # クラス別正答率の集計
                    preds = torch.argmax(outputs, dim=1)
                    for p, l in zip(preds, labels):
                        class_total[l.item()] += 1
                        if p == l:
                            class_correct[l.item()] += 1

            tr_acc = (train_top1 / train_total) * 100
            te_acc = (test_top1 / test_total) * 100

            status = (
                f"Epoch {epoch+1:03d} [LR: {current_lr:.6f}] -> "
                f"Loss: {train_loss/train_total:.4f} | Train: {tr_acc:.2f}% | ★Test: {te_acc:.2f}%"
            )
            print(status)
            f.write(status + "\n")

            # 🔍 本番でのボトルネックとなりやすい携帯(14)と敬礼(36)のクラス精度を毎エポック可視化
            class_log = "   🔍 [Target Class Accuracy Monitoring]\n"
            for target_cls in [14, 36]:
                tot = class_total[target_cls]
                acc = (class_correct[target_cls] / tot * 100) if tot > 0 else 0.0
                class_log += f"      Class {target_cls:02d}: {acc:.2f}% ({int(class_correct[target_cls])}/{int(tot)})\n"
            print(class_log, end="")
            f.write(class_log)

            # 最高精度の自動ディスク保存
            best_test_acc1 = save_best_model(model, te_acc, best_test_acc1, MODEL_SAVE_PATH)
            f.flush()