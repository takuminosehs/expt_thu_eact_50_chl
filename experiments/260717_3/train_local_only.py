import copy
import math
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from expt_thu_eact_50_chl import config

# ====================================================
# ─── ⚙️ 再現性のためのグローバル設定 ───
# ====================================================
SEED = 42

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

        # ボクセルの外れ値（極値）に対するマイルドな正規化
        ml = np.max(np.abs(feat_l))
        if ml > 0: 
            feat_l = feat_l / ml

        return (
            torch.tensor(feat_l, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )


# ----------------------------------------------------
# ─── 🧠 高解像度維持型 ResNet18 (選択肢A) ───
# ----------------------------------------------------
class HighResResNet18(nn.Module):
    """
    入力解像度 [260, 346] のエッジを保持するため、
    最初の畳み込み層のstrideを1にし、MaxPoolを完全に無効化（Identity化）した特製ResNet。
    """
    def __init__(self, num_classes=50):
        super().__init__()
        # PyTorch標準のResNet18をベースラインとしてロード (weights=None)
        self.resnet = models.resnet18(weights=None)
        
        # 1. 入力チャンネルを「4」に変更、かつ高解像度維持のため stride=1 に設定
        self.resnet.conv1 = nn.Conv2d(
            in_channels=4, 
            out_channels=64, 
            kernel_size=7, 
            stride=1, 
            padding=3, 
            bias=False
        )
        
        # 2. 初期解像度を急激に1/4へ落とす MaxPool を完全にバイパス (無効化)
        self.resnet.maxpool = nn.Identity()
        
        # 3. 最終出力クラス数を 50 に変更
        self.resnet.fc = nn.Linear(512, num_classes)
        
        self._init_custom_weights()

    def _init_custom_weights(self):
        nn.init.kaiming_normal_(self.resnet.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.normal_(self.resnet.fc.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.resnet.fc.bias, 0.0)

    def forward(self, x):
        return self.resnet(x)


# ====================================================
# ─── 🏃 トレーニングメイン処理 (FP32 + ASAM) ───
# ====================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # 高解像度維持型 ResNet18 の初期化
    model = HighResResNet18(num_classes=50).to(device)
    
    g_init = torch.Generator()
    g_init.manual_seed(SEED)
    
    train_loader = DataLoader(
        LocalOnlyDataset(mode="train"), 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g_init
    )
    test_loader = DataLoader(
        LocalOnlyDataset(mode="test"), 
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
        f.write("=== Local-Only High-Resolution ResNet18 Training Log ===\n")

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
            
            # クラスごとの正答率追跡用の辞書
            class_correct = np.zeros(50)
            class_total = np.zeros(50)

            with torch.no_grad():
                for x_l, labels in test_loader:
                    x_l, labels = x_l.to(device), labels.to(device)
                    outputs = model(x_l)
                    test_total += labels.size(0)
                    acc1, _ = calculate_topk_accuracy(outputs, labels, topk=(1, 5))
                    test_top1 += acc1
                    
                    # 🔍 クラス別正答率の計算
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

            # 🔍 特に注目したい携帯(14)と敬礼(36)のクラス別精度をログ出力
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