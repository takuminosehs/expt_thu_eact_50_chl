import copy
import math
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
# 直近の前処理スクリプトで生成した高解像度4chデータのディレクトリを指定
PROCESSED_DIR = PROJECT_ROOT / "experiments" / "260717_3" / "processed_data"
CURRENT_DIR = Path(__file__).parent.resolve()

from expt_thu_eact_50_chl.utils import (
    calculate_topk_accuracy,
    save_best_model,
)

# ====================================================
# ─── 🎯 動的サブスペシャリスト設定 (最重要) ───
# ====================================================
# 前回の実験でGlobalモデルが誤答した「苦手な31クラス」を定義。
# 統合ステージの状況に応じて、このリストを動的に変更可能です。
TARGET_CLASSES = [
    0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19, 
    21, 22, 27, 29, 30, 34, 35, 36, 37, 39, 40, 41, 42, 46
]

# 🔍 質問1に対する安全設計: 対象クラスが1以下の場合はローカルの学習をバイパス
NUM_TARGET_CLASSES = len(TARGET_CLASSES)
if NUM_TARGET_CLASSES <= 1:
    print(f"⚠️ 対象クラス数が {NUM_TARGET_CLASSES} のため、ローカル専門モデルの構築をスキップします（グローバル単体で処理可能）。")
    exit(0)

# ─── ⚙️ ハイパーパラメータ設定 ───
NUM_EPOCHS = 300
# LR_INITIAL = 0.0003
LR_INITIAL = 0.0005
BATCH_SIZE = 4

MODEL_SAVE_PATH = CURRENT_DIR / "best_sub_specialist_model.pth"
LOG_FILENAME = CURRENT_DIR / "sub_specialist_result.txt"


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
# ─── 📊 サブスペシャリスト専用データセット定義 ───
# ----------------------------------------------------
class SubSpecialistLocalDataset(Dataset):
    """
    指定された TARGET_CLASSES のデータのみを厳格にフィルタリングし、
    元のラベル(0~49)を、ボトルネックのないローカル連続インデックス(0~K-1)へ動的に再マッピングするデータセット。
    """
    def __init__(self, mode="train", target_classes=None):
        self.local_dir = PROCESSED_DIR / mode / "local"
        if not self.local_dir.exists():
            raise FileNotFoundError(f"データディレクトリが見つかりません: {self.local_dir}")

        if target_classes is None:
            raise ValueError("target_classes のリストを指定してください。")
            
        self.target_set = set(target_classes)
        
        # ─── 順マッピング・逆マッピング辞書の動的生成 ───
        self.src_to_local = {src_cls: local_idx for local_idx, src_cls in enumerate(sorted(target_classes))}
        self.local_to_src = {local_idx: src_cls for src_cls, local_idx in self.src_to_local.items()}

        # 該当するクラスのファイルのみをフィルタリングして保持
        all_files = sorted([p.name for p in self.local_dir.glob("*.npy")])
        self.file_names = []
        
        for f_name in all_files:
            label_str = f_name.split("_label_")[-1].split(".npy")[0]
            src_label = int(label_str.replace("A", ""))
            
            if src_label in self.target_set:
                self.file_names.append((f_name, src_label))

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        f_name, src_label = self.file_names[idx]
        
        # 元のグローバルラベルから、ローカルの連続インデックス（0 ~ K-1）へ変換
        local_label = self.src_to_local[src_label]

        feat_l = np.load(self.local_dir / f_name)  # 形状: (4, 260, 346)

        # テンソルの極値正規化
        ml = np.max(np.abs(feat_l))
        if ml > 0: 
            feat_l = feat_l / ml

        return (
            torch.tensor(feat_l, dtype=torch.float32),
            torch.tensor(local_label, dtype=torch.long),
            torch.tensor(src_label, dtype=torch.long)  # デバッグ・検証用に元のラベルも返却
        )


# ----------------------------------------------------
# ─── 🧠 空洞アトラスBasicBlockモジュール ───
# ----------------------------------------------------
class DilatedBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dilation=1):
        super(DilatedBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, 
            padding=dilation, dilation=dilation, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, 
            padding=dilation, dilation=dilation, bias=False
        )
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


# ----------------------------------------------------
# ─── 🧠 高解像度維持×チャネル半減×空洞アトラスResNet18 ───
# ----------------------------------------------------
class HighResDilatedResNet18HalfWidth(nn.Module):
    """
    入力 [B, 4, 260, 346] から、動的に決定された [B, K] のロジットを出力する。
    """
    def __init__(self, num_classes):
        super(HighResDilatedResNet18HalfWidth, self).__init__()
        self.in_planes = 32

        self.conv1 = nn.Conv2d(4, 32, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Identity()

        # 空洞アトラス構造で解像度を100%維持しながら広域の視野（位置関係）を確保
        self.layer1 = self._make_layer(DilatedBasicBlock, 32,  2, stride=1, dilation=1)
        self.layer2 = self._make_layer(DilatedBasicBlock, 64,  2, stride=1, dilation=2)
        self.layer3 = self._make_layer(DilatedBasicBlock, 128, 2, stride=1, dilation=4)
        self.layer4 = self._make_layer(DilatedBasicBlock, 256, 2, stride=1, dilation=8)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 最終全結合層の次元数を動的なクラス数 K に設定
        self.fc = nn.Linear(256 * DilatedBasicBlock.expansion, num_classes)

        self._init_custom_weights()

    def _make_layer(self, block, planes, num_blocks, stride, dilation):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, dilation))
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
        out = self.maxpool(out)
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
    print(f"🎯 サブスペシャリスト学習対象クラス数: {NUM_TARGET_CLASSES} / 50")

    # 動的クラス数でモデルをビルド
    model = HighResDilatedResNet18HalfWidth(num_classes=NUM_TARGET_CLASSES).to(device)
    
    train_dataset = SubSpecialistLocalDataset(mode="train", target_classes=TARGET_CLASSES)
    test_dataset = SubSpecialistLocalDataset(mode="test", target_classes=TARGET_CLASSES)

    print(f"📊 フィルタリング後データ数 - Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    g_init = torch.Generator()
    g_init.manual_seed(SEED)
    
    # 全体を苦手クラスに絞り込んでいるため、通常のシャッフルで均等に重点学習されます
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
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
        f.write("=== Local-Only Sub-Specialist Gated ResNet18 Training Log ===\n")
        f.write(f"Target Classes ({NUM_TARGET_CLASSES} ch): {sorted(TARGET_CLASSES)}\n\n")

        for epoch in range(NUM_EPOCHS):
            # コサインアニーリングLR
            cos_factor = (1 + math.cos(math.pi * epoch / NUM_EPOCHS)) / 2
            current_lr = LR_INITIAL * cos_factor
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            # --- Training with ASAM ---
            model.train()
            train_loss, train_total = 0.0, 0
            train_top1 = 0.0

            for x_l, local_labels, _ in train_loader:
                x_l, local_labels = x_l.to(device), local_labels.to(device)
                
                # ─── 1st Step ───
                optimizer.zero_grad()
                outputs = model(x_l)
                loss = criterion(outputs, local_labels)
                loss.backward()
                optimizer.first_step(zero_grad=True)

                # ─── 2nd Step ───
                outputs_adv = model(x_l)
                loss_adv = criterion(outputs_adv, local_labels)
                loss_adv.backward()
                optimizer.second_step(zero_grad=True)

                train_loss += loss.item() * local_labels.size(0)
                train_total += local_labels.size(0)
                acc1, _ = calculate_topk_accuracy(outputs, local_labels, topk=(1, 5 if NUM_TARGET_CLASSES >= 5 else 1))
                train_top1 += acc1

            # --- Validation ---
            model.eval()
            test_total, test_top1 = 0, 0.0
            
            # 再マップされたローカル次元でのクラス別カウント
            class_correct = np.zeros(NUM_TARGET_CLASSES)
            class_total = np.zeros(NUM_TARGET_CLASSES)

            with torch.no_grad():
                for x_l, local_labels, src_labels in test_loader:
                    x_l, local_labels = x_l.to(device), local_labels.to(device)
                    outputs = model(x_l)
                    test_total += local_labels.size(0)
                    acc1, _ = calculate_topk_accuracy(outputs, local_labels, topk=(1, 5 if NUM_TARGET_CLASSES >= 5 else 1))
                    test_top1 += acc1
                    
                    preds = torch.argmax(outputs, dim=1)
                    for p, l in zip(preds, local_labels):
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

            # 🔍 再マッピングを逆変換し、本来の「携帯(14)」と「敬礼(36)」の純粋な識別精度をモニタリング
            class_log = "   🔍 [Target Class Accuracy Monitoring (Sub-Specialist)]\n"
            for src_cls in [14, 36]:
                if src_cls in train_dataset.src_to_local:
                    local_idx = train_dataset.src_to_local[src_cls]
                    tot = class_total[local_idx]
                    acc = (class_correct[local_idx] / tot * 100) if tot > 0 else 0.0
                    class_log += f"      Original Class {src_cls:02d}: {acc:.2f}% ({int(class_correct[local_idx])}/{int(tot)})\n"
            print(class_log, end="")
            f.write(class_log)

            best_test_acc1 = save_best_model(model, te_acc, best_test_acc1, MODEL_SAVE_PATH)
            f.flush()