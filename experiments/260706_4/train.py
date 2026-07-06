# train.py (v-3 with Label Smoothing + Cosine Annealing)
import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.models as models
from pathlib import Path

# utils.py が配置されている前提です
from expt_thu_eact_50_chl.utils import measure_model_complexity, measure_inference_latency, calculate_topk_accuracy, save_best_model

CURRENT_DIR = Path(__file__).parent.resolve()
PROCESSED_DIR = CURRENT_DIR / "processed_data"
MODEL_SAVE_PATH = CURRENT_DIR / "best_model.pth"

# ====================================================
# 1. 専用データセットクラス
# ====================================================
class HoloEvDataset(Dataset):
    def __init__(self, mode="train"):
        self.dir_path = PROCESSED_DIR / mode
        self.file_paths = glob.glob(os.path.join(self.dir_path, "*_orig_*.npy"))
        
    def __len__(self):
        return len(self.file_paths)
        
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label_str = file_path.split("_label_")[-1].split(".npy")[0]
        label = int(label_str.replace("A", ""))
        
        # v-3 の 4チャネル CHSR 配列
        features = np.load(file_path)
        
        # 最大値による正規化（高精度レシピ）
        max_val = np.max(np.abs(features))
        if max_val > 0:
            features = features / max_val
            
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# ====================================================
# 2. Global Spectral Gating (GSG) モジュール
# ====================================================
class GlobalSpectralGating(nn.Module):
    def __init__(self, channels, T_prime, H_prime):
        super().__init__()
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.ln = nn.LayerNorm([channels, T_prime, H_prime])
        self.complex_weight = nn.Parameter(
            torch.randn(channels, T_prime, H_prime // 2 + 1, dtype=torch.cfloat) * 0.02
        )
        self.gate_conv = nn.Conv2d(channels, channels, kernel_size=1)
        
    def forward(self, x):
        x_local = self.dw_conv(x)
        x_freq = torch.fft.rfft2(x_local, dim=(2, 3), norm="ortho")
        x_freq = x_freq * self.complex_weight
        z_tilde = torch.fft.irfft2(x_freq, s=(x.size(2), x.size(3)), dim=(2, 3), norm="ortho")
        carrier = F.silu(self.ln(z_tilde))
        gate = torch.sigmoid(self.gate_conv(z_tilde))
        return carrier * gate

# ====================================================
# 3. HoloEv-Net 本体 (4チャネル入力 ResNet-18)
# ====================================================
class HoloEvNetBaseV3(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        resnet = models.resnet18(weights=None)
        
        # 4チャネル入力
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.gsg = GlobalSpectralGating(channels=512, T_prime=7, H_prime=9)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)
        
    def forward(self, v_tensor):
        x = self.conv1(v_tensor)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_in = self.layer4(x)
        
        x_gsg = self.gsg(x_in)
        x_out = x_in + x_gsg
        
        feat = self.avgpool(x_out).view(x_out.size(0), -1)
        return self.classifier(feat)

# ====================================================
# 4. 学習ループ
# ====================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    
    INPUT_SIZE = (1, 4, 224, 260)
    NUM_CLASSES = 50
    num_epochs = 200 # 🌟コサイン波でじっくり減衰させるため、200エポック回します
    
    model = HoloEvNetBaseV3(num_classes=NUM_CLASSES).to(device)
    
    macs, params = measure_model_complexity(model, input_size=INPUT_SIZE, device=device)
    latency_ms = measure_inference_latency(model, input_size=INPUT_SIZE, device=device)
    
    train_dataset = HoloEvDataset(mode="train")
    test_dataset = HoloEvDataset(mode="test")
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    
    # 成果を出した Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
    
    # 🌟 今回追加する学習率スケジューラ (Cosine Annealing)
    # T_max は減衰がゼロに達するまでのステップ数（エポック数）を指定します
    # eta_min は最小学習率（ここでは完全にゼロまで落とします）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
    
    best_test_acc1 = 0.0
    log_filename = CURRENT_DIR / "result_v3_ls_cos.txt"

    with open(log_filename, "w", encoding="utf-8") as f:
        f.write("=== THUE-ACT-50 CHL HoloEv-Net-V3 (4-Ch) LS + Cosine Annealing ===\n")
        if params is not None:
            f.write(f"Model Params: {params/1e6:.2f} M\n")
            f.write(f"Model FLOPs: {macs/1e9:.2f} G\n")
        f.write(f"Inference Latency: {latency_ms:.2f} ms\n")
        f.write("=================================================\n")
            
        print(f"🚀 V3 (4チャネル) + LS + Cosine スケジューラによる最終特訓を開始します")
        
        for epoch in range(num_epochs):
            model.train()
            train_loss, train_total = 0.0, 0
            train_top1, train_top5 = 0.0, 0.0
            
            # 現在のエポックの学習率を取得して表示
            current_lr = optimizer.param_groups[0]['lr']
            
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * features.size(0)
                train_total += labels.size(0)
                acc1, acc5 = calculate_topk_accuracy(outputs, labels, topk=(1, 5))
                train_top1 += acc1
                train_top5 += acc5
                
            # 🌟 エポックの終了時にスケジューラを更新（学習率をコサイン波に従って減衰）
            scheduler.step()
                
            model.eval()
            test_total = 0
            test_top1, test_top5 = 0.0, 0.0
            
            with torch.no_grad():
                for features, labels in test_loader:
                    features, labels = features.to(device), labels.to(device)
                    outputs = model(features)
                    test_total += labels.size(0)
                    acc1, acc5 = calculate_topk_accuracy(outputs, labels, topk=(1, 5))
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
            best_test_acc1 = save_best_model(model, te_acc1_pct, best_test_acc1, MODEL_SAVE_PATH)
            f.flush()

    print(f"\n🎉 すべての学習が完了しました！最高の Test Top-1 精度は {best_test_acc1:.2f}% でした。")