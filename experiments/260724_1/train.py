# src/holoev-net-strict-re/train.py
import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.models as models
from pathlib import Path

# utils.py が同じディレクトリ内にある前提です
from src.utils.utils import measure_model_complexity, measure_inference_latency, calculate_topk_accuracy, save_best_model

CURRENT_DIR = Path(__file__).parent.resolve()
# データは前回の holoev-net-re のものを使い回せます
PROCESSED_DIR = CURRENT_DIR / "processed_data" 
MODEL_SAVE_PATH = CURRENT_DIR / "best_model_strict.pth"

# ====================================================
# 1. 専用データセットクラス (🌟 正規化を削除)
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
        
        # [3, 224, 260] のCHSR配列 (生のカウント値と位相)
        features = np.load(file_path)
        
        # 論文に記載がないため、以前おこなっていた最大値による正規化処理を削除
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# ====================================================
# 2. Global Spectral Gating (GSG) モジュール (変更なし)
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
# 3. HoloEv-Net 本体 (変更なし)
# ====================================================
class HoloEvNetBaseOriginal(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        resnet = models.resnet18(weights=None)
        
        self.conv1 = resnet.conv1 # 3チャネル入力
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
# 4. 学習ループ (🌟 論文の条件に完全一致)
# ====================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    
    INPUT_SIZE = (1, 3, 224, 260)
    NUM_CLASSES = 50
    
    model = HoloEvNetBaseOriginal(num_classes=NUM_CLASSES).to(device)
    
    macs, params = measure_model_complexity(model, input_size=INPUT_SIZE, device=device)
    latency_ms = measure_inference_latency(model, input_size=INPUT_SIZE, device=device)
    
    train_dataset = HoloEvDataset(mode="train")
    test_dataset = HoloEvDataset(mode="test")
    
    # 🌟 バッチサイズを 32 に設定 (論文準拠) 
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    criterion = nn.CrossEntropyLoss()
    # 🌟 AdamW ではなく標準の Adam を使用し、学習率を 1e-4 に設定 (論文準拠) 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 🌟 エポック数を 50 に設定 (論文準拠) 
    num_epochs = 50
    best_test_acc1 = 0.0
    log_filename = CURRENT_DIR / "result_strict.txt"

    with open(log_filename, "w", encoding="utf-8") as f:
        f.write("=== THUE-ACT-50 CHL HoloEv-Net (Strict Paper Reproduction) Log ===\n")
        if params is not None:
            f.write(f"Model Params: {params/1e6:.2f} M\n")
            f.write(f"Model FLOPs: {macs/1e9:.2f} G\n")
        f.write(f"Inference Latency: {latency_ms:.2f} ms\n")
        f.write("=================================================\n")
            
        print(f"🚀 論文と全く同じ条件（Adam, 1e-4, Batch 32）による特訓を開始します")
        
        for epoch in range(num_epochs):
            model.train()
            train_loss, train_total = 0.0, 0
            train_top1, train_top5 = 0.0, 0.0
            
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
                f"Epoch {epoch+1:03d}/{num_epochs} -> Loss: {train_loss/train_total:.4f} | "
                f"Train Top-1: {tr_acc1_pct:.2f}% (Top-5: {tr_acc5_pct:.2f}%) | "
                f"★Test Top-1: {te_acc1_pct:.2f}% (Top-5: {te_acc5_pct:.2f}%)"
            )
            print(epoch_status)
            f.write(epoch_status + "\n")
            best_test_acc1 = save_best_model(model, te_acc1_pct, best_test_acc1, MODEL_SAVE_PATH)
            f.flush()

    print(f"\n🎉 厳密な論文ベースラインの学習が完了しました！最高の Test Top-1 精度は {best_test_acc1:.2f}% でした。")