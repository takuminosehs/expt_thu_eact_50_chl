import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.models as models
from pathlib import Path

# config.py から PROJECT_ROOT をインポート
from expt_thu_eact_50_chl.config import PROJECT_ROOT
# 既存の共通ユーティリティ関数
from expt_thu_eact_50_chl.utils import (
    measure_model_complexity, 
    measure_inference_latency, 
    calculate_topk_accuracy, 
    save_best_model
)

# 260707_1 で作成済みの前処理データディレクトリを直接参照
PROCESSED_DIR = PROJECT_ROOT / "experiments" / "260707_1" / "processed_data"

# 今回の実験ディレクトリ（260709_2）のパス設定
CURRENT_DIR = Path(__file__).parent.resolve()
MODEL_SAVE_PATH = CURRENT_DIR / "best_model_cbam.pth"
LOG_FILENAME = CURRENT_DIR / "result_cbam.txt"

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
        
        features = np.load(file_path)
        
        # 最大値による正規化
        max_val = np.max(np.abs(features))
        if max_val > 0:
            features = features / max_val
            
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# ====================================================
# 2. CBAM (Convolutional Block Attention Module)
# ====================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        max_avg = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(max_avg))

class CBAM(nn.Module):
    def __init__(self, channels, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# ====================================================
# 3. Global Spectral Gating (GSG) モジュール (AMP・cfloat不具合修正済)
# ====================================================
class GlobalSpectralGating(nn.Module):
    def __init__(self, channels, T_prime, H_prime):
        super().__init__()
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.ln = nn.LayerNorm([channels, T_prime, H_prime])
        
        # 複素数勾配エラー対策: 実部と虚部を分離して保持
        self.weight_real = nn.Parameter(
            torch.randn(channels, T_prime, H_prime // 2 + 1, dtype=torch.float32) * 0.02
        )
        self.weight_imag = nn.Parameter(
            torch.randn(channels, T_prime, H_prime // 2 + 1, dtype=torch.float32) * 0.02
        )
        self.gate_conv = nn.Conv2d(channels, channels, kernel_size=1)
        
    def forward(self, x):
        x_local = self.dw_conv(x)
        
        # cuFFTサイズ制約対策: 一時的にFloat32にアップキャスト
        orig_dtype = x_local.dtype
        x_local_fp32 = x_local.float()
        
        x_freq = torch.fft.rfft2(x_local_fp32, dim=(2, 3), norm="ortho")
        
        # フォワード計算直前に複素数として動的結合
        complex_weight = torch.complex(self.weight_real, self.weight_imag)
        x_freq = x_freq * complex_weight
        
        z_tilde = torch.fft.irfft2(x_freq, s=(x.size(2), x.size(3)), dim=(2, 3), norm="ortho")
        z_tilde = z_tilde.to(orig_dtype)
        
        carrier = F.silu(self.ln(z_tilde))
        gate = torch.sigmoid(self.gate_conv(z_tilde))
        return carrier * gate

# ====================================================
# 4. HoloEv-Net V4 (CBAM 深層限定アプローチモデル)
# ====================================================
class HoloEvNetBaseV4_CBAM(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        resnet = models.resnet18(weights=None)
        
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # 🌟 浅い層（Layer 1, 2）は基本特徴抽出に専念させ、CBAMをあえて適用しない
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        
        # 🌟 セマンティクスが固まる深い層（Layer 3, 4）にのみCBAMを導入
        self.layer3 = resnet.layer3
        self.cbam3 = CBAM(256)
        
        self.layer4 = resnet.layer4
        self.cbam4 = CBAM(512)
        
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
        
        # 深い層で適応的な空間・チャンネルへのアテンションを発動
        x = self.layer3(x)
        x = self.cbam3(x)
        
        x_in = self.layer4(x)
        x_in = self.cbam4(x_in)
        
        x_gsg = self.gsg(x_in)
        x_out = x_in + x_gsg
        
        feat = self.avgpool(x_out).view(x_out.size(0), -1)
        return self.classifier(feat)

# ====================================================
# 5. 学習ループ
# ====================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    
    INPUT_SIZE = (1, 4, 224, 260)
    NUM_CLASSES = 50
    num_epochs = 100
    
    model = HoloEvNetBaseV4_CBAM(num_classes=NUM_CLASSES).to(device)
    
    macs, params = measure_model_complexity(model, input_size=INPUT_SIZE, device=device)
    latency_ms = measure_inference_latency(model, input_size=INPUT_SIZE, device=device)
    
    train_dataset = HoloEvDataset(mode="train")
    test_dataset = HoloEvDataset(mode="test")
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
    
    scaler = torch.amp.GradScaler()
    best_test_acc1 = 0.0

    with open(LOG_FILENAME, "w", encoding="utf-8") as f:
        f.write("=== THUE-ACT-50 CHL HoloEv-Net-V4 (CBAM 深層限定版) ===\n")
        if params is not None:
            f.write(f"Model Params: {params/1e6:.2f} M\n")
            f.write(f"Model FLOPs: {macs/1e9:.2f} G\n")
        f.write(f"Inference Latency: {latency_ms:.2f} ms\n")
        f.write("=================================================\n")
            
        print(f"🚀 V4 (深層限定CBAM) による実験を開始します")
        
        for epoch in range(num_epochs):
            model.train()
            train_loss, train_total = 0.0, 0
            train_top1, train_top5 = 0.0, 0.0
            
            current_lr = optimizer.param_groups[0]['lr']
            
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad()
                
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item() * features.size(0)
                train_total += labels.size(0)
                acc1, acc5 = calculate_topk_accuracy(outputs, labels, topk=(1, 5))
                train_top1 += acc1
                train_top5 += acc5
                
            scheduler.step()
                
            model.eval()
            test_total = 0
            test_top1, test_top5 = 0.0, 0.0
            
            with torch.no_grad():
                for features, labels in test_loader:
                    features, labels = features.to(device), labels.to(device)
                    with torch.amp.autocast(device_type=device.type):
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