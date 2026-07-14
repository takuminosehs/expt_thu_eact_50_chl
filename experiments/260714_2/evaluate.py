import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from tqdm import tqdm

# ====================================================
# ─── 🏗️ モデル定義 (train_4stage.py と完全に同一) ───
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

class HoloEvNetFourStageGated(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
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

        self.gating_layer = nn.Sequential(nn.Linear(512, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x_global, x_local, mode='both'):
        pass


# ====================================================
# ─── 📂 データセット定義 ───
# ====================================================
class HoloEvTwinInferenceDataset(Dataset):
    def __init__(self, test_dir: Path):
        self.global_dir = test_dir / "global"
        self.local_dir = test_dir / "local"
        self.file_paths = list(self.global_dir.glob("*.npy"))
        self.file_names = [p.name for p in self.file_paths]

        if not self.file_paths:
            raise FileNotFoundError(f"データが見つかりません: {self.global_dir}")

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
            torch.tensor(label, dtype=torch.long),
            f_name
        )


# ====================================================
# ─── 🚀 メイン評価ロジック ───
# ====================================================
def main():
    CURRENT_DIR = Path(__file__).parent.resolve()

    parser = argparse.ArgumentParser(description="4ステージモデルの予測確率とゲート重みを詳細出力するスクリプト")
    parser.add_argument("--test-dir", type=str, default=str(CURRENT_DIR / "processed_data" / "test"))
    parser.add_argument("--model-path", type=str, default=str(CURRENT_DIR / "best_model_augmented.pth"))
    parser.add_argument("--output-dir", type=str, default=str(CURRENT_DIR))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 使用デバイス: {device}")

    print(f"[*] 新しい4ステージモデルをロード中: {model_path}")
    model = HoloEvNetFourStageGated(num_classes=50)
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    dataset = HoloEvTwinInferenceDataset(test_dir=test_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    results = []

    print("[*] 推論および各ストリームの予測抽出を開始...")
    with torch.no_grad():
        for x_g, x_l, labels, f_names in tqdm(dataloader):
            x_g, x_l = x_g.to(device), x_l.to(device)
            
            with torch.amp.autocast(device_type=device.type):
                # 1. Global ストリーム順伝播
                g = model.g_conv1(x_g)
                g = model.g_bn1(g)
                g = model.g_relu(g)
                g = model.g_maxpool(g)
                g = model.g_layer1(g)
                g = model.g_layer2(g)
                g = model.g_layer3(g)
                g_in = model.g_layer4(g)
                g_in = model.g_spatial4(g_in)
                g_out = g_in + model.g_gsg(g_in)
                feat_global = model.g_avgpool(g_out).view(g_out.size(0), -1)
                y_global = model.g_classifier(feat_global)

                # 2. Local ストリーム順伝播
                l = model.l_conv1(x_l)
                l = model.l_bn1(l)
                l = model.l_relu(l)
                l = model.l_maxpool(l)
                l = model.l_layer1(l)
                l = model.l_layer2(l)
                l = model.l_layer3(l)
                l = model.l_layer4(l)
                feat_local = model.l_avgpool(l).view(l.size(0), -1)
                y_local = model.l_classifier(feat_local)

                # 3. ゲート結合の計算
                gate_logits = model.gating_layer(feat_global)
                gate_weights = F.softmax(gate_logits, dim=1)
                alpha_g, alpha_l = gate_weights[:, 0:1], gate_weights[:, 1:2]

                # 4. 最終融合予測
                y_final = alpha_g * y_global + alpha_l * y_local

                # 確率値へ変換
                y_final_probs = F.softmax(y_final, dim=1)
                y_global_probs = F.softmax(y_global, dim=1)
                y_local_probs = F.softmax(y_local, dim=1)

                confidences, predicted_labels = torch.max(y_final_probs, dim=1)

            # CPU転送
            confidences = confidences.cpu().numpy()
            predicted_labels = predicted_labels.cpu().numpy()
            labels = labels.numpy()
            alpha_g = alpha_g.cpu().numpy().squeeze(axis=1)
            alpha_l = alpha_l.cpu().numpy().squeeze(axis=1)
            y_global_probs = y_global_probs.cpu().numpy()
            y_local_probs = y_local_probs.cpu().numpy()

            for i in range(len(f_names)):
                is_correct = bool(predicted_labels[i] == labels[i])
                
                res_dict = {
                    "ファイル名": f_names[i],
                    "正解ラベル": int(labels[i]),
                    "予測ラベル": int(predicted_labels[i]),
                    "正誤": is_correct,
                    "最終確信度": float(confidences[i]),
                    "ゲート重み_Global": float(alpha_g[i]),
                    "ゲート重み_Local": float(alpha_l[i]),
                }
                
                for c in range(50):
                    res_dict[f"y_global_prob_{c}"] = float(y_global_probs[i, c])
                for c in range(50):
                    res_dict[f"y_local_prob_{c}"] = float(y_local_probs[i, c])
                
                results.append(res_dict)

    # 保存処理
    json_path = output_dir / "4stage_inference_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    df = pd.DataFrame(results)
    excel_path = output_dir / "4stage_inference_results.xlsx"
    df.to_excel(excel_path, index=False, engine="openpyxl")
    
    print(f"[✓] 解析完了！保存先: {excel_path}")
    print(f"[+] テスト精度: {df['正誤'].mean()*100:.2f}%")

if __name__ == "__main__":
    main()