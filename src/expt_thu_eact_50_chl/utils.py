# src/holoev-net-v-1/utils.py
import torch
import time
import os
try:
    from thop import profile, clever_format
except ImportError:
    raise ImportError("thop がインストールされていません。ターミナルで 'pip install thop' を実行してください。")

def measure_model_complexity(model, input_size, device="cpu"):
    """モデルのパラメータ数とFLOPs(MACs)を測定する汎用関数"""
    model.eval()
    dummy_input = torch.randn(*input_size).to(device)
    model = model.to(device)
    
    print("\n--- 📊 モデル計算量プロファイリング ---")
    try:
        macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
        macs_str, params_str = clever_format([macs, params], "%.2f")
        print(f"Model: {model.__class__.__name__}")
        print(f"Input Shape: {input_size}")
        print(f"Parameters: {params_str}")
        print(f"FLOPs (MACs): {macs_str}")
        print("--------------------------------------\n")
        return macs, params
    except Exception as e:
        print(f"⚠️ 計算量の測定に失敗しました: {e}")
        return None, None

def measure_inference_latency(model, input_size, device="cpu", num_samples=100, warmup=10):
    """
    モデルの推論レイテンシ（ms）を測定する汎用関数。
    入力サイズやデバイスに依存せず、GPUの非同期実行も考慮して正確に計測します。
    """
    model.eval()
    dummy_input = torch.randn(*input_size).to(device)
    model = model.to(device)
    
    # ウォームアップ（GPUの初期化オーバーヘッドを排除）
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
            
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    start_time = time.time()
    
    # 本計測
    with torch.no_grad():
        for _ in range(num_samples):
            _ = model(dummy_input)
            
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    end_time = time.time()
    
    avg_latency_ms = ((end_time - start_time) / num_samples) * 1000
    print(f"⏱️ 推論レイテンシ (Batch Size {input_size[0]}): {avg_latency_ms:.2f} ms")
    return avg_latency_ms

def calculate_topk_accuracy(output, target, topk=(1, 5)):
    """
    Top-k 精度を計算する汎用関数。
    クラス数が少ないデータセットでもエラーにならないように保護しています。
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        # モデルの出力クラス数が指定した maxk より少ない場合の安全策
        maxk = min(maxk, output.size(1))
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            if k <= output.size(1):
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.item())
            else:
                res.append(0.0) # クラス数以上のkを要求された場合は0を返す
        return res

def save_best_model(model, current_acc, best_acc, save_path):
    """
    現在の精度が過去最高であればモデルの重みを保存する汎用関数。
    """
    if current_acc > best_acc:
        print(f"🌟 Best Accuracy 更新! ({best_acc:.2f}% -> {current_acc:.2f}%) 重みを保存しました: {os.path.basename(save_path)}")
        torch.save(model.state_dict(), save_path)
        return current_acc
    return best_acc