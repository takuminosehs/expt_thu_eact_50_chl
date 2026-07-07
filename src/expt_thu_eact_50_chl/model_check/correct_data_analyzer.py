# src/expt_thu_eact_50_chl/analyzer.py
import json
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# プロジェクトルートのパス解決とインポート環境の整備
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def analyze_predictions(
    model_path: Path, 
    data_paths: list[Path], 
    model_instance: nn.Module = None
) -> list[dict]:
    """
    指定されたモデルを用いてテストデータリストの予測を行い、詳細な正誤結果を辞書のリストで返却する。
    
    Args:
        model_path (Path): 保存されているモデル（.pth）のパス
        data_paths (list[Path]): テストデータのPathオブジェクトのリスト
        model_instance (nn.Module, optional): state_dictのみをロードする場合に必要なモデルインスタンス
        
    Returns:
        list[dict]: 各データの詳細な予測結果が格納された辞書のリスト
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. モデルのロード処理（重みのみ / モデル全体の双方に対応）
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        # 独自フォーマットのチェックポイント対策
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        state_dict = None
        model = checkpoint  # モデルオブジェクトそのものが保存されている場合
        
    if state_dict is not None:
        if model_instance is None:
            raise ValueError(
                f"❌ ロードされたファイル '{model_path.name}' は state_dict (重みのみ) です。\n"
                f"この関数を呼び出す際は、引数 'model_instance' に該当するモデル構造オブジェクトを渡してください。"
            )
        model = model_instance
        # 🌟 【追加】計測ライブラリ（thop等）が埋め込んだ不要なメタデータキーを除外する
        clean_state_dict = {
            k: v for k, v in state_dict.items() 
            if "total_ops" not in k and "total_params" not in k
        }
        
        # 綺麗にした重みをロード
        model.load_state_dict(clean_state_dict)
        
    model = model.to(device)
    model.eval()
    
    results = []
    
    print(f"🧠 モデル '{model_path.name}' によるデータ検証を開始します（データ数: {len(data_paths)}）...")
    
    # 2. 推論ループ
    with torch.no_grad():
        for file_path in data_paths:
            if not file_path.exists():
                print(f"⚠️ 警告: ファイルが見つかりません。スキップします: {file_path}")
                continue
                
            # 正解ラベルの取得 (ファイル名末尾の数字から取得)
            # 例: "0_xxx_ch4_label_12.npy" -> "12"
            try:
                label_str = file_path.stem.split("_label_")[-1]
                ground_truth = int(label_str)
            except (ValueError, IndexError) as e:
                raise ValueError(
                    f"❌ ファイル名から正解ラベルを正しく抽出できませんでした: {file_path.name}\n"
                    f"ファイル名が '*_label_{{数字}}.npy' になっているか確認してください。"
                ) from e
                
            # データの読み込み
            features = np.load(file_path)
            
            # 推論時正規化 (train.py のレシピと完全統一)
            max_val = np.max(np.abs(features))
            if max_val > 0:
                features = features / max_val
                
            # Tensor化型変換 [C, T, H] -> [1, C, T, H]
            input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
            
            # --- 形状の整合性チェック (ミスマッチ時は明示的にエラーを発生) ---
            try:
                outputs = model(input_tensor)
            except RuntimeError as e:
                raise RuntimeError(
                    f"❌ モデルと入力データの形状が整合していません。\n"
                    f"データファイル: {file_path.name} (形状: {features.shape})\n"
                    f"エラー詳細: {e}"
                ) from e
                
            # 確率値（確信度）と予測クラスの算出
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_tensor = torch.max(probabilities, dim=1)
            
            predicted = int(predicted_tensor.item())
            conf_value = float(confidence.item())
            is_correct = bool(predicted == ground_truth)
            
            # 取得できた全ての情報を格納
            results.append({
                "file_path": str(file_path.resolve()),
                "filename": file_path.name,
                "ground_truth": ground_truth,
                "predicted": predicted,
                "is_correct": is_correct,
                "confidence": round(conf_value, 6)
            })
            
    return results


def analyze_and_export_to_json(
    model_path: Path, 
    data_dir: Path, 
    output_json_path: Path,
    model_instance: nn.Module = None
) -> None:
    """
    指定されたディレクトリ内のすべてのデータを解析し、結果をJSONファイルとして保存するラッパー関数。
    """
    model_path = Path(model_path)
    data_dir = Path(data_dir)
    output_json_path = Path(output_json_path)
    
    # ディレクトリ内のすべての .npy ファイルを検索
    data_paths = sorted(list(data_dir.glob("*.npy")))
    
    if not data_paths:
        print(f"⚠️ 指定されたディレクトリに .npy ファイルが見つかりませんでした: {data_dir}")
        return
        
    # 解析実行
    analysis_results = analyze_predictions(
        model_path=model_path, 
        data_paths=data_paths, 
        model_instance=model_instance
    )
    
    # 親ディレクトリが存在しない場合は作成
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    
    # JSONとして綺麗に整形（インデント付き）して書き出し
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=4)
        
    print(f"🎉 解析結果が正常に保存されました ➡️ {output_json_path.resolve()}")