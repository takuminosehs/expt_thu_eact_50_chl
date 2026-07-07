import json
from pathlib import Path
from typing import Optional

# 制約: config.pyのインポート
from expt_thu_eact_50_chl import config


def get_reduction_rate_ranking(metrics_data: list[dict], n: int = 5) -> dict:
    """
    JSONから読み込んだメトリクスデータのリストを受け取り、
    削減率の高い順（top）と低い順（bottom）のN件を抽出して返す関数。
    
    Parameters
    ----------
    metrics_data : list[dict]
        denoise_metrics.json から読み込まれたデータのリスト
    n : int, default 5
        上位・下位それぞれ何件取得するか
        
    Returns
    -------
    dict
        "top_reductions" と "bottom_reductions" をキーに持ち、
        それぞれ順位(rank)と元の詳細情報を結合した辞書のリストを格納した辞書
    """
    # 成功したデータのみを抽出（エラーデータには reduction_rate_percent が存在しないため）
    valid_data = [d for d in metrics_data if d.get("status") == "success" and "reduction_rate_percent" in d]
    
    # 削減率で降順（高い順）にソート
    sorted_desc = sorted(valid_data, key=lambda x: x["reduction_rate_percent"], reverse=True)
    # 削減率で昇順（低い順）にソート
    sorted_asc = sorted(valid_data, key=lambda x: x["reduction_rate_percent"])
    
    # 上位データの構築
    top_n_data = []
    for i, item in enumerate(sorted_desc[:n], start=1):
        # 順位情報を先頭に配置するために新しい辞書を作成
        ranked_item = {"rank": i}
        ranked_item.update(item)
        top_n_data.append(ranked_item)
        
    # 下位データの構築
    bottom_n_data = []
    for i, item in enumerate(sorted_asc[:n], start=1):
        ranked_item = {"rank": i}
        ranked_item.update(item)
        bottom_n_data.append(ranked_item)
        
    return {
        "top_reductions": top_n_data,
        "bottom_reductions": bottom_n_data
    }


def generate_ranking_file(
    input_path: Optional[Path] = None, 
    output_path: Optional[Path] = None, 
    n: int = 5
) -> Path:
    """
    入出力のファイルパスを解決し、JSONの読み書きを行うラッパー関数。
    
    Parameters
    ----------
    input_path : Path, optional
        読み込む対象のJSONファイルパス。未指定時は実行ディレクトリの denoise_metrics.json
    output_path : Path, optional
        出力するJSONファイルパス。未指定時は実行ディレクトリの ranking_reduction_rate.json
    n : int, default 5
        上位・下位何位までを出力するか
        
    Returns
    -------
    Path
        出力されたファイルのパス
    """
    current_dir = Path(__file__).parent
    
    # パスが指定されていない場合のデフォルトフォールバック
    input_file = Path(input_path) if input_path else current_dir / "denoise_metrics.json"
    output_file = Path(output_path) if output_path else current_dir / "ranking_reduction_rate.json"
        
    if not input_file.exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {input_file}")
        
    # データの読み込み
    with open(input_file, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
        
    # コアロジックの実行
    ranking_result = get_reduction_rate_ranking(metrics_data, n=n)
    
    # データの書き込み
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(ranking_result, f, indent=4, ensure_ascii=False)
        
    print(f"[INFO] ランキング上位・下位 {n} 件のデータを保存しました: {output_file}")
    
    return output_file


if __name__ == "__main__":
    # モジュールとしてではなく直接実行された場合のデフォルト挙動
    generate_ranking_file(n=5)