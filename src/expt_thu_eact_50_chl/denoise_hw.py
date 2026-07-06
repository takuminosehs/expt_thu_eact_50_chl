from pathlib import Path
from typing import Optional
import numpy as np
import numba

# 制約: config.pyのインポート（プロジェクトルートからの相対/絶対パスを通す前提）
# import sys
# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.append(str(PROJECT_ROOT))
from expt_thu_eact_50_chl import config


@numba.njit(parallel=False, cache=True)
def _local_spatiotemporal_denoise_jit(
    events: np.ndarray, 
    L: int = 5, 
    dt: int = 5000, 
    psi: int = 5
) -> np.ndarray:
    """
    論文の手法を高速に実行するためのJITコンパイル用ループ関数。
    
    Parameters
    ----------
    events : np.ndarray
        (N, 4) の配列 (x, y, t, p)。tは通常マイクロ秒(us)を想定。
    L : int
        空間近傍のサイズ (L x L)
    dt : int
        時間窓 Δt (マイクロ秒単位)。5ms = 5000us想定。
    psi : int
        第1段階のノイズ判定閾値 ψ
        
    Returns
    -------
    np.ndarray
        各イベントが実イベントか否かを示すブールマスク (形状: (N,))
    """
    N = events.shape[0]
    mask = np.zeros(N, dtype=np.bool_)
    half_L = L // 2

    # イベントストリームは通常タイムスタンプ(t)に関して昇順にソートされている前提
    for i in range(N):
        x0 = int(events[i, 0])
        y0 = int(events[i, 1])
        t0 = events[i, 2]
        
        d = 0
        has_adjacent = False
        
        # 現在のイベントから過去方向に向かって時間窓の範囲を探索
        for j in range(i, -1, -1):
            tj = events[j, 2]
            
            # 時間窓 Δt を超えたら過去方向の探索を終了
            if t0 - tj > dt:
                break
                
            xj = int(events[j, 0])
            yj = int(events[j, 1])
            
            # 【第1段階】 L×L の空間近傍内のイベント数をカウント (自身も含む)
            if abs(xj - x0) <= half_L and abs(yj - y0) <= half_L:
                d += 1
                
                # 【第2段階】 隣接画素（自身を除く周囲8画素）にイベントが存在するか (R > 0)
                if (xj != x0 or yj != y0) and abs(xj - x0) <= 1 and abs(yj - y0) <= 1:
                    has_adjacent = True
                    
        # 論文の条件：背景アクティビティ除去(d >= psi) かつ ホットピクセル除去(R != 0)
        if d >= psi and has_adjacent:
            mask[i] = True
            
    return mask


def denoise_event_file(
    input_path: Path, 
    output_path: Optional[Path] = None, 
    L: int = 5, 
    dt_ms: float = 5.0, 
    psi: int = 5,
    timestamp_unit: str = "us"
) -> Path:
    """
    THU-EACT-50-CHLデータセットのnpyファイルを入力とし、
    論文の手法でノイズ除去を行った結果をnpyファイルとして出力する関数。
    
    Parameters
    ----------
    input_path : Path
        入力する .npy ファイルのパス (必須)
    output_path : Path, optional
        出力する .npy ファイルのパス。指定がない場合は自動生成。
    L : int, default 5
        空間近傍窓のサイズ
    dt_ms : float, default 5.0
        時間窓 Δt (ミリ秒単位)
    psi : int, default 5
        第1段階の閾値 ψ
    timestamp_unit : str, default "us"
        データセットのタイムスタンプの単位 ('us': マイクロ秒, 's': 秒)
        ※一般的なMetavision出力は 'us' です。
    """
    input_path = Path(input_path)
    
    # 制約: 出力パス未指定時の自動生成ロジック (_hw_filtered.npy)
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_hw_filtered.npy")
    else:
        output_path = Path(output_path)
        
    # データの読み込み
    if not input_path.exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {input_path}")
        
    events = np.load(input_path)
    
    # 形状チェック
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError(f"入力データの形状は (N, 4) である必要があります。現在の形状: {events.shape}")
    
    # タイムスタンプの単位変換 (JIT関数内部での計算用)
    if timestamp_unit == "us":
        dt = int(dt_ms * 1000)
    elif timestamp_unit == "s":
        dt = dt_ms / 1000.0
    else:
        dt = int(dt_ms * 1000) # デフォルトはマイクロ秒
        
    print(f"[INFO] フィルタリング処理を開始します... (イベント数: {events.shape[0]})")
    
    # 高速ノイズ除去の実行
    mask = _local_spatiotemporal_denoise_jit(events, L=L, dt=dt, psi=psi)
    filtered_events = events[mask]
    
    print(f"[INFO] 処理完了: {events.shape[0]} -> {filtered_events.shape[0]} イベント (削減率: {(1 - filtered_events.shape[0]/events.shape[0])*100:.2f}%)")
    
    # 結果の保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, filtered_events)
    
    return output_path


def convert_labels_file(input_path, output_path, suffix):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        # 空行でなければ処理
        if line.strip():
            # '.npy' を '_hw_filtered.npy' に置換
            new_line = line.replace('.npy', f'{suffix}.npy')
            new_lines.append(new_line)
            
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"変換が完了しました: {output_path}")


if __name__ == "__main__":
    # 使用例：config.pyに定義されたディレクトリからサンプルを処理する場合
    # ※実際のファイル名に合わせて変更してください
    sample_file = config.ORIGINAL_DATA_DIR / "A0P8C0-2021_11_02_21_47_15.npy"
    output_path = Path(__file__).resolve().parent / "result" / "A0P8C0-2021_11_02_21_47_15_hw_filtered.npy"
    if sample_file.exists():
        output = denoise_event_file(input_path=sample_file, output_path=output_path)
        print(f"[SUCCESS] 保存先: {output}")
    else:
        print(f"[NOTE] サンプルファイルが指定のデータディレクトリ内に見つかりません: {sample_file}")