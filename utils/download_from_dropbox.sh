#!/bin/bash

# 保存先ディレクトリ
TARGET_DIR="/home/nose/ghq/github.com/takuminosehs/expt_thu_eact_50_chl/data_hardvs"
mkdir -p "$TARGET_DIR"

# (ファイル名|URL) のリスト定義
DOWNLOAD_LIST=(
  "HARDVS001-010.zip|https://www.dropbox.com/scl/fo/8ipo3ezz10coen0acefiu/AOfV3_7jmNQ5zlAgeXZH88Q/HARDVS001-010.zip?rlkey=oqzgm8237acit1kqglfr4h4lt&dl=1"
  "HARDVS011-020.zip|https://www.dropbox.com/scl/fo/8ipo3ezz10coen0acefiu/AC6oViWZJv_8diZsNmcbIhY/HARDVS011-020.zip?rlkey=oqzgm8237acit1kqglfr4h4lt&dl=1"
  "HARDVS021-030.zip|https://www.dropbox.com/scl/fo/8ipo3ezz10coen0acefiu/AEsZkYKQ_OkDQwmsgp2BfFw/HARDVS021-030.zip?rlkey=oqzgm8237acit1kqglfr4h4lt&dl=1"
  "HARDVS031-040.zip|https://www.dropbox.com/scl/fo/8ipo3ezz10coen0acefiu/AAXWFtd4YdosBgkrDu5_Lmk/HARDVS031-040.zip?rlkey=oqzgm8237acit1kqglfr4h4lt&dl=1"
  "HARDVS041-050.zip|https://www.dropbox.com/scl/fo/8ipo3ezz10coen0acefiu/ABLwDybBzjyFdmtjopojiZU/HARDVS041-050.zip?rlkey=oqzgm8237acit1kqglfr4h4lt&dl=1"
  "HARDVS051-060.zip|https://www.dropbox.com/scl/fo/8ipo3ezz10coen0acefiu/APJv_MI9uUK17z-H8ocpnGQ/HARDVS051-060.zip?rlkey=oqzgm8237acit1kqglfr4h4lt&dl=1"
  "HARDVS061-070.zip|https://www.dropbox.com/scl/fo/8ipo3ezz10coen0acefiu/AN0AeMpukSmgdlPQk-hmQTw/HARDVS061-070.zip?rlkey=oqzgm8237acit1kqglfr4h4lt&dl=1"
  "HARDVS071-080.zip|https://www.dropbox.com/scl/fo/8ipo3ezz10coen0acefiu/AEXRjLT3SlcFhOEzB2efYkM/HARDVS071-080.zip?rlkey=oqzgm8237acit1kqglfr4h4lt&dl=1"
  "HARDVS081-090.zip|https://www.dropbox.com/scl/fo/8ipo3ezz10coen0acefiu/ABZ3P6_MH9KSlYDt16ZZWmg/HARDVS081-090.zip?rlkey=oqzgm8237acit1kqglfr4h4lt&dl=1"
  "HARDVS091-100.zip|https://www.dropbox.com/scl/fo/8ipo3ezz10coen0acefiu/AKaPB36DL8qHNq1mudtVOuI/HARDVS091-100.zip?rlkey=oqzgm8237acit1kqglfr4h4lt&dl=1"
  "train_label.txt|https://www.dropbox.com/scl/fo/8ipo3ezz10coen0acefiu/ABPfKGIN3_6KoxWCD9CZzv8/train_label.txt?rlkey=oqzgm8237acit1kqglfr4h4lt&dl=1"
)

total=${#DOWNLOAD_LIST[@]}
count=1

echo "=========================================="
echo "一括ダウンロードを開始します (全 ${total} 件)"
echo "保存先: ${TARGET_DIR}"
echo "=========================================="

for item in "${DOWNLOAD_LIST[@]}"; do
  filename="${item%%|*}"
  url="${item#*|}"
  output_path="${TARGET_DIR}/${filename}"

  echo "[${count}/${total}] ${filename} をダウンロード中..."

  # -sS: プログレスバーを非表示（ログ肥大化防止）、エラー時のみ出力
  # -L : リダイレクトを自動追跡
  curl -sSL -o "${output_path}" "${url}"

  if [ $? -eq 0 ] && [ -f "${output_path}" ]; then
    echo " -> 完了"
  else
    echo " -> [エラー] ダウンロードに失敗しました"
  fi

  count=$((count + 1))
done

echo "=========================================="
echo "すべてのダウンロード処理が終了しました。"
echo "=========================================="