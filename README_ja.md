<h1 align="center">IOPaint — モダナイズ版フォーク</h1>
<p align="center">セルフホストで動く画像インペイント（消しゴム）・アウトペイント・オブジェクト除去・AI画像編集ツール</p>

<p align="center">
  <img alt="Python 3.10–3.13" src="https://img.shields.io/badge/Python-3.10--3.13-blue" />
  <img alt="Apache-2.0 license" src="https://img.shields.io/badge/license-Apache--2.0-blue" />
</p>

<p align="center"><a href="README.md">English</a> | 日本語</p>

本プロジェクトは [Sanster/IOPaint](https://github.com/Sanster/IOPaint) をベースにしたフォークです。オリジナルのワークフローとモデル構成はそのままに、Python / PyTorch / Hugging Face / フロントエンド / パッケージング / デプロイの全スタックを2026年基準にモダナイズしています。

| 入力 | 結果 |
|---|---|
| ![除去したいオブジェクト](assets/unwant_object.jpg) | ![除去後](assets/unwant_object_clean.jpg) |
| ![除去したい文字](assets/unwant_text.jpg) | ![除去後](assets/unwant_text_clean.jpg) |

## 動作要件

- Python `>=3.10,<3.14`
- PyTorch `>=2.4`
- CPU / NVIDIA CUDA / Apple Silicon に対応（モデルごとの対応状況に準じます）
- Blackwell 世代を含む新しめの NVIDIA GPU には **CUDA 12.8 以降ビルドの PyTorch** が必要です

モデルの重みは初回使用時に自動ダウンロードされます。保存先は `--model-dir` で変更でき、取得済みなら `--local-files-only` でオフライン起動できます。

## インストール

### Windows — ワンクリック起動（いちばん簡単）

[最新リリース](https://github.com/daraskme/IOpaint/releases/latest) から `IOPaint-OneClick.bat` をダウンロードしてダブルクリックするだけです。

- 初回実行時に自動で環境構築します：uv の導入 → Python 環境作成 → GPU の有無を判定して CUDA 12.8 版 / CPU 版 PyTorch をインストール → GitHub の最新リリースから IOPaint 本体をインストール
- 2回目以降はそのまま即起動し、ブラウザが自動で開きます
- インストール先は `%LOCALAPPDATA%\IOPaint` です。uv のキャッシュやダウンロードしたモデルの重みもすべてこの中に置かれるので、アンインストールはこのフォルダを削除するだけです

保存先を変えたい場合（数GBあるモデルを別ドライブに置きたい等）は、初回実行時に .bat の隣に生成される `IOPaint-OneClick.cfg` を編集するか、同名の環境変数を設定します:

```ini
# IOPaint-OneClick.cfg: 1行に KEY=VALUE、引用符なし、"#" で始まる行はコメント
# インストール先（Python 環境・uv キャッシュ・既定ではモデルも）
IOPAINT_HOME=D:\IOPaint
# モデルの保存先（iopaint に --model-dir として渡されます。HF / torch hub のキャッシュもこの中）
IOPAINT_MODEL_DIR=D:\IOPaint\models
# 起動するモデル、ポート、`iopaint start` に追加する引数
IOPAINT_MODEL=lama
IOPAINT_PORT=8080
IOPAINT_EXTRA_ARGS=--low-mem
# uv のダウンロードキャッシュと uv 管理の Python（既定は IOPAINT_HOME 配下）
UV_CACHE_DIR=D:\IOPaint\uv-cache
UV_PYTHON_INSTALL_DIR=D:\IOPaint\python
```

v2.0.0rc3 より前のランチャーはモデルを `%USERPROFILE%\.cache` に保存していました。そのフォルダが存在する場合、新しいランチャーは再ダウンロードを避けるためそのまま使い続けます。別の場所へ移したい場合はフォルダを移動して `IOPAINT_MODEL_DIR` を設定してください。

### uv — 推奨（macOS / Linux / Windows）

[uv](https://docs.astral.sh/uv/) をインストールした上で:

```bash
uv venv
uv pip install torch torchvision --torch-backend=auto
uv pip install iopaint-ng
iopaint start --model lama --device cuda --port 8080 --inbrowser
```

`--torch-backend=auto` が GPU 環境を自動判定して適切な PyTorch を選択します。GPU がない場合は `--device cpu` にしてください。

### pip

仮想環境を作成・有効化した上で、NVIDIA CUDA 12.8 の場合:

```bash
python -m pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install iopaint-ng
iopaint start --model lama --device cuda --port 8080 --inbrowser
```

CPU のみの場合は index URL を `https://download.pytorch.org/whl/cpu` に置き換え、`--device cpu` を指定します。

> **注記:** PyPI パッケージ名（現在は仮名 `iopaint-ng`）は正式公開前に変更される可能性があります。コマンド名は `iopaint` のまま変わりません。

### Docker

Git チェックアウトから両イメージをビルド:

```bash
bash scripts/build_docker.sh 2.0.0
```

NVIDIA GPU で実行（NVIDIA Container Toolkit が必要）:

```bash
docker run --rm --gpus all -p 8080:8080 \
  -v iopaint-cache:/root/.cache \
  iopaint-ng:2.0.0-cuda
```

## モデル構成

### 消しゴム系（軽量・高速）

`lama`（デフォルト） / `ldm` / `zits` / `mat` / `fcf` / `manga` / `cv2` / `migan`

### 拡散モデル系（生成的な補完・編集）

- Stable Diffusion / SDXL のインペイント（Hugging Face リポジトリまたはローカルの safetensors/ckpt 単一ファイル）
- BrushNet（SD / SDXL）、PowerPaint V1 / V2
- `Sanster/AnyText`（画像内テキスト生成）
- `Fantasy-Studio/Paint-by-Example`（見本画像による置換）
- Kandinsky 2.2 インペイント、InstructPix2Pix
- ControlNet 連携（canny / openpose / depth / inpaint）

モデルごとのオプションやメモリ節約設定は `iopaint start --help` を参照してください。

## プラグインとオプション依存

背景除去（最新の rembg / BiRefNet 系モデル対応）などのプラグイン依存は extra として分離されています:

```bash
uv pip install "iopaint-ng[plugins]"
```

利用可能なプラグイン: インタラクティブセグメンテーション（SAM 系）、RemoveBG / BiRefNet、アニメセグメンテーション、RealESRGAN（超解像）、GFPGAN / RestoreFormer（顔補正）。

Gradio ベースの設定 UI はオプションです:

```bash
uv pip install "iopaint-ng[web-config]"
iopaint start-web-config
```

## SAM3 セグメンテーション（このフォークの新機能）

SAM3 により、従来のクリック選択に加えて**テキスト指定によるコンセプトセグメンテーション**（例:「person」と指定して画像内の人物を全部選択）が使えます。重みはゲート付きです:

1. [facebook/sam3](https://huggingface.co/facebook/sam3) でライセンスに同意する
2. Hugging Face CLI でログインする:

   ```bash
   hf auth login
   ```

3. SAM3 を有効にして起動する:

   ```bash
   iopaint start --model lama --device cuda \
     --enable-interactive-seg \
     --interactive-seg-model sam3 \
     --interactive-seg-device cuda
   ```

クリック選択は既存のインタラクティブセグメンテーション UI がそのまま使えます。テキスト指定は `POST /api/v1/segment_by_text` で利用でき、`score_threshold` 以上の全インスタンスを 1 枚の消去用マスクに合成して返します:

```bash
curl http://127.0.0.1:8080/api/v1/segment_by_text \
  -H "Content-Type: application/json" \
  --data-binary "{\"name\":\"InteractiveSeg\",\"image\":\"$(base64 -w0 input.png)\",\"prompt\":\"person\",\"score_threshold\":0.5}" \
  --output mask.png
```

テキスト用検出器は初回リクエスト時に遅延ロードされます。フロントエンドのテキスト入力 UI は今後追加予定です。

## バッチ処理

```bash
iopaint run --model lama --device cuda \
  --image /path/to/images \
  --mask /path/to/masks \
  --output /path/to/results
```

`--mask` にファイルを 1 つ指定するとそのマスクを全画像に適用し、ディレクトリを指定すると入力画像と同名のマスクを対応させます。

## 開発

```bash
git clone https://github.com/daraskme/IOpaint.git
cd IOpaint

uv venv --python 3.12
uv pip install torch torchvision --torch-backend=auto
uv pip install -e ".[dev,plugins,web-config]"

python -m pytest
python -m ruff check .

cd web_app
npm ci
npm run build
```

パッケージに組み込む場合は `web_app/dist` を `iopaint/web_app` にコピーしてから `python -m build` を実行します。`python scripts/check_package_contents.py dist` で wheel / sdist の同梱資産を検証できます。

## 本家との違い

- PEP 621 ベースのパッケージング（wheel / sdist のビルド・内容検証を CI 化）
- PyTorch 2.4+ / 最新 CUDA ビルド対応（Blackwell 世代 GPU で動作検証済み）
- diffusers 0.39 / transformers 5.14 / huggingface_hub 1.x / Pillow 12 / NumPy 2 / FastAPI / Gradio 6 対応
- フロントエンドを React 19 / Vite 8 / TypeScript 5.9 / zustand 5 / Radix UI 最新に刷新
- **SAM3**（transformers ネイティブ）によるクリック＋テキスト指定セグメンテーション
- 最新 rembg による BiRefNet 系背景除去
- リリース / Docker / Windows セットアップの各ワークフローを刷新

## ライセンスとクレジット

Apache-2.0 ライセンスです。[LICENSE](LICENSE) を参照してください。本フォークの土台はすべて [Sanster/IOPaint](https://github.com/Sanster/IOPaint) とそのコントリビュータの成果です。ぜひ本家プロジェクトも応援してください。
