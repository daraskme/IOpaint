<h1 align="center">IOPaint — modernized fork</h1>
<p align="center">Self-hosted image inpainting, outpainting, object removal, and AI-assisted image editing.</p>

<p align="center">
  <img alt="Python 3.10–3.13" src="https://img.shields.io/badge/Python-3.10--3.13-blue" />
  <img alt="Apache-2.0 license" src="https://img.shields.io/badge/license-Apache--2.0-blue" />
</p>

<p align="center">English | <a href="README_ja.md">日本語</a></p>

This project is based on [Sanster/IOPaint](https://github.com/Sanster/IOPaint). The original project, model integrations, interface, and years of community work made this fork possible. This fork keeps the existing IOPaint workflow and model lineup while modernizing its Python, PyTorch, Hugging Face, frontend, packaging, and deployment stack for 2026.

> **Distribution-name placeholder:** `iopaint-ng` is intentionally temporary. The owner must choose the final PyPI name before the first publish. The installed command remains `iopaint`.

| Input | Result |
|---|---|
| ![Object to remove](assets/unwant_object.jpg) | ![Object removed](assets/unwant_object_clean.jpg) |
| ![Text to remove](assets/unwant_text.jpg) | ![Text removed](assets/unwant_text_clean.jpg) |

## Requirements

- Python `>=3.10,<3.14`
- PyTorch `>=2.4`
- CPU, NVIDIA CUDA, and Apple Silicon are supported where the selected model supports that device.
- Recent NVIDIA GPUs, including Blackwell, require a PyTorch CUDA 12.8-or-newer build.

Models are downloaded on first use. Use `--model-dir` to choose the cache location or `--local-files-only` after the required weights have been cached.

## Installation

### Windows — one-click launcher (easiest)

Download `IOPaint-OneClick.bat` from the [latest release](https://github.com/daraskme/IOpaint/releases/latest) and double-click it. On first run it installs uv, creates a Python environment under `%LOCALAPPDATA%\IOPaint`, detects your NVIDIA GPU (CUDA 12.8 vs CPU PyTorch), and installs the latest release wheel from GitHub. Subsequent runs start IOPaint immediately and open the browser. Everything, including the uv cache and downloaded model weights, lives under `%LOCALAPPDATA%\IOPaint`, so uninstalling is just deleting that folder.

To change where things are stored (for example to keep multi-GB model weights on another drive), edit the `IOPaint-OneClick.cfg` file that the launcher writes next to itself on first run, or set the same names as environment variables:

```ini
# IOPaint-OneClick.cfg: one KEY=VALUE per line, no quotes, lines starting with "#" are ignored
# install root (Python env, uv cache, models by default)
IOPAINT_HOME=D:\IOPaint
# model downloads (passed to iopaint as --model-dir; HF and torch hub caches live inside)
IOPAINT_MODEL_DIR=D:\IOPaint\models
# model to start with, HTTP port, extra arguments appended to `iopaint start`
IOPAINT_MODEL=lama
IOPAINT_PORT=8080
IOPAINT_EXTRA_ARGS=--low-mem
# uv download cache and uv-managed Python (default: under IOPAINT_HOME)
UV_CACHE_DIR=D:\IOPaint\uv-cache
UV_PYTHON_INSTALL_DIR=D:\IOPaint\python
```

Moving `IOPAINT_MODEL_DIR` after models were downloaded? Move the folder and update the setting; nothing else references the old path. Launchers older than v2.0.0rc3 stored models in `%USERPROFILE%\.cache`; the current launcher keeps using that folder when it exists so nothing is downloaded twice.

### uv — recommended

Install [uv](https://docs.astral.sh/uv/), then create an environment and let uv select the PyTorch backend:

```bash
uv venv
uv pip install torch torchvision --torch-backend=auto
uv pip install iopaint-ng
iopaint start --model lama --device cpu --port 8080 --inbrowser
```

For a one-shot trial:

```bash
uvx --from iopaint-ng iopaint start --model lama
```

### pip

Create and activate a virtual environment first. For NVIDIA CUDA 12.8:

```bash
python -m pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install iopaint-ng
iopaint start --model lama --device cuda --port 8080 --inbrowser
```

For CPU-only installations, replace the PyTorch index URL with `https://download.pytorch.org/whl/cpu` and use `--device cpu`.

### Windows bootstrap

From a source checkout, run:

```bat
scripts\install_windows.bat
scripts\start_windows.bat
```

The installer installs uv if necessary, creates `.iopaint-env`, detects `nvidia-smi`, and installs either CUDA 12.8 or CPU PyTorch wheels. A detected NVIDIA system fails clearly if its driver cannot use the CUDA build; it does not silently switch to CPU. Until the package is published, the script contains a commented local-wheel `--find-links dist` alternative.

### Docker

Build both images from a Git checkout:

```bash
bash scripts/build_docker.sh 2.0.0
```

Run the NVIDIA image with the NVIDIA Container Toolkit:

```bash
docker run --rm --gpus all -p 8080:8080 \
  -v iopaint-cache:/root/.cache \
  iopaint-ng:2.0.0-cuda
```

Or run the CPU image:

```bash
docker run --rm -p 8080:8080 \
  -v iopaint-cache:/root/.cache \
  iopaint-ng:2.0.0-cpu
```

Both images embed a production frontend and listen on `0.0.0.0:8080`.

## Models

### Erase models

- `lama`
- `ldm`
- `zits`
- `mat`
- `fcf`
- `manga`
- `cv2`
- `migan`

### Diffusion and guided-editing models

- Stable Diffusion and SDXL normal or inpainting repositories and local checkpoints
- `runwayml/stable-diffusion-inpainting`
- `Uminosachi/realisticVisionV51_v51VAE-inpainting`
- `redstonehero/dreamshaper-inpainting`
- `Sanster/anything-4.0-inpainting`
- `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`
- `RunDiffusion/Juggernaut-XI-v11`
- `SG161222/RealVisXL_V5.0`
- `eienmojiki/Anything-XL`
- BrushNet for SD and SDXL
- PowerPaint V1 and V2
- `Sanster/AnyText`
- `Fantasy-Studio/Paint-by-Example`
- Kandinsky 2.2 inpainting
- InstructPix2Pix
- ControlNet-assisted SD and SDXL workflows

Use `iopaint start --help` for model-specific and memory-management options.

## Plugins and optional dependencies

Install background-removal dependencies, including current rembg/BiRefNet support, with:

```bash
uv pip install "iopaint-ng[plugins]"
```

Available plugins include interactive segmentation, RemoveBG/BiRefNet, anime segmentation, RealESRGAN, GFPGAN, and RestoreFormer. Enable them with the corresponding `iopaint start` options.

The Gradio configuration UI is optional:

```bash
uv pip install "iopaint-ng[web-config]"
iopaint start-web-config
```

## SAM3 segmentation

SAM3 provides click-prompt segmentation and text-prompt concept segmentation. Its weights are gated:

1. Visit [facebook/sam3](https://huggingface.co/facebook/sam3) and accept the repository license.
2. Authenticate locally with the current Hugging Face CLI:

   ```bash
   hf auth login
   ```

3. Start IOPaint with the SAM3 interactive-segmentation plugin:

   ```bash
   iopaint start --model lama --device cuda \
     --enable-interactive-seg \
     --interactive-seg-model sam3 \
     --interactive-seg-device cuda
   ```

Click prompts use the existing interactive-segmentation UI and API. Text prompts are available through `POST /api/v1/segment_by_text`; all instances above `score_threshold` are combined into one erase mask:

```bash
curl http://127.0.0.1:8080/api/v1/segment_by_text \
  -H "Content-Type: application/json" \
  --data-binary "{\"name\":\"InteractiveSeg\",\"image\":\"$(base64 -w0 input.png)\",\"prompt\":\"person\",\"score_threshold\":0.5}" \
  --output mask.png
```

The text detector is loaded lazily and remains resident alongside the click tracker. There is no frontend text-prompt control yet.

## Batch processing

```bash
iopaint run --model lama --device cpu \
  --image /path/to/images \
  --mask /path/to/masks \
  --output /path/to/results
```

If `--mask` points to one file, that mask is applied to every input image. If it points to a directory, mask filenames must match the input filenames.

## Development

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

For an embedded backend build, copy `web_app/dist` to `iopaint/web_app`, then run `python -m build`. `python scripts/check_package_contents.py dist` verifies the wheel and sdist assets.

## Differences from upstream

- Python packaging is PEP 621-based and verified in wheel and sdist builds.
- PyTorch supports modern CUDA builds; the dependency floor is PyTorch 2.4.
- Diffusers 0.39, Transformers 5.14, Hugging Face Hub 1.x, Pillow 12, NumPy 2, FastAPI/Pydantic 2, and Gradio 6 compatibility.
- React 19, Zustand 5, TanStack Query 5, Radix UI updates, react-zoom-pan-pinch 4, TypeScript 5.9, and Vite 8.
- Native Transformers SAM3 click and text-prompt segmentation.
- Current rembg with BiRefNet-family models; background-removal dependencies are optional.
- Gradio is isolated in the `web-config` extra instead of being a mandatory runtime dependency.
- Reproducible release, Docker, and Windows bootstrap workflows replace legacy lama-cleaner/torch-cu118 scripts.

## License and credit

Licensed under Apache-2.0. See [LICENSE](LICENSE). Upstream credit and history belong to [Sanster/IOPaint](https://github.com/Sanster/IOPaint); please support the original project and its contributors.
