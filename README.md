<p align="center">
<img src="https://github.com/Steffenhir/GraXpert/blob/main/img/GraXpert_LOGO_Hauptvariante.png" width="500"/>
</p>

# GraXpert macOS Metal/Core ML patch

This fork is a local macOS patch for running GraXpert AI models through Apple
Core ML on modern Apple Silicon Macs. The goal is simple: keep the excellent
GraXpert app, but let macOS use the GPU/Apple Neural Engine for model inference
instead of falling back to slow CPU-only ONNX Runtime paths.

The original project lives here:

- Homepage: https://www.graxpert.com
- Original repository: https://github.com/Steffenhir/GraXpert
- Official releases: https://github.com/Steffenhir/GraXpert/releases/latest

Please use the official GraXpert releases if you want the upstream app. Use this
fork if you want a local macOS build that runs converted `*-metal` model folders
through Core ML.

## How It Works

The converter scans GraXpert model directories, finds `model.onnx`, and creates
sibling Core ML model folders that appear as separate local model versions:

```text
denoise-ai-models/
  3.0.2/
    model.onnx
  3.0.2-metal/
    model.mlpackage/
```

Runtime behavior:

- normal versions such as `3.0.2` keep using `model.onnx`;
- metal versions such as `3.0.2-metal` use `model.mlpackage`;
- deleting the `*-metal` folder or selecting the non-metal model returns you to
  upstream behavior.

The converter works with the same local model layout GraXpert already uses:

- Background Extraction models
- Denoise models
- Object Deconvolution models
- Stars Deconvolution models, when downloaded locally

## Why This Exists

GraXpert already has an “AI Hardware Acceleration” switch, but on macOS the
standard ONNX Runtime path can still be much slower than native Core ML. A user
reported a large speedup in the upstream issue tracker after converting the
denoise ONNX model to Core ML and calling `MLModel.predict()` directly:

https://github.com/Steffenhir/GraXpert/issues/252

This fork turns that idea into a repeatable local workflow.

## Requirements

Recommended target:

- Apple Silicon Mac
- macOS with Homebrew
- Python 3.12 for the converter
- GraXpert AI models already downloaded locally

Python 3.12 matters for conversion. Newer Python versions may install
`coremltools` without native modules needed to save `mlprogram` packages.

## Quick Start

Clone this fork:

```bash
git clone git@github.com:lonely-lockley/GraXpert.git
cd GraXpert
git switch macos-metal-coreml
```

Install Python 3.12 and create the converter environment:

```bash
brew install python@3.12
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --no-user -r requirements-coreml-converter.txt
```

Make sure GraXpert has downloaded the original ONNX models. The easiest route is
to run the official GraXpert once, select/install the AI models you want, then
close it. The models normally live under:

```text
$HOME/Library/Application Support/GraXpert/bge-ai-models
$HOME/Library/Application Support/GraXpert/denoise-ai-models
$HOME/Library/Application Support/GraXpert/deconvolution-object-ai-models
$HOME/Library/Application Support/GraXpert/deconvolution-stars-ai-models
```

Convert every local model found under GraXpert’s application-support directory:

```bash
.venv/bin/python tools/convert_models_to_coreml.py \
  "$HOME/Library/Application Support/GraXpert"
```

Preview what will be generated without writing anything:

```bash
.venv/bin/python tools/convert_models_to_coreml.py \
  "$HOME/Library/Application Support/GraXpert" \
  --dry-run
```

Overwrite existing converted packages:

```bash
.venv/bin/python tools/convert_models_to_coreml.py \
  "$HOME/Library/Application Support/GraXpert" \
  --overwrite
```

## Converting Individual Model Roots

You can also convert one model family at a time:

```bash
.venv/bin/python tools/convert_models_to_coreml.py \
  "$HOME/Library/Application Support/GraXpert/denoise-ai-models"

.venv/bin/python tools/convert_models_to_coreml.py \
  "$HOME/Library/Application Support/GraXpert/bge-ai-models"

.venv/bin/python tools/convert_models_to_coreml.py \
  "$HOME/Library/Application Support/GraXpert/deconvolution-object-ai-models"

.venv/bin/python tools/convert_models_to_coreml.py \
  "$HOME/Library/Application Support/GraXpert/deconvolution-stars-ai-models"
```

By default, the converter creates sibling `*-metal` folders. If you want to put
`model.mlpackage` next to `model.onnx` instead, use:

```bash
.venv/bin/python tools/convert_models_to_coreml.py "/path/to/model-root" --in-place
```

## Running GraXpert From Source

Install the app dependencies into the same venv:

```bash
brew install python-tk@3.12
.venv/bin/python -m pip install --no-user -r requirements.txt
.venv/bin/python -m pip install --no-user onnxruntime
```

The public repository does not include the private upstream S3 credentials file.
For local/offline use with already downloaded models, create a stub:

```bash
cat > graxpert/s3_secrets.py <<'PY'
endpoint = ""
ro_access_key = ""
ro_secret_key = ""
bucket_name = ""
bge_bucket_name = ""
denoise_bucket_name = ""
deconvolution_object_bucket_name = ""
deconvolution_stars_bucket_name = ""
PY
```

Run the GUI:

```bash
.venv/bin/python -m graxpert.main
```

If remote model listing is unavailable, the app can still use local model
folders found in `~/Library/Application Support/GraXpert/...`.

## Building a macOS App Bundle

Install build tools:

```bash
.venv/bin/python -m pip install --no-user pyinstaller setuptools wheel
```

Build the Apple Silicon app bundle:

```bash
.venv/bin/pyinstaller ./GraXpert-macos-arm64.spec
```

The resulting app is created under:

```text
dist/GraXpert.app
```

Create a local DMG if desired:

```bash
brew install create-dmg

create-dmg \
  --volname "GraXpert-macos-arm64-metal" \
  --window-pos 50 50 \
  --window-size 1920 1080 \
  --icon-size 100 \
  --icon "GraXpert.app" 200 190 \
  --app-drop-link 400 190 \
  "dist/GraXpert-macos-arm64-metal.dmg" \
  "dist/GraXpert.app/"
```

This local build is not notarized. macOS Gatekeeper may warn when opening it.
For private testing, right-click the app and choose Open, or remove quarantine
from your own local build:

```bash
xattr -dr com.apple.quarantine dist/GraXpert.app
```

## Expected Local Model Layout

After conversion, a useful local setup looks like:

```text
~/Library/Application Support/GraXpert/
  bge-ai-models/
    1.0.1/
      model.onnx
    1.0.1-metal/
      model.mlpackage/
  deconvolution-object-ai-models/
    1.0.1/
      model.onnx
    1.0.1-metal/
      model.mlpackage/
  denoise-ai-models/
    3.0.2/
      model.onnx
    3.0.2-metal/
      model.mlpackage/
```

## Troubleshooting

If conversion fails with `BlobWriter not loaded`, recreate the venv with Python
3.12:

```bash
brew install python@3.12
rm -rf .venv
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install --no-user -r requirements-coreml-converter.txt
```

If the app cannot import `graxpert.s3_secrets`, create the local stub shown
above.

If a converted model behaves incorrectly, delete its `*-metal` folder and select
the original ONNX version again.

## License and Upstream Credits

GraXpert is developed by the GraXpert team and licensed upstream under GPL-3.0.
The AI models have their own licenses in the `licenses/` directory. This fork is
a local macOS acceleration patch on top of the original project.
