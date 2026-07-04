# macOS Core ML model conversion

This helper converts downloaded GraXpert `model.onnx` folders into sibling
Core ML model folders that can be selected separately in the UI, for example:

```text
denoise-ai-models/
  3.0.2/
    model.onnx
  3.0.2-metal/
    model.mlpackage/
```

Create the converter environment:

```bash
brew install python@3.12
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-coreml-converter.txt
```

Use Python 3.12 for this environment. Newer Python versions may install
`coremltools` without the native modules required to save `mlprogram`
packages.

Convert every model under one GraXpert model root:

```bash
.venv/bin/python tools/convert_models_to_coreml.py \
  "$HOME/Library/Application Support/GraXpert/denoise-ai-models"
```

The same command can be used for:

```text
$HOME/Library/Application Support/GraXpert/bge-ai-models
$HOME/Library/Application Support/GraXpert/deconvolution-object-ai-models
$HOME/Library/Application Support/GraXpert/deconvolution-stars-ai-models
```

Useful options:

```bash
# Show planned outputs without converting
.venv/bin/python tools/convert_models_to_coreml.py "/path/to/models" --dry-run

# Replace an existing model.mlpackage
.venv/bin/python tools/convert_models_to_coreml.py "/path/to/models" --overwrite

# Put model.mlpackage next to model.onnx instead of creating 3.0.2-metal
.venv/bin/python tools/convert_models_to_coreml.py "/path/to/models" --in-place
```

The converter uses `onnx2torch` and `coremltools`, so it must be run on macOS
for practical Core ML testing.
