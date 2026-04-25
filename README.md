Т# Seam Harmonizer v1

Internal strip-based seam harmonization project for ComfyUI workflows.

The implementation follows [new_acrhitecture.md](/Users/andreyev-a/pet_projects/unet_seam/new_acrhitecture.md):

- canonical `1024x256` strips with `RGB + inner_mask + distance_to_seam`;
- `SeamHarmonizerV1`: NAFNet-lite encoder + monotonic RGB curves + 1-channel low-frequency shading;
- parametric correction of the inner band only;
- on-the-fly synthetic corruption from clean source images;
- optional real finetune from production paired strips with Sobel structural filtering;
- `.safetensors + .json` export and ComfyUI node loading.

## Data

Use `manifests/input_raw_manifest.jsonl` with clean source images. The old cached strip dataset is not part of this pipeline.

Existing `input_raw_manifest.jsonl` is compatible with the new training code as long as every `source_path` exists in the current environment. For Colab, build a fresh bundle with:

```bash
python3 scripts/build_final_training_bundle.py --bundle-output outputs/training_bundle/seam_harmonizer_training_bundle.tar.gz
```

## Quick Start

```bash
python3 scripts/prepare_source.py --input input_raw --output data/source_images --manifest manifests/input_raw_manifest.jsonl
python3 scripts/build_split.py --manifest manifests/input_raw_manifest.jsonl
python3 -m scripts.train_harmonizer --config configs/train_harmonizer_v1.yaml
python3 -m scripts.run_eval_harmonizer --config configs/eval_harmonizer_v1.yaml
python3 -m scripts.export_harmonizer_safetensors --config configs/export_harmonizer_v1.yaml
python3 -m scripts.verify_harmonizer_export --config configs/export_harmonizer_v1.yaml
```

## Colab

Use [seam_harmonizer_train_eval_colab.ipynb](/Users/andreyev-a/pet_projects/unet_seam/colab/seam_harmonizer_train_eval_colab.ipynb). It expects the harmonizer training bundle above and does not require cached strips.
