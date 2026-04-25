# Seam Harmonizer v3

Internal strip-based seam harmonization project for ComfyUI workflows.

The implementation follows [new_architecture.md](/Users/andreyev-a/pet_projects/unet_seam/new_architecture.md):

- canonical `1024x256` strips with `RGB + mask/distance/boundary/decay/luma/gradient`;
- `SeamHarmonizerV3`: NAFNet-lite encoder + multi-scale decoder + local correction fields;
- analytic local correction of the inner band only;
- on-the-fly synthetic corruption from clean source images;
- optional real finetune from production paired strips with Sobel structural filtering;
- `.safetensors + .json` export and ComfyUI node loading.

## Data

Use `manifests/input_raw_manifest.jsonl` with clean source images. The old cached strip dataset is not part of this pipeline.

Use `input_raw_manifest.jsonl` as the source manifest as long as every `source_path` exists in the current environment. For Colab, build a fresh bundle with:

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

## Resume And Finetune

Use `--resume` only when you want to continue the exact same run with optimizer / scheduler state restored.

```bash
python3 -m scripts.train_harmonizer \
  --config configs/train_harmonizer_v1.yaml \
  --resume outputs/checkpoints/best_harmonizer_quality.pt \
  --additional-epochs 4
```

Use `--load-weights` when you want to start a new stage from a previous checkpoint but reset optimizer / scheduler, for example real-pair finetune after synthetic pretrain.

```bash
python3 -m scripts.train_harmonizer \
  --config configs/finetune_harmonizer_v1.yaml \
  --load-weights outputs/checkpoints/best_harmonizer_quality.pt
```

The current export artifact name is `seam_harmonizer_v3.safetensors` and the ComfyUI node is `SeamHarmonizerV3`.

## Colab

Use [seam_harmonizer_train_eval_colab.ipynb](/Users/andreyev-a/pet_projects/unet_seam/colab/seam_harmonizer_train_eval_colab.ipynb). It expects the harmonizer training bundle above and does not require cached strips.
