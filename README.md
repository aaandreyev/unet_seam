# Seam Residual Corrector v1

Research-only / internal-use repository for a strip-based seam residual corrector targeting ComfyUI / FLUX inpaint seam-fix workflows.

## Scope

This project implements the normative contract from [seam_residual_corrector_spec.md](/Users/andreyev-a/pet_projects/unet_seam/seam_residual_corrector_spec.md):

- synthetic-only strip training for v1;
- canonical `256x1024` strips with `RGB + inner_mask + distance_to_seam`;
- small ResUNet with internal low-frequency branch and GAP->FiLM;
- seam-specific losses, metrics, export, viewer, and ComfyUI node.

## Data policy

- `input_raw/` is internal-only and must not be published.
- `data/source_images/` and all derived artifacts are internal-only.
- The repo is intended for research and local experimentation, not public dataset release.

## Layout

- `src/`: model, data, losses, metrics, train, inference utilities
- `scripts/`: preprocessing, split, train, eval, export, smoke tests
- `comfy_node/`: ComfyUI integration
- `tests/`: geometry, merge, residual, dataset tests
- `static/` + `strip_dataset_viewer.py`: strip cache viewer

## Quick start

```bash
python3 scripts/prepare_source.py --input input_raw --output data/source_images
python3 scripts/build_split.py
python3 scripts/cache_val_strips.py --config configs/train_synth_v1.yaml
python3 scripts/train_resunet.py --config configs/train_synth_v1.yaml
python3 scripts/run_eval.py --config configs/eval_v1.yaml
python3 scripts/export_safetensors.py --config configs/export_v1.yaml
python3 scripts/verify_export.py --config configs/export_v1.yaml
```

## Notes

- `dataset_viewer.py` is legacy from an older canvas pipeline and is not part of the v1 baseline.
- The new strip-based viewer is `strip_dataset_viewer.py`.
