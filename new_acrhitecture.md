**Project type:** production-first strip-based seam correction model  
**Target framework:** PyTorch 2.x + safetensors + ComfyUI custom node  
**Primary goal:** remove visible color / tone / low-frequency seam drift around generated center borders, without changing scene stru
---

## 1. Executive summary

This document defines the **final v1 target architecture** and all required implementation changes relative to the previous approach.

### Final v1 direction
We replace the earlier **direct RGB residual corrector** with a more controlled and task-shaped model:

- **Backbone:** compact `NAFNet-lite` style encoder
- **Head 1:** **monotonic RGB curve head**
- **Head 2:** **low-frequency shading head**
- **Inference output:** parametric correction of the **inner seam band only**
- **Deployment:** custom ComfyUI node that extracts seam strips, runs the model, reconstructs corrected bands analytically, and merges them back into the full image

### Why this is the final direction
The seam defect is mostly:
- color drift
- tone drift
- low-frequency luminance / shading mismatch
- slight saturation / temperature / contrast mismatch

It is **not primarily** a need to redraw new content.  
Therefore the model should **not** behave like a generic image generator.  
It should behave like a **controlled seam harmonizer**.

### Final design principles
1. Correct only where needed: **narrow bands inside the mask**
2. Preserve structure and detail
3. Use **parametric correction**, not free-form repainting
4. Train on **strip-based samples**
5. Use **synthetic pretrain + real finetune**
6. Keep inference fast enough for interactive ComfyUI usage

---

## 2. Problem statement

We have a full image with a generated center region.  
The outer context is original.  
The generated center may match semantics well but still produce visible seams on the borders because of:

- different brightness / exposure
- different gamma / contrast
- different saturation / hue / tint
- different tonal roll-off in shadows / highlights
- low-frequency shading drift
- VAE / decode / crop-stitch / pipeline-induced color mismatch

### What we want
Given a narrow strip around each seam:
- use the original outer side as anchor
- correct only the inner side
- remove visible seam mismatch
- preserve texture and content

### What we do **not** want
- redraw the center
- hallucinate objects
- edit structure
- change outer/original pixels
- run a slow second generative pass

---

## 3. Final scope of v1

### v1 includes
- strip-based dataset generation
- on-the-fly synthetic corruption
- real seam extraction for finetune
- NAFNet-lite encoder
- RGB monotonic curve head
- 1-channel low-frequency shading head
- complete training / eval pipeline
- export to `.safetensors`
- ComfyUI custom node
- inference-time corner handling
- strength control and merge tapering

### v1 does not include
- diffusion-based seam repair
- full-image harmonization
- VAE retraining
- semantic content correction
- dedicated corner model
- transformer-only / Mamba-only architecture

---

## 4. Canonical seam strip formulation

To simplify the problem, every training and inference example must be converted into a canonical strip layout.

### Canonical strip geometry
Each seam sample is a **vertical strip** of size:

- **width:** `256`
- **height:** `1024`

Split by width into two halves:

- `0..127`: **outer/original half**
- `128..255`: **inner/generated half**

Meaning:
- left side = original anchor
- right side = region to correct

### Horizontal seams
For top/bottom seams:
- extract horizontal strip
- rotate by 90 degrees
- process as canonical vertical strip
- rotate back after inference

### Why canonical layout is mandatory
This makes:
- architecture simpler
- dataset uniform
- debugging easier
- model behavior consistent
- inference code much easier

---

## 5. Input / output contract

## 5.1 Model input

Each sample input tensor:

- shape: `B x 5 x 1024 x 256`

Channels:
1. `RGB strip` — 3 channels
2. `inner_mask` — 1 channel
3. `distance_to_seam` — 1 channel

### Channel semantics

#### RGB strip
The full canonical strip:
- outer/original half unchanged
- inner/corrupted or inner/generated half to be corrected

#### inner_mask
Binary mask:
- `0` on outer half
- `1` on inner half

#### distance_to_seam
A scalar field over the strip, designed to explicitly tell the model where the seam is.

Recommended definition:
- on the seam line: `0`
- at the far right edge of the inner half: `1`
- outer half may be `0` or smoothly extended negative/constant; simplest stable version:
  - `0` on outer half
  - linear ramp `0..1` across inner half

Recommended implementation:
- outer half = `0`
- inner half pixel coordinate `u = 0..127`
- `distance_to_seam[u] = u / 127`

---

## 5.2 Model output

The model outputs **parameters**, not a full corrected RGB image.

### Output A — RGB curves
Shape:
- `B x 3 x K`

where:
- `K = 16` knots per channel

This represents:
- one monotonic curve for `R`
- one monotonic curve for `G`
- one monotonic curve for `B`

### Output B — shading map
Shape:
- low-resolution map: `B x 1 x 256 x 32`

This is later upsampled to the inner half size:
- `B x 1 x 1024 x 128`

### Final corrected inner band
The final corrected inner band is computed analytically inside the forward logic or inside a post-forward reconstruction function:

1. apply RGB curves
2. apply shading gain
3. clamp to `[0, 1]`

---

## 6. Dataset design

## 6.1 Final dataset strategy

Use two stages:

### Stage A — synthetic pretrain
Train on synthetic seam strips generated from clean images.

### Stage B — real finetune
Finetune on strips extracted from the **real production pipeline output**.

This is the final recommended strategy.

---

## 6.2 Clean base dataset

Store:
- clean full images
- clean extracted strips
- seam orientation metadata
- optional scene category tags

### Minimum useful size
- `200–500` source images: proof-of-concept possible
- `1000–3000`: good
- `5000+`: strong

### Preferred content diversity
The source images must overrepresent cases where seams are visually obvious:
- skies
- smooth gradients
- walls / interiors
- skin / faces
- night scenes
- sunsets / dawn
- haze / fog / smoke
- low-contrast backgrounds
- large uniform surfaces
- food / plates / bright surfaces

---

## 6.3 Synthetic training sample generation

Each training sample is built from a clean canonical strip.

### Pipeline
1. Start from clean strip
2. Keep outer half unchanged
3. Apply randomized corruption **only to inner half**
4. Optionally apply slight seam-local blending artifacts
5. Return:
   - corrupted strip
   - clean strip as target
   - mask
   - distance map

### Important rule
Corruption must be generated **on-the-fly**, not pre-baked into fixed image files.

This increases effective dataset diversity dramatically.

---

## 6.4 Synthetic corruption families

Each sample should apply a random combination of `2–5` corruption modules.

### Family A — global photometric corruption
Apply to inner half only:
- exposure shift
- brightness shift
- contrast scaling
- gamma shift
- saturation scaling
- hue shift
- temperature shift
- tint shift
- per-channel RGB gain
- black point shift
- white point shift

### Family B — tonal curve corruption
Apply:
- shadow lift
- shadow crush
- highlight compression
- highlight boost
- midtone shift
- S-curve
- reverse S-curve

### Family C — low-frequency spatial corruption
Apply:
- horizontal low-frequency luminance gradient
- vertical low-frequency luminance gradient
- 2D smooth illumination field
- smooth color temperature field
- slow saturation drift field

### Family D — pipeline-like corruption
Apply lightly:
- mild blur
- slight microcontrast loss
- slight noise
- slight compression-like color shift
- optional VAE round-trip on inner half for a percentage of samples

### Recommended corruption probabilities
Per sample:
- choose `2–5` modules
- at least one from `A` or `B`
- `C` applied in about `30–40%` of samples
- `D` applied in about `20–30%`

---

## 6.5 Real finetune dataset

Real finetune examples must come from the actual production pipeline.

### Real finetune sample generation
For each clean full image:
1. run the current generation / decode / stitch pipeline
2. align result with original
3. extract seam strips from left / right / top / bottom borders
4. keep only samples where the inner region still corresponds structurally to the original

### Structural filtering
Do **not** include strips where the first pixels inside the seam differ semantically / structurally too much from the original.

Otherwise the harmonizer will learn to fix semantic mismatch, which is not its job.

#### Recommended filter
On the first `32 px` inside the seam:
- compute Sobel gradients for generated and original
- compute gradient cosine similarity

If:
- `grad_cosine < 0.6`

then discard the sample.

### Recommended real finetune dataset size
- minimum: `2k`
- preferred: `5k–10k`

---

## 7. Final architecture

## 7.1 Backbone choice

Use a compact `NAFNet-lite` style encoder.

### Why
- efficient
- strong for low-level vision
- stable
- easy to implement
- suitable for narrow strip restoration / harmonization

This is the final recommended backbone family for v1.

---

## 7.2 Encoder specification

### Input
`B x 5 x 1024 x 256`

### Stem
- `Conv3x3`, `5 -> 32`

### Encoder stages
- Stage 1:
  - channels = `32`
  - blocks = `2`
- Downsample
- Stage 2:
  - channels = `64`
  - blocks = `2`
- Downsample
- Stage 3:
  - channels = `128`
  - blocks = `4`
- Downsample
- Stage 4:
  - channels = `192`
  - blocks = `6`

### Block type
Use NAFNet-style blocks or a simplified equivalent:
- depthwise conv
- channel mixing
- simple gating
- residual connection
- normalization consistent with the chosen NAFNet implementation

### Notes
- no transformer block is required
- no attention block is required
- no full decoder is required

This keeps compute low.

---

## 7.3 Curve head

The curve head predicts **three monotonic piecewise-linear curves**, one for each RGB channel.

### Global pooling
Take Stage 4 features and apply global average pooling.

### MLP head
Recommended:
- Linear: `192 -> 256`
- SiLU
- Linear: `256 -> 3*(K-1)`

with:
- `K = 16`

### Monotonic parameterization
The network predicts raw values `z_{c,i}`.

Convert them to positive increments:
- `delta_{c,i} = softplus(z_{c,i}) + eps`

Use:
- `eps = 1e-4`

Then accumulate and normalize:
- `v_{c,0} = 0`
- `v_{c,k} = sum(delta_{c,1..k}) / sum(delta_{c,1..K-1})`
- final knot equals `1`

This guarantees:
- monotonicity
- bounded range
- stable optimization

### Curve application
For each channel independently:
- map input values in `[0,1]`
- use piecewise-linear interpolation over the `K` knot values

This curve is applied **only to the inner half**.

---

## 7.4 Shading head

The shading head predicts **one low-frequency luminance-like gain field** for the inner half.

### Why 1-channel only
Because:
- global color/tone is already handled by RGB curves
- low-frequency local mismatch is mostly luminance / shading
- 1-channel shading is more stable and interpretable

### Head structure
Use Stage 3 features as input.

Recommended head:
- `Conv3x3: 128 -> 64`
- SiLU
- `Conv3x3: 64 -> 32`
- SiLU
- `Conv3x3: 32 -> 1`

Output low-res shading:
- `B x 1 x 256 x 32`

Upsample using bilinear interpolation to:
- `B x 1 x 1024 x 128`

### Gain conversion
Convert the shading map to a multiplicative gain:
- `gain = exp(alpha * tanh(S))`

Recommended:
- `alpha = 0.20`

This constrains the shading amplitude to a safe range.

### Why multiplicative gain
Because it behaves like a smooth exposure / luminance correction rather than a free-form additive paint.

---

## 7.5 No residual head in v1 final
For the final v1 target architecture, **do not include** the optional seam residual head.

### Reason
The goal is:
- fastest route to a strong and stable target architecture
- maximum interpretability
- minimum architectural complexity

If later evaluation shows a small remaining seam mismatch after curves + shading, that becomes **v2**, not v1.

---

## 8. Mathematical formulation

## 8.1 Notation

Let:
- `X_outer` = outer/original half
- `X_inner` = inner/generated or corrupted half
- `f_r, f_g, f_b` = predicted RGB curves
- `g(x,y)` = predicted shading gain map over inner half

Target:
- `Y_gt_inner`

### Step 1 — apply channel curves
For each pixel and channel:
- `Y_curve_r = f_r(X_inner_r)`
- `Y_curve_g = f_g(X_inner_g)`
- `Y_curve_b = f_b(X_inner_b)`

Concatenate:
- `Y_curve = [Y_curve_r, Y_curve_g, Y_curve_b]`

### Step 2 — apply shading gain
- `Y = clip(Y_curve * g, 0, 1)`

This is the final corrected inner half.

### Step 3 — inference-time merge into full image
The custom node blends `Y` back into the full image using a seam-distance taper.

---

## 9. Loss design

## 9.1 Reconstruction loss

Use Charbonnier loss on the corrected inner half:
- `L_rec = charbonnier(Y, Y_gt_inner)`

Recommended epsilon:
- `1e-3`

### Why Charbonnier instead of L1
- similar robustness
- slightly smoother gradients
- very common and stable in low-level vision

---

## 9.2 Seam-weighted loss

Most visible seam errors occur very close to the seam line.

Define inner-half width coordinate:
- `u = 0..127`
- `u = 0` at the seam

Define seam weight:
- `w_seam(u) = exp(-u / tau)` for `u < 32`
- `w_seam(u) = 0` for `u >= 32`

Recommended:
- `tau = 12`

Then:
- `L_seam = weighted_charbonnier(Y, Y_gt_inner, w_seam)`

This must be a high-priority loss term.

---

## 9.3 Low-frequency loss

Blur both predicted and target inner halves and compare:
- `L_low = L1(gaussian(Y), gaussian(Y_gt_inner))`

Recommended Gaussian:
- sigma `5`
- kernel size `21`

### Purpose
This directly teaches the model to fix:
- shading
- smooth tone drift
- low-frequency mismatch

---

## 9.4 Gradient loss

Compute Sobel gradients and compare:
- `L_grad = L1(grad(Y), grad(Y_gt_inner))`

### Purpose
This reduces:
- seam contrast jumps
- unnatural microcontrast discontinuity

---

## 9.5 Curve smoothness loss

Even though monotonicity is guaranteed, curves should still be smooth.

For each channel curve knot sequence `v`:
- penalize the second finite difference

Define:
- `L_curve_smooth = sum |(v[i+1]-v[i]) - (v[i]-v[i-1])|`

This discourages overly jagged curve shapes.

---

## 9.6 Curve identity regularization

If no correction is needed, the curves should stay close to identity.

For channel knots:
- identity target = evenly spaced line from `0` to `1`

Define:
- `L_curve_id = mean |v - identity|`

This should be weakly weighted.

---

## 9.7 Shading total variation loss

The shading map must remain smooth and low-frequency.

Use:
- `L_tv = TV(S)`

where `S` is the upsampled shading map or low-res shading map.

Recommended:
- isotropic or anisotropic TV; either is acceptable
- simplest implementation is anisotropic TV

---

## 9.8 Shading mean regularization

The shading map should not absorb the entire global exposure correction.

Define:
- `L_mean = |mean(S)|`

This nudges the shading field toward zero-mean behavior and lets global color/tone correction stay in the curve head.

---

## 9.9 Final loss

The final training objective:

- `L_total = 1.0*L_rec + 2.0*L_seam + 0.75*L_low + 0.25*L_grad + 0.02*L_curve_smooth + 0.01*L_curve_id + 0.02*L_tv + 0.01*L_mean`

This is the default final v1 configuration.

---

## 10. Training setup

## 10.1 Framework
Use:
- Python 3.10+
- PyTorch 2.x
- torchvision
- safetensors
- numpy
- pillow
- albumentations optional, not required
- simple pure-PyTorch training loop preferred

### Why not overcomplicate the stack
The project must be implementable quickly and debuggable by any strong Python developer.

---

## 10.2 Precision
- NVIDIA A100 / Blackwell: `bf16`
- fallback: `fp16`

---

## 10.3 Optimizer
Use:
- `AdamW`

Parameters:
- learning rate: `2e-4`
- betas: `(0.9, 0.99)`
- weight_decay: `1e-4`

---

## 10.4 Scheduler
Use:
- warmup: `1000` steps
- cosine decay
- min learning rate: `1e-6`

---

## 10.5 Gradient clipping
Use:
- `clip_grad_norm = 1.0`

---

## 10.6 EMA
Use EMA of model weights:
- `ema_decay = 0.999`

EMA checkpoint is usually the default validation / export candidate.

---

## 10.7 Batch size
Initial targets:

### A100 40 GB
- start with `batch = 32`

### 96 GB GPU
- start with `batch = 64`

Adjust only if memory demands require it.

---

## 10.8 Epochs

### Synthetic pretrain
- `18–20` epochs

### Real finetune
- `6–10` epochs

### During real finetune
Mix batches:
- `80% real`
- `20% synthetic`

This prevents forgetting the synthetic distribution entirely.

---

## 10.9 Estimated train time

Realistic target ranges:

### A100 40 GB
- synthetic pretrain: approximately `3–8 hours`
- real finetune: approximately `1–3 hours`

### 96 GB Blackwell-class GPU
- synthetic pretrain: approximately `2–6 hours`
- real finetune: approximately `1–2 hours`

These are target engineering estimates, assuming:
- clean pipeline
- no major dataloader bottlenecks
- no excessive CPU augmentation overhead

---

## 11. Validation and evaluation

## 11.1 Required validation splits
Maintain separate validation sets:

1. synthetic validation
2. real validation

### Real validation categories
Split or tag by content type:
- sky / gradients
- faces / skin
- walls / interiors
- dark / night
- saturated scenes
- food / glossy surfaces

---

## 11.2 Core metrics

### Boundary MAE
Compute MAE in the inner half only, separately for:
- first `8 px`
- first `16 px`
- first `32 px`

### Boundary Delta E
Compute `DeltaE2000` in Lab space for:
- first `16 px`
- first `32 px`

### Low-frequency MAE
Blur prediction and target, then compare in the inner half.

### Gradient discrepancy
Compare Sobel gradients in the inner half.

### Seam visibility proxy
Mean absolute color error in the first `8 px` from the seam.

---

## 11.3 Human A/B review
For real validation:
- blind compare current baseline vs v1 target model
- rate seam visibility only
- keep content fixed
- require at least 70% preference for the new model before rollout

---

## 11.4 Success criteria
Declare v1 successful if all of the following hold:

1. `Boundary MAE@16px` improves by at least `15%`
2. `Boundary DeltaE2000@16px` improves by at least `15%`
3. `Low-frequency MAE` improves by at least `20%`
4. visual A/B wins in at least `70%` of real cases
5. inference latency fits the target budget

---

## 12. Export

## 12.1 Model format
Export the final weights as:
- `.safetensors`

### What to save
Save:
- encoder state dict
- curve head state dict
- shading head state dict
- config dictionary

### Also save
A JSON or YAML config with:
- knot count
- input channels
- image geometry
- alpha value
- preprocessing settings
- normalization settings

---

## 12.2 Recommended file layout
- `model.safetensors`
- `config.json`

Config example fields:
- model_name
- input_channels = 5
- strip_width = 256
- strip_height = 1024
- inner_width = 128
- num_knots = 16
- alpha = 0.20
- dtype
- normalization mean/std if used

---

## 13. ComfyUI custom node specification

## 13.1 Node name
Recommended:
- `SeamHarmonizerV1`

## 13.2 Node inputs
- `image`
- `mask`
- `model_name`
- `band_width` default `128`
- `strength` default `1.0`
- `process_left` default `true`
- `process_right` default `true`
- `process_top` default `true`
- `process_bottom` default `true`

Optional:
- `device`
- `feather`
- `debug_previews`

## 13.3 Node output
- `corrected_image`

Optional debug outputs:
- corrected left strip
- corrected right strip
- corrected top strip
- corrected bottom strip
- predicted curve parameters
- shading previews

---

## 13.4 Node runtime steps

### Step 1 — extract seam strips
From the full image and center mask:
- extract left seam strip
- extract right seam strip
- extract top seam strip
- extract bottom seam strip

Each strip should contain:
- 128 px outer side
- 128 px inner side

### Step 2 — canonical orientation
Convert all strips to the canonical layout:
- outer half left
- inner half right
- vertical orientation

### Step 3 — build input tensors
For each strip:
- RGB strip
- mask channel
- distance-to-seam channel

### Step 4 — batch all strips together
This is preferred for speed.

### Step 5 — run model
Model returns:
- curve knots
- shading map

### Step 6 — reconstruct corrected inner half
For each strip:
- decode curves
- apply curves to inner RGB
- upsample shading
- apply shading gain
- clamp to `[0,1]`

### Step 7 — strength-controlled merge
Do not hard replace the full inner strip.  
Blend corrected inner strip back using a taper.

Recommended taper across inner width:
- cosine taper from seam inward

Define for inner coordinate `u = 0..127`:
- `w(u) = 0.5 * (1 + cos(pi * u / 127))`

Then:
- `w_eff(u) = strength * w(u)`

Final merge:
- `inner_final = (1 - w_eff) * inner_old + w_eff * inner_fixed`

### Step 8 — corner fusion
Corners receive contributions from two sides.

Fuse by weighted average:
- each side has a weight based on distance to its corresponding seam edge
- final corner pixel = normalized weighted sum

### Step 9 — insert back into full image
Write corrected bands back into the original full image tensor.

---

## 13.5 Performance targets

### GPU inference target
For 4 strips batched together:

#### A100 40 GB
Target:
- `10–40 ms`

#### 96 GB Blackwell-class GPU
Target:
- `8–30 ms`

### Acceptable non-server target
On Apple Silicon or weaker hardware, proportionally slower is acceptable, but the node should still remain practical.

---

## 14. Edge cases and failure handling

## 14.1 Structural mismatch near the seam
If structure already differs between original and generated near the seam, the harmonizer should not aggressively try to fix it.

### Handling
At inference time:
- compute quick gradient similarity in first `8–16 px` inside the seam
- if too low:
  - reduce strength
  - or skip that seam correction entirely

Recommended threshold:
- `grad_cosine < 0.5` → halve strength
- `grad_cosine < 0.35` → skip seam

---

## 14.2 Uniform regions
Flat skies and walls are sensitive to:
- banding
- overcorrection
- unnatural shading

### Handling
- keep shading 1-channel
- limit alpha to `0.20`
- use TV regularization
- inspect low-frequency metrics carefully

---

## 14.3 Strongly saturated scenes
Curve head may become too aggressive.

### Handling
- keep curve smoothness loss active
- monitor max slope of each curve
- oversample saturated examples in training

---

## 14.4 Noisy or compressed inputs
Pipeline-like corruption must include these cases, but lightly.

### Handling
- do not let the harmonizer become a denoiser
- keep blur/noise/compression perturbations secondary to photometric perturbations

---

## 14.5 Outer half corruption by bug
The model must never learn to modify outer/original half.

### Handling
- mask-aware reconstruction only on inner half
- optional weak identity penalty on outer half if any reconstruction logic touches it
- custom node must never overwrite outer pixels

---

## 15. Required code modules

## 15.1 Dataset package
Modules:
- strip extraction
- canonical rotation
- synthetic corruption
- real strip harvesting
- real strip structural filter
- tensor builder

## 15.2 Model package
Modules:
- NAFNet-lite encoder
- monotonic curve head
- shading head
- curve reconstruction utilities
- final corrected inner-half reconstruction function

## 15.3 Training package
Modules:
- train loop
- EMA
- schedulers
- mixed precision
- checkpointing
- logging
- validation metrics

## 15.4 Evaluation package
Modules:
- boundary metrics
- DeltaE2000
- low-frequency metrics
- gradient metrics
- benchmark script
- A/B export tooling

## 15.5 ComfyUI custom node package
Modules:
- model loader
- strip extractor
- seam input builder
- forward pass wrapper
- curve/shading reconstruction
- band merger
- corner fusion
- debug preview helpers

---

## 16. Recommended implementation order

### Phase 1 — data
1. implement clean strip extractor
2. implement canonical strip conversion
3. implement on-the-fly synthetic corruption
4. implement real strip extraction
5. implement structural mismatch filter

### Phase 2 — model
6. implement NAFNet-lite encoder
7. implement monotonic curve head
8. implement shading head
9. implement reconstruction logic

### Phase 3 — training
10. implement losses
11. implement train loop
12. implement validation metrics
13. run synthetic pretrain
14. inspect curve outputs and shading maps

### Phase 4 — finetune
15. build real strip dataset
16. run real finetune
17. compare against previous residual baseline

### Phase 5 — inference
18. implement ComfyUI custom node
19. add corner fusion and taper merging
20. benchmark latency
21. run visual QA in real workflows

---

## 17. Hard requirements for the developer / AI agent

The implementer must satisfy all of the following:

1. keep the model parametric: **curves + shading only**
2. do not add a residual RGB head in v1
3. keep strip canonicalization strict
4. train with on-the-fly corruption
5. include real finetune
6. implement boundary-focused metrics
7. export `.safetensors + config.json`
8. integrate via ComfyUI custom node
9. keep inference batched across all seam strips
10. preserve outer/original pixels exactly

---

## 18. Final decisions summary

### Final architecture
- `NAFNet-lite encoder + monotonic RGB curves + 1-channel low-frequency shading`

### Final strip geometry
- `256 x 1024`
- `128 px` outer + `128 px` inner

### Final data strategy
- synthetic pretrain + real finetune

### Final losses
- reconstruction
- seam-weighted
- low-frequency
- gradient
- curve smoothness
- curve identity
- shading TV
- shading mean

### Final deployment
- ComfyUI node
- batched seam strips
- analytical correction
- taper merge
- weighted corner fusion

---

## 19. Definition of done

The work is complete only if all items below are true:

- model trains end-to-end without instability
- curves remain monotonic and reasonable
- shading maps stay smooth and interpretable
- validation metrics beat the previous residual baseline
- real visual A/B confirms seam improvement
- inference integrates into ComfyUI without outer-pixel corruption
- latency fits the target budget
- model exports and reloads from `.safetensors`
- documentation and config are sufficient for reproducibility

---

## 20. Final note

This document intentionally defines the **target v1 architecture**, not a lightweight MVP.  
The priority is to reach a strong, production-oriented, interpretable seam harmonizer as quickly as possible, with minimal wasted iteration on architectures that are too generic for the actual seam problem.
