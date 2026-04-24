# Seam Residual Corrector для ComfyUI / FLUX / inpaint seam-fix

Нормативная спецификация v1 после консолидации аудита от `2026-04-24`.
Этот документ заменяет предыдущую черновую версию и фиксирует итоговый
контракт для архитектуры, датасета, обучения, экспорта и ComfyUI-ноды.

---

## 0. TL;DR

Нужно построить **узкую пост-коррекционную модель для шва** между:
- внешним оригинальным контекстом;
- внутренней сгенерированной зоной.

Модель:
- работает **после финального decode/composite**;
- видит только **полосу вокруг шва**;
- предсказывает **residual RGB correction**;
- меняет только **inner**-часть;
- обязана сохранять **outer**-часть byte-exact на inference.

Базовый v1:
- canonical strip: `256 x 1024` (`W x H`);
- `outer = 128 px`, `inner = 128 px` по умолчанию;
- один общий ResUNet для left/right/top/bottom;
- вход: `RGB + inner_mask + distance_to_seam` = **5 каналов**;
- архитектура: small ResUNet + internal low-frequency branch + GAP->FiLM;
- выход: bounded residual `tanh(raw) * 0.3`;
- synthetic-only training в v1;
- реальный dataset и finetune вынесены в v2+;
- irregular masks в v1 **не поддерживаются**.

---

## 1. Контракт проекта

### 1.1. Основная цель

Убрать видимый seam по цвету, тону, насыщенности, теням/светам и
low-frequency drift на границе между:
- сохраненным оригинальным изображением;
- сгенерированным центром.

### 1.2. Что считается успехом

Успех для v1:
- шов визуально заметно слабее;
- граница перестает читаться как вставленный прямоугольник;
- outer-контекст не деградирует;
- текстуры и микро-контраст не разрушаются;
- inference достаточно быстрый для ComfyUI;
- результат стабилен на всех 4 сторонах и в углах.

### 1.3. Не-цели

Проект не должен:
- заново генерировать внутреннюю область;
- менять композицию и семантику;
- чинить всё изображение целиком;
- заменять VAE / UNet / LoRA / sampler;
- вводить canvas-логику внутрь сети.

### 1.4. Зафиксированные решения v1

Это инварианты документа:

1. `canvas-spec` не входит в сеть и используется только как reference
   для ComfyUI-ноды и опциональных validation buckets.
2. Вход модели: **5 каналов** (`RGB + inner_mask + distance_to_seam`).
3. Hard-gate вроде "50 px lock" в модели не используется.
4. Residual ограничивается формулой `tanh(raw) * 0.3`.
5. VAE round-trip augmentation в v1 запрещен.
6. v1 обучается только на synthetic strips.
7. seam jitter на train: `+-16 px`.
8. random rotation на train: `0/90/180/270`.
9. На один source image приходится `20-30 strips` за эпоху.
10. Long-range context в модели обязателен:
    internal low-frequency branch + GAP->FiLM.
11. LPIPS используется только как boundary HF loss с весом `0.1`.
12. RGBA raw inputs исключаются на этапе подготовки датасета.
13. `input_raw/` и derived data считаются internal-only.

---

## 2. Геометрия шва и формат полосы

### 2.1. Канонический strip

Единое определение canonical strip:

```text
canonical strip:    256 x 1024 (W x H)
tensor shape CHW:   [C, 1024, 256]
seam boundary:      between x=127 and x=128 (pixel centers at 127.5)
outer/reference:    x in [0, 128)
inner/editable:     x in [128, 256)
```

Инвариант v1:
- в canonical tensor **outer всегда слева**;
- **inner всегда справа**.

### 2.2. Canonicalization для 4 сторон

- `left`  -> identity;
- `right` -> horizontal flip (`torch.flip` / `[:, ::-1]` по оси ширины полосы);
- `top`   -> извлечь полосу `1024 x (outer_w + inner_w)`, затем `rot90` на
  `90 deg` CCW (`k=1` в `torch.rot90`);
- `bottom`-> извлечь полосу `1024 x (outer_w + inner_w)`, затем `rot90` на
  `90 deg` CW (`k=-1` / `k=3` в `torch.rot90`).

Эквивалентно spec: `top` / `bottom` через crop `1024x256` и rotate
`±90 deg` после приведения полосы к вертикальному виду. После инференса
выполняется обратная decanonicalization.

### 2.3. Поддерживаемые маски

v1 поддерживает только **rectangular mask bbox**.

В ComfyUI-ноду добавить валидацию:

```python
rectangularity = mask_area / mask_bbox_area
if rectangularity < 0.9:
    raise RuntimeError("v1 supports only rectangular masks")
```

Для irregular masks:
- либо явная ошибка;
- либо identity fallback.

### 2.4. Поведение у края изображения

Если доступный outer-контекст меньше `128 px`:
- использовать `replicate padding`;
- учитывать `edge_padded_pixels` в meta;
- маскировать pad-region в `L_identity` и `L_inner`.

Если доступный outer `< 32 px`:
- сторону полностью пропускать;
- писать warning.

Запрещено:
- zero pad;
- reflect через seam.

### 2.5. Variable inner width

Хотя canonical baseline использует `inner = 128`, сеть должна оставаться
fully-convolutional и поддерживать на inference значения:

```text
inner_width in {96, 128, 160, 192}
```

Правила:
- в ноде есть параметр `inner_width`;
- distance map нормализуется относительно фактического `inner_width`;
- на train `inner_width` случайно сэмплируется из допустимого списка.

### 2.6. Residual decay к дальнему краю inner

Чтобы не создавать новый шов на дальнем конце inner, residual должен
затухать от seam к дальнему краю inner по cosine window:

```python
t = clamp((x - seam_x) / inner_width, 0, 1)
decay = 0.5 * (1.0 + cos(pi * t))
residual = residual * decay
```

Это мягкий falloff, а не hard-gate.

### 2.7. Jitter позиции seam

На training seam сдвигается на `+-16 px`:
- целевая зона положения шва: `[112, 144]`;
- `inner_mask` и `distance_to_seam` считаются от реального jittered seam;
- на inference seam снова приводится к canonical положению.

---

## 3. Архитектура модели

### 3.1. Тип модели

Модель v1: **small ResUNet residual corrector**.

### 3.2. Базовая конфигурация

```text
in_channels   = 5
out_channels  = 3
base_channels = 32
depth         = 4
norm          = GroupNorm(8)
activation    = SiLU
residual      = yes
skip          = yes
attention     = no in v1
head          = linear conv, zero-init
```

Каналы по уровням:

```text
Encoder:   32 -> 64 -> 128 -> 256
Bottleneck 256
Decoder:   256 -> 128 -> 64 -> 32
Head:      32 -> 3
```

### 3.3. Почему не diffusion / transformer

Задача:
- low-level;
- локальная;
- photometric;
- должна быть быстрой.

Поэтому CNN/ResUNet в v1 проще и эффективнее.

### 3.4. Long-range context обязателен

Базового receptive field depth=4 недостаточно для drift вдоль шва на
сотни пикселей. Поэтому в v1 обязательно добавить два механизма:

1. **Internal low-frequency branch**
2. **GAP->FiLM в bottleneck**

Рекомендуемая реализация:

```python
class SeamResUNet(nn.Module):
    def forward(self, x):
        rgb = x[:, :3]
        lf  = gaussian_blur(rgb, sigma=6.0)

        main_feats, skips = self.encoder(x)
        lf_feats = self.lf_branch(lf)
        bn = self.bottleneck(main_feats + lf_feats)
        bn = self.apply_film(bn)
        dec = self.decoder(bn, skips)
        raw = self.head(dec)
        residual = torch.tanh(raw) * 0.3
        residual = residual * self.build_decay_mask(x)
        return residual
```

Low-frequency branch не должен нарушать контракт 5-channel input:
- `lf` строится **внутри модели** из входного RGB;
- не добавляется как внешний канал.

### 3.5. Zero-init head

Последний conv инициализируется нулями:

```python
nn.init.zeros_(self.head.weight)
nn.init.zeros_(self.head.bias)
```

Это гарантирует identity start:
- на старте `residual == 0`;
- `corrected == input`;
- training стабильнее.

### 3.6. Residual cap

Нормативный контракт выхода:

```python
raw = self.head(decoder_features)
residual = torch.tanh(raw) * 0.3
```

Причина:
- `0.1` слишком узко для комбинированных corruptions;
- `0.5` слишком рискованно;
- `0.3` покрывает training distribution с запасом.

### 3.7. Residual scale warm-up

Допустима и рекомендована learnable scalar-переменная:

```python
self.residual_scale = nn.Parameter(torch.tensor(0.1))
scale = torch.clamp(self.residual_scale, 0.0, 1.0)
residual = torch.tanh(raw) * 0.3 * scale
```

### 3.8. Что откладывается на v2+

В v1 не входят:
- self-attention / axial attention;
- two-stream reference-aware encoder;
- CoordConv;
- canvas/source-id channels.

### 3.9. Опциональный режим `low_freq_only` (config)

В `configs/train_synth_v1.yaml` допускается флаг:

```yaml
model:
  residual_mode: full          # или low_freq_only
  low_freq_sigma: 6.0
```

При `low_freq_only` к предсказанному residual перед сложением с RGB
применяется Gaussian blur с `sigma = low_freq_sigma`, чтобы подавить
HF-артефакты на чувствительных сценах. **Дефолт v1:** `full`.

---

## 4. Входы, выходы и нормализация

### 4.1. Входные каналы

Единственный контракт входа:

```python
input_tensor: FloatTensor[B, 5, 1024, 256]

channels:
  [0:3] strip_rgb
  [3:4] inner_mask
  [4:5] distance_to_seam
```

### 4.2. RGB policy

- диапазон: `[0, 1]`, `float32`;
- вход хранится в **sRGB gamma space**;
- z-score normalization не используется;
- линеаризация RGB для метрик Lab / CIEDE2000 выполняется **только**
  внутри кода метрик (`skimage.color.rgb2lab` и т.п.), не на входе сети;
- clamp выполняется только после применения residual на inference.

### 4.3. `inner_mask`

```python
inner_mask(x, y) = 1.0 if x >= x_seam else 0.0
```

Mask обязана:
- быть бинарной;
- учитывать jittered seam;
- вращаться и флипаться вместе с RGB при augmentation.

### 4.4. `distance_to_seam`

```python
distance_to_seam(x, y) = abs(x - x_seam) / max_distance
```

Где:
- `x_seam` - фактическая позиция шва в данном sample;
- `max_distance = max(x_seam, W - x_seam)` для ширины полосы `W`
  (per-sample), чтобы значение оставалось в `[0, 1]` при jitter и
  variable `inner_width`.

Это важно при:
- seam jitter;
- variable `inner_width`.

### 4.5. Почему `inner_mask` и `distance_to_seam` не дублируют друг друга

В v1 оба канала нужны:
- `inner_mask` даёт sharp editable region;
- `distance_to_seam` даёт smooth positional prior.

При jitter и variable width они перестают быть тривиально взаимно
выводимыми.

### 4.6. Выход модели

Выход модели:

```python
residual_rgb: FloatTensor[B, 3, H, W]
```

Финальная схема применения:

```python
residual = model(x)
corrected = input_rgb + strength * residual
corrected[:, :, :, outer_slice] = input_rgb[:, :, :, outer_slice]
corrected = corrected.clamp(0.0, 1.0)
```

### 4.7. Outer hard-copy

Нормативная гарантия inference:

```text
corrected[outer_region] == input[outer_region]
```

На inference это обеспечивается **физическим копированием** outer-пикселей
из входа после merge.

Дополнительно требуется assert:

```python
max_diff = (corrected[outer] - input[outer]).abs().max()
assert max_diff < 1e-6
```

---

## 5. Датасет и подготовка исходных данных

### 5.1. Источник данных для v1

v1 использует только:

```text
input_raw/ -> prepared source images -> synthetic strip dataset
```

Real pipeline dataset не используется в v1 train loop.

### 5.2. Подготовка `input_raw/`

Нужен offline script:

```text
scripts/prepare_source.py
```

Он обязан:
1. читать `.jpg/.jpeg/.png`;
2. пропускать `.txt`, `.DS_Store` и битые файлы;
3. применять EXIF transpose;
4. конвертировать ICC -> sRGB;
5. отбрасывать RGBA;
6. режим `L` / `P` / `CMYK` конвертировать в `RGB` перед дальнейшей
   обработкой;
7. center-crop до квадрата;
8. resize до `1024x1024`;
9. сохранять prepared PNG в `data/source_images/`;
10. вычислять `phash64`;
11. собирать manifest.

Причины отказа (RGBA, слишком малый размер, битый файл и т.д.) логировать
в `outputs/eval_reports/excluded_sources.jsonl` (append-only).

### 5.3. Исключения на preprocess

Обязательно исключать:
- RGBA PNG;
- `.DS_Store`;
- text captions из image loader;
- файлы, которые не открываются;
- файлы меньше `512 px` по меньшей стороне.

### 5.4. Manifest schema

Нормативный manifest для source images:

```json
{
  "id": "000123",
  "source_path": "data/source_images/000123.png",
  "original_path": "input_raw/000123.jpg",
  "caption_path": "input_raw/000123.txt",
  "caption": "Red-brown leaves scattered on white background...",
  "scene_tags": ["leaves", "white_background"],
  "phash64": "abc123de45f67890",
  "cluster_id": 42,
  "split": "train",
  "width": 1024,
  "height": 1024,
  "has_icc": true,
  "sha256": "...",
  "source_domain": "photo"
}
```

Путь:

```text
manifests/input_raw_manifest.jsonl
```

### 5.5. Split policy

Train/val/bench split делается **по pHash clusters**, а не по отдельным
изображениям:

```text
Hamming(phash_a, phash_b) <= 6 -> same cluster
```

Поля `cluster_id` и `split` в manifest на этапе `prepare_source` могут
быть `null` и заполняются на шаге `build_split.py`.

Затем кластеры делятся, например:
- `80% train`
- `12% val`
- `8% bench`

### 5.6. Scene tags

Scene tags извлекаются из caption и сохраняются в manifest. Они нужны для:
- hard sample mining;
- validation buckets;
- отчетов.

Критичные теги:
- `sky`
- `skin`
- `gradient`
- `night`
- `wall`
- `water`
- `glass`
- `architecture`
- `leaves`

Для приоритета сложных сцен допускается `WeightedRandomSampler` с весами
по `scene_tags` (например удвоить вес для подмножества
`{sky, skin, gradient, night, wall}`).

---

## 6. Synthetic strip dataset

### 6.1. Общая идея

Для каждого source image:
- сэмплируются strip-конфигурации;
- извлекается clean target strip;
- портится только inner half;
- outer half остается эталонной.

### 6.2. Покрытие source image

На одно source image за эпоху нужно брать **20-30 strips**.

Детерминированное пространство конфигураций (до джиттера и коррапций):

```text
оси:              {vertical, horizontal}                    # 2
seam positions:   сетка из 8 позиций в [W/4, 3W/4]         # 8
flip:             {none, horizontal_flip}                  # 2
rotation_k:       {0, 1, 2, 3}                             # 4

Итого: 2 * 8 * 2 * 4 = 128 базовых вариантов на изображение.
```

За эпоху — **случайный subsample без повторов** нужного размера
(`20-30`) из этого пространства (плюс `seam_jitter_px`,
`inner_width`, коррапции). Sampling должен покрывать:
- вертикальные и горизонтальные оси;
- разные позиции seam;
- flip;
- `rotation_k in {0, 1, 2, 3}`;
- `inner_width in {96, 128, 160, 192}`.

### 6.3. Пример dataset-конфига

```python
cfg = {
    "axis": rng.choice(["vertical", "horizontal"]),
    "seam_x_frac": rng.uniform(0.25, 0.75),
    "flip_h": rng.random() < 0.5,
    "rotation_k": rng.integers(0, 4),
    "seam_jitter_px": rng.integers(-16, 17),
    "inner_width": rng.choice([96, 128, 160, 192]),
}
```

### 6.4. Canonicalization invariant

До модели все strips приводятся к виду:
- вертикальный tensor;
- outer слева;
- inner справа.

**Random rotation / flip (train, D9):** собирать `input_5ch =
concat(strip_rgb, inner_mask, distance_to_seam)` и применять одну и ту
же геометрию ко всем 5 каналам и к `target` (общий seed), чтобы маска и
distance оставались согласованы с RGB.

Должен существовать unit test:

```python
decanonicalize(canonicalize(strip, side), side) == strip
```

### 6.5. Формат sample из `Dataset.__getitem__`

```python
{
    "input": FloatTensor[5, H, W],
    "target": FloatTensor[3, H, W],
    "input_rgb": FloatTensor[3, H, W],
    "mask": FloatTensor[1, H, W],
    "distance": FloatTensor[1, H, W],
    "inner_region_mask": FloatTensor[1, H, W],
    "boundary_band_mask": FloatTensor[1, H, W],
    "meta": {
        "image_id": "...",
        "axis": "vertical",
        "side": "left",
        "rotation_k": 2,
        "flip_h": false,
        "seam_jitter_px": 8,
        "inner_width": 128,
        "edge_padded_pixels": 0,
        "ops": ["exposure", "temperature"]
    }
}
```

### 6.6. Fixed validation cache

Validation должен быть cached на диск детерминированно. Layout совпадает
с §13.3 (одна директория на sample):

```text
outputs/strip_cache/val/{sample_id}/
  input.png
  target.png
  mask.png
  distance.png
  meta.json
```

---

## 7. Synthetic corruptions

### 7.1. Общий принцип

В v1 разрешены только **photometric / tonal / low-frequency** corruptions.
Они применяются **только к inner half**.

### 7.2. Разрешенные группы corruptions

#### A. Exposure / brightness
- additive brightness: `[-0.08, +0.08]`
- exposure EV: `[-0.3, +0.5]`
- multiplicative gain: `[0.85, 1.20]`

#### B. Contrast / gamma
- contrast: `[0.80, 1.25]`
- gamma: `[0.85, 1.20]`

#### C. Color balance
- temperature: `[-0.06, +0.06]`
- tint: `[-0.05, +0.05]`
- per-channel gains: `[0.90, 1.10]`
- hue shift: `[-12 deg, +12 deg]`
- saturation: `[0.75, 1.35]`

#### D. Tone curves
- shadow lift/crush: `[0.0, 0.12]`
- highlight rolloff: `[0.0, 0.12]`
- midtone shift: `[-0.08, +0.08]`

#### E. Spatial low-frequency drift
- brightness gradient: `[0.0, 0.12]`
- color gradient: `[0.0, 0.12]`
- low-order polynomial illumination: `[0.0, 0.10]`
- vignette-like field: `[0.0, 0.15]`

#### F. Mild pipeline-like defects
- Gaussian blur sigma: `[0.0, 1.5]`
- JPEG quality simulation: `[75, 95]`
- mild noise sigma: `[0.0, 0.01]`
- mild microcontrast shift: `[0.0, 0.1]`

### 7.3. Запрещенные corruptions

Запрещено:
- VAE round-trip;
- геометрические искажения;
- semantic corruption;
- сильный blur `sigma > 2.0`;
- сильный noise `sigma > 0.05`;
- тяжелые compression artifacts;
- всё, что превращает задачу в restoration вместо seam correction.

### 7.4. Combination policy

На sample применять `2-5` операций:

```python
n_ops = rng.choice([2, 3, 4, 5], p=[0.2, 0.4, 0.3, 0.1])
```

Правила:
- минимум одна операция из групп `A/B/C` (exposure / contrast / color);
- минимум одна из группы `D` (tone curves);
- иногда одна из `E`;
- редко одна из `F`.

### 7.5. Optional block-level corruption

Разрешено добавить легкий block-level corruption `16x16` с вероятностью
`p ≈ 0.15`, чтобы слегка приблизиться к future VAE-like pipelines, но без
фактического VAE round-trip.

---

## 8. Loss-функции

### 8.1. Total loss

Нормативный стартовый состав:

```python
L_total = (
      1.00 * L_inner
    + 2.00 * L_boundary
    + 1.00 * L_lowfreq_ms
    + 0.50 * L_grad
    + 0.20 * L_identity
    + 0.10 * L_lpips_hf
    + 0.05 * L_residual_smooth
    + 0.01 * L_residual_magnitude
)
```

### 8.2. `L_inner`

Использовать Charbonnier на inner region:

```python
diff = (pred - target) * inner_mask
L_inner = torch.sqrt(diff * diff + 1e-3 * 1e-3).mean()
```

### 8.3. `L_boundary`

Boundary band должен быть усилен:
- рабочий band: `24 px`;
- допустимый смысловой диапазон: `16-32 px`.

```python
band = abs(x - seam_x) <= 24
```

### 8.4. `L_lowfreq_ms`

Обязателен multi-scale low-frequency loss:

```python
sigmas = (2.0, 4.0, 8.0, 16.0, 32.0)
```

Считать только внутри inner region.

### 8.5. `L_grad`

Использовать Sobel-based gradient consistency.

### 8.6. `L_identity`

Считать на outer region:

```python
diff = (pred - input_rgb) * outer_mask
L_identity = diff.abs().mean()
```

Этот loss обязателен даже при hard-copy outer на inference.

### 8.7. `L_lpips_hf`

Использовать LPIPS только на **high-frequency component** и только
на boundary band:

```python
hf = img - gaussian_blur(img, sigma=4.0)
```

Сеть LPIPS ожидает вход в диапазоне `[-1, 1]` на канал; перед вызовом:
`hf_norm = hf * 2.0 - 1.0` (аналогично для `target`).

Вес в total loss: `0.10`.

### 8.8. `L_residual_smooth`

Наказывает резкие spatial производные residual:

```python
dx = residual[..., :, 1:] - residual[..., :, :-1]
dy = residual[..., 1:, :] - residual[..., :-1, :]
```

### 8.9. `L_residual_magnitude`

Guard loss против numerical spikes:

```python
over = (residual.abs() - 0.3).clamp(min=0)
L_residual_magnitude = over.mean()
```

Обычно почти ноль благодаря `tanh`.

### 8.10. Опциональные ablations

Можно исследовать:
- luma-weighted `L_inner`;
- MS-SSIM вместо части pixel loss;
- low-frequency-only residual mode.

Но не включать в baseline v1 по умолчанию.

---

## 9. Метрики, evaluation, acceptance

### 9.1. Основные метрики

Для model selection считать:

1. `Boundary CIEDE2000`
2. `Boundary MAE`
3. `Inner MAE`
4. `Outer identity error`
5. `Low-frequency MAE (sigma=16)`
6. `Residual magnitude stats`
7. `LPIPS HF` на boundary band

Primary metric v1:

```text
Boundary CIEDE2000
```

### 9.2. CIEDE2000 policy

Lab conversion делать через `skimage.color.rgb2lab` на входе в sRGB
диапазоне `[0, 1]`.

Запрещено передавать туда linearized RGB.

### 9.3. Baseline и oracle

Для каждого eval sample логировать:
- baseline metric на `input` vs `target`;
- metric модели (`pred` vs `target`);
- oracle low-frequency-matched baseline (например
  `match_lowfreq(input, target, sigma=16)` vs `target`).

Относительное улучшение (для отчётов и gates):

```text
relative_improvement =
  (baseline - pred) / (baseline - oracle + 1e-8)
```

Интерпретация: `> 0.5` — модель закрывает более половины зазора до
oracle; `> 0.8` — близко к потолку для выбранного oracle.

Это позволяет измерять не просто абсолютное улучшение, а долю от
доступного потолка.

### 9.4. Validation buckets

Считать метрики не только по среднему, но и по bucket'ам:

Scene buckets:
- sky / gradients
- skin / faces
- walls / interiors
- architecture
- foliage / textures
- procedural textures
- water / reflections
- night / high contrast

Orientation buckets:
- vertical seam
- horizontal seam

Corruption buckets:
- subtle
- medium
- strong

Опционально для v2+ (real validation): buckets по neighbor-set из
canvas-spec (`2^8 - 1` конфигураций, сведённые к каноническим классам
симметрии).

### 9.5. Visual report

`run_eval.py` обязан сохранять артефакты в каталог вида
`outputs/eval_reports/run_YYYYMMDD_HHMMSS/`:

- `summary.json` (метрики, CI, прохождение gates);
- `gates.txt` (человекочитаемый статус);
- `metrics_by_bucket.csv`;
- `visuals/best_10/`, `visuals/median_10/`, `visuals/worst_10/`;
- `visuals/grids/` — на sample: `input | pred | target | |error|×5 |
  residual×10 | low-frequency (σ=16)`;
- `seam_profile_plots/` — 1D профиль вдоль шва (pred vs target), опционально
  для подвыборки.

Дополнительно для отладки merge в углах (все 4 стороны активны): визуализация
per-side weight maps и итогового merged residual.

### 9.6. Numerical gates для v1

Production-candidate на synthetic val только если выполняются все условия:

```yaml
gates:
  boundary_ciede2000_mean:
    pred < 0.7 * baseline
  boundary_ciede2000_p95:
    pred < 0.8 * baseline
  outer_identity_mae:
    max < 1e-6
  lpips_hf_delta_mean:
    < 0.02
  residual_magnitude_p99:
    < 0.3
  worst_bucket_relative_improvement:
    > 0.2
```

### 9.7. Bootstrap CI

Все основные метрики в отчете должны иметь 95% bootstrap CI.

---

## 10. Train protocol

### 10.1. Baseline hyperparameters

```yaml
train:
  seed: 42
  optimizer: AdamW
  lr: 2.0e-4
  weight_decay: 1.0e-4
  betas: [0.9, 0.999]
  grad_clip: 1.0
  precision: bf16
  batch_size: 16
  num_epochs: 20

scheduler:
  type: cosine_with_warmup
  warmup_steps: 1000
  min_lr: 1.0e-6

ema:
  enabled: true
  decay: 0.999

plateau:
  patience_epochs: 3
  reduce_factor: 0.5
  min_lr_stop: 1.0e-6

early_stop:
  patience_epochs: 5
  metric: val_boundary_ciede2000
```

### 10.2. EMA обязателен

В v1 EMA больше не опционален.
Validation и export выполняются по `ema_model`, а не по raw weights.

### 10.3. Worker seeding

Нужен reproducible worker seeding для `numpy`, `random` и `torch`
workers. Пример для `DataLoader`:

```python
def worker_init_fn(worker_id):
    base = torch.initial_seed() % (2**32)
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)

DataLoader(..., worker_init_fn=worker_init_fn, persistent_workers=True)
```

### 10.4. Checkpoint policy

Сохранять:
- `last.pt`
- `best_boundary_ciede2000.pt`
- `best_boundary_mae.pt`
- `best_outer_identity.pt`
- `best_relative_improvement.pt`

Checkpoint должен содержать:
- raw model;
- EMA state;
- optimizer;
- scheduler;
- scaler;
- epoch;
- RNG states;
- `config_hash`;
- `git_hash`.

### 10.5. Resume + determinism

Resume training обязан восстанавливать:
- raw model;
- EMA state;
- optimizer;
- scheduler;
- scaler;
- epoch;
- все RNG states (`torch`, `cuda`, `numpy`, `python`);
- `config_hash` и `git_hash`.

Цель:
- reproducible restart без скрытого сдвига training trajectory.

### 10.6. Dataset sizing

При `~606` valid prepared images и `25 strips / image / epoch`:
- около `15k strips / epoch`;
- около `300k seen strips` за `20 epochs`.

### 10.7. Baseline ablation plan

Порядок ablations:
1. `L_lowfreq_ms` on/off
2. GAP->FiLM on/off
3. residual cap `0.1 / 0.3 / 0.5`
4. rotation aug on/off
5. seam jitter on/off
6. `base_channels`: `32 / 48 / 64`
7. GroupNorm vs InstanceNorm
8. LPIPS HF on/off
9. luma-weighted pixel loss
10. `low_freq_only` residual mode

### 10.8. Monitoring / logging

Per-step логировать:
- все компоненты loss;
- `residual` stats: `mean / max / p99`;
- learning rate;
- grad norm.

Per-epoch логировать:
- validation metrics;
- gates pass/fail;
- bootstrap CI;
- visual report.

Backend:
- TensorBoard или W&B;
- JSONL fallback в `outputs/eval_reports/run_*/metrics.jsonl`.

---

## 11. Export и ComfyUI node

### 11.1. Формат export

Экспортировать:
- EMA weights в `.safetensors`;
- sidecar JSON с полным config-контрактом.

### 11.2. Sidecar minimum schema

```json
{
  "model_name": "seam_residual_corrector_v1",
  "schema_version": 1,
  "architecture": {
    "in_channels": 5,
    "out_channels": 3,
    "base_channels": 32,
    "depth": 4,
    "bottleneck": {
      "gap_film": true,
      "lf_branch": true
    },
    "residual_cap_tanh_scale": 0.3,
    "decay_window": "cosine"
  },
  "strip": {
    "canonical_shape_chw": [5, 1024, 256],
    "outer_width": 128,
    "inner_width_default": 128,
    "supported_inner_widths": [96, 128, 160, 192],
    "seam_jitter_train_px": 16
  },
  "preprocess": {
    "rgb_range": [0.0, 1.0],
    "expects_srgb_gamma": true,
    "channels_order": ["R", "G", "B", "inner_mask", "distance_to_seam"]
  },
  "orientation": {
    "canonical": "vertical_outer_left",
    "train_rotation_aug": true
  },
  "inference": {
    "strength_range": [0.0, 1.0],
    "strength_default": 1.0,
    "hard_copy_outer": true,
    "clamp_output": [0.0, 1.0]
  },
  "training": {
    "dataset": "synthetic_only_v1",
    "epochs": 20,
    "ema_decay": 0.999,
    "git_hash": "<filled at export>",
    "commit_date": "<filled at export>"
  }
}
```

### 11.3. Verify-export smoke test

После export обязательно:
1. загрузить `.safetensors` и sidecar;
2. собрать модель из sidecar;
3. прогнать `1-3` sample strips;
4. сравнить output с checkpoint output;
5. убедиться, что `max diff < 1e-5`.

### 11.4. Contract с canvas-spec

`seam_canvas_target_spec.md` используется только как reference для
ComfyUI-ноды:
- какие типы `IMAGE + MASK` может получить нода;
- как трактовать bbox и стороны маски;
- какая последовательность side-inference допустима.

Canvas-logic не должна попадать внутрь модели.
Максимум side-inference в одном вызове: 4 стороны.

### 11.5. ComfyUI node inputs

Нода принимает:
- `IMAGE`
- `MASK`
- `model_path`
- `inner_width`
- `strength`
- `process_left`
- `process_right`
- `process_top`
- `process_bottom`
- `debug_previews`

`outer_width` в v1 фиксирован контрактом sidecar и не должен свободно
разъезжаться с моделью.

Если пользовательский `inner_width` не входит в
`strip.supported_inner_widths` sidecar — `RuntimeError` с понятным
сообщением.

### 11.6. ComfyUI node logic

Нормативный порядок:

1. валидировать mask и bbox;
2. для каждой активной стороны извлечь strip;
3. canonicalize;
4. собрать 5-channel tensor;
5. прогнать модель;
6. decanonicalize residual;
7. собрать `side_residuals`;
8. merge side residuals (COLA-safe взвешенное суммирование, §12.4);
9. применить merged residual только внутри mask;
10. hard-copy outer из исходного full image;
11. clamp в `[0, 1]`.

### 11.7. Sidecar validation

Перед inference нода обязана проверить:
- `schema_version` поддерживается;
- `in_channels == 5`;
- `canonical_shape_chw == [5, 1024, 256]`;
- `outer_width == 128`;
- `residual_cap_tanh_scale == 0.3`;
- `hard_copy_outer == true`;
- пользовательский `inner_width` входит в `supported_inner_widths`.

При несовместимости:
- явная ошибка;
- silent fallback запрещен.

### 11.8. Strength policy

В v1:
- default `strength = 1.0`;
- UI range: `[0.0, 1.0]`;
- значения `> 1.0` запрещены.

### 11.9. Model caching

Нода обязана кэшировать уже загруженные модели по ключу `(path, device)`.

### 11.10. Edge cases в ноде

- irregular mask -> error / identity fallback;
- outer `< 32 px` на стороне -> skip side;
- толщина inner вдоль нормали к шву `< 64 px` -> skip corrector, вернуть
  вход;
- full-image mask -> warning + skip.

### 11.11. Debug preview output

При `debug_previews=true` сохранять:
- `input.png`;
- `side_{left,right,top,bottom}_residual.png`;
- `merged_residual.png`;
- `weight_map_{side}.png`;
- `summary.json` с per-side stats и ключевыми метриками.

Путь:

```text
outputs/debug_previews/{timestamp}/
```

---

## 12. Merge в углах

### 12.1. Проблема

Угловые области попадают одновременно в 2 side-corrections:
- left + top
- left + bottom
- right + top
- right + bottom

### 12.2. Требования к merge

Merge обязан гарантировать:
- outer contamination отсутствует;
- correction не усиливается искусственно;
- corner transition гладкий;
- unit tests покрывают симметрию и identity.

### 12.3. Weight maps

Для каждой стороны строится Hann-like weight map вдоль глубины inner:
- вес максимален у seam;
- вес затухает к дальнему краю inner;
- вне editable region вес равен нулю.

### 12.4. Практический merge-контракт

Нормативное требование для реализации:
- side residuals вне своей editable region должны быть нулевыми;
- итоговый merge должен сохранять `merged_residual[outer] == 0`;
- итог не должен превышать амплитуду strongest side residual больше,
  чем на численную погрешность.

**Запрещённая** эвристика вида `sum(w_i * c_i) / (sum(w_i) + eps)` для
смешивания **несогласованных** карт коррекций: на outer-пикселе, где
своя сторона даёт `c=0`, чужая сторона может оставить ненулевой вклад
после нормировки (нарушение hard-copy).

**Допустимая** схема (COLA-safe, согласована с D4): для каждой стороны
построить неотрицательные веса `w_side` (Hann / wedge), равные `0` на
outer и вне editable region данной стороны; затем

```python
total_w = sum(w_side for all sides) + 1e-8
merged = sum((w_side / total_w) * correction_side for all sides)
merged *= mask_inner_region   # финальное обнуление вне маски
```

При `w_side == 0` на outer вклад этой стороны нулевой; деление на
`total_w` не создаёт утечки, если все активные `correction_side` там
тоже нулевые (см. unit tests §12.5).

Допустимы эквивалентные реализации Hann overlap-add при тех же
инвариантах и покрытии тестами.

### 12.5. Обязательные unit tests

Нужны тесты:
1. `one_side_only`;
2. `corner_symmetry`;
3. `outer_identity_exact`;
4. `no_amplification`;
5. `canonicalize -> decanonicalize roundtrip`.

---

## 13. Dataset viewer

### 13.1. Старый viewer считать устаревшим

Старый `dataset_viewer.py` относился к предыдущей canvas-задаче и не
является валидной основой для v1.

### 13.2. Новый viewer

Новая утилита:

```text
strip_dataset_viewer.py
```

Нужна для:
- просмотра cached strips;
- фильтрации по split / scene / ops / orientation;
- просмотра input / target / mask / distance;
- сравнения нескольких prediction runs.

### 13.3. Refactor policy

Viewer должен быть переписан под strip-based pipeline.

Разрешено переиспользовать:
- базовый FastAPI setup;
- comparison slider;
- lightbox / zoom.

Нужно удалить legacy-логику canvas-v2:
- regimes / neighbors / difficulty;
- старые dataset roots;
- старые endpoints, завязанные на canvas composition.

### 13.4. Viewer dataset layout

```text
outputs/strip_cache/{split}/{id}/
  input.png
  target.png
  mask.png
  distance.png
  meta.json
```

### 13.5. Viewer sample schema

`meta.json` должен содержать минимум:

```json
{
  "id": "000001",
  "source_image": "data/source_images/000123.png",
  "cluster_id": 42,
  "split": "val",
  "scene_tags": ["sky", "gradient"],
  "strip": {
    "axis": "vertical",
    "original_side": "left",
    "seam_x_frac_in_source": 0.5,
    "flip_h": false,
    "rotation_k": 2,
    "seam_jitter_px": 8,
    "inner_width": 128,
    "edge_padded_pixels": 0
  },
  "corruption": {
    "ops": ["exposure", "temperature", "gradient_drift"]
  },
  "metrics_precomputed": {
    "baseline_boundary_mae": 0.042,
    "baseline_boundary_ciede2000": 4.7
  }
}
```

### 13.6. Viewer UI minimum

UI должен содержать:
- sidebar с фильтрами по split / tags / orientation / ops;
- strip inspector;
- comparison slider;
- seam close-up zoom;
- dataset stats tab;
- run comparison tab.

### 13.7. Viewer API minimum

Нужны endpoints:
- `/api/samples`
- `/api/sample/{id}`
- `/api/strip/{id}/input.png`
- `/api/strip/{id}/target.png`
- `/api/strip/{id}/residual.png`
- `/api/strip/{id}/error.png`
- `/api/strip/{id}/seam_profile`
- `/api/strip/{id}/histogram`
- `/api/stats`
- `/api/runs`

Опционально:
- `/api/run/{run_id}/metrics`
- `/api/inspect_strip/{id}`.

### 13.8. Viewer config / security

- root path задается через CLI/ENV;
- bind по умолчанию только на `127.0.0.1`;
- dev-only запуск через явный CLI flag допустим и рекомендован;
- path resolution через `Path(__file__).resolve().parent`;
- кэш и refresh должны быть thread-safe.

### 13.9. Viewer performance

- metadata читать из `meta.json`, а не вычислять на лету из PNG;
- thumbnails допускается готовить offline (`thumb_128.png`);
- кэширование `meta.json` и run summaries допустимо через LRU.

### 13.10. Viewer code layout

UI следует вынести из inline-HTML в:
- `static/index.html`;
- `static/viewer.js`;
- `static/viewer.css`.

### 13.11. Viewer optional debug panels

Опционально добавить:
- histogram view;
- spectral / FFT view на boundary band;
- overlay weight maps для debug corner merge.

---

## 14. Структура проекта

Нормативная baseline-структура. `README.md` должен явно указывать
**research_only / internal use** и непубликацию `input_raw/` (D15).

```text
unet_seam/
├─ input_raw/
├─ data/
│  └─ source_images/
├─ manifests/
│  ├─ input_raw_manifest.jsonl
│  ├─ source_train.jsonl
│  ├─ source_val.jsonl
│  ├─ source_bench.jsonl
│  └─ strip_val_cache.jsonl
├─ configs/
│  ├─ model_resunet_v1.yaml
│  ├─ train_synth_v1.yaml
│  ├─ eval_v1.yaml
│  └─ export_v1.yaml
├─ src/
│  ├─ data/
│  │  ├─ manifest.py
│  │  ├─ synthetic_strip_dataset.py
│  │  ├─ strip_geometry.py
│  │  ├─ corruptions.py
│  │  └─ preprocess.py
│  ├─ models/
│  │  ├─ resunet.py
│  │  └─ blocks.py
│  ├─ losses/
│  │  ├─ seam_losses.py
│  │  ├─ lowfreq.py
│  │  ├─ perceptual.py
│  │  └─ residual_guard.py
│  ├─ metrics/
│  │  ├─ seam_metrics.py
│  │  ├─ deltae.py
│  │  ├─ lowfreq_metrics.py
│  │  ├─ bootstrap.py
│  │  └─ reports.py
│  ├─ infer/
│  │  ├─ extract_strips.py
│  │  ├─ merge_residuals.py
│  │  └─ correct_full_frame.py
│  ├─ train/
│  │  ├─ train_loop.py
│  │  ├─ ema.py
│  │  ├─ checkpoint.py
│  │  └─ scheduler.py
│  └─ utils/
│     ├─ image_io.py
│     ├─ phash.py
│     └─ seed.py
├─ scripts/
│  ├─ prepare_source.py
│  ├─ build_split.py
│  ├─ preview_synthetic_strips.py
│  ├─ cache_val_strips.py
│  ├─ train_resunet.py
│  ├─ run_eval.py
│  ├─ run_bench.py
│  ├─ export_safetensors.py
│  ├─ verify_export.py
│  └─ smoke_test_comfy_node.py
├─ comfy_node/
│  ├─ __init__.py
│  ├─ seam_corrector_node.py
│  ├─ strip_ops.py
│  ├─ hann_merge.py
│  └─ model_loader.py
├─ outputs/
│  ├─ strip_cache/
│  ├─ checkpoints/
│  ├─ eval_reports/
│  ├─ exports/
│  ├─ runs/
│  └─ debug_previews/
├─ tests/
│  ├─ test_strip_geometry.py
│  ├─ test_corner_merge.py
│  ├─ test_residual_bound.py
│  ├─ test_synthetic_dataset.py
│  └─ test_hann_windows.py
├─ strip_dataset_viewer.py
├─ static/
├─ pyproject.toml
├─ requirements.txt
├─ README.md
├─ CLAUDE.md
├─ seam_canvas_target_spec.md
├─ seam_residual_corrector_spec.md
└─ audits/
   ├─ audit_consolidated.md
   ├─ audit_claude_opus.md
   ├─ аудит_Codex.md
   ├─ аудит_unet_Composer.md
   └─ audit_gemini.md
```

### 14.1. `.gitignore`

Минимум:

```gitignore
input_raw/
data/source_images/
outputs/
__pycache__/
*.pyc
.venv/
.vscode/
.idea/
.DS_Store
*.safetensors
*.pt
```

### 14.2. `requirements.txt`

Минимальный состав:

```text
torch>=2.3
torchvision>=0.18
numpy>=1.24
pillow>=10.0
opencv-python>=4.9
scipy>=1.11
scikit-image>=0.22
safetensors>=0.4
pyyaml>=6.0
tqdm>=4.65
lpips>=0.1.4
imagehash>=4.3
fastapi>=0.110
uvicorn>=0.27
pytest>=8.0
```

---

## 15. Скрипты и smoke tests

### 15.1. Обязательные scripts v1

- `scripts/prepare_source.py`
- `scripts/build_split.py`
- `scripts/preview_synthetic_strips.py`
- `scripts/cache_val_strips.py`
- `scripts/train_resunet.py`
- `scripts/run_eval.py`
- `scripts/run_bench.py`
- `scripts/export_safetensors.py`
- `scripts/verify_export.py`
- `scripts/smoke_test_comfy_node.py`

### 15.2. Smoke tests

Нужны smoke tests:

1. Dataset smoke test
   - shapes корректны;
   - mask корректна;
   - нет NaN/Inf.

2. Model smoke test
   - один forward pass;
   - loss считается;
   - backward проходит.

3. Export smoke test
   - safetensors грузится;
   - output совпадает с checkpoint.

4. Comfy node smoke test
   - node обрабатывает sample;
   - shape совпадает;
   - repeated runs не текут по памяти.

---

## 16. Roadmap

### 16.1. v1

- synthetic-only training;
- fixed rectangular masks;
- small ResUNet;
- 5-channel input;
- residual seam correction;
- ComfyUI integration;
- seam-specific eval gates.

### 16.2. v2

Рассматривать только после стабильного v1:
- real pipeline finetune;
- irregular masks;
- axial attention;
- improved corner weighting;
- distance-transform merge;
- real validation buckets по neighbor-sets.

### 16.3. Если v1 не хватает

Последовательность эскалации:
1. расширить `inner_width`;
2. включить low-frequency-only residual mode;
3. собрать real dataset для v2;
4. усилить long-range branch;
5. добавить second pass глубже внутрь.

---

## 17. Definition of Done

Проект считается доведенным до рабочего v1, если:

1. есть reproducible repo structure;
2. `prepare_source.py` и split pipeline работают;
3. synthetic dataset корректно строится и валидируется;
4. baseline model стабильно обучается;
5. eval считает seam-specific metrics и gates;
6. export в `.safetensors` проходит verify-load;
7. ComfyUI node работает на ручных кейсах;
8. outer identity сохраняется строго;
9. synthetic acceptance gates пройдены;
10. viewer и debug artifacts позволяют разбирать failed cases.

---

## 18. Финальная рекомендация для реализации

Самый рациональный v1:
- strip-based approach;
- canonical strips `256x1024`;
- `RGB + inner_mask + distance_to_seam`;
- small ResUNet с internal LF branch и GAP->FiLM;
- bounded residual `tanh * 0.3`;
- hard-copy outer;
- synthetic-only train;
- seam-specific eval;
- export в safetensors + sidecar;
- ComfyUI node после финального composite.

Если проект реализован по этому документу, архитектурной
неопределенности для v1 оставаться не должно.
