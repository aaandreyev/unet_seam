**Цель**

Мы строим модель, которая исправляет seam так, чтобы одновременно выполнялись две вещи:

- шов реально исчезал по пиксельным и цветовым метрикам
- модель не вносила новых артефактов: halo, glare, narrow dark line, overcorrection, лишний low-frequency drift

То есть цель не просто “снизить `MAE`”, а получить **сильную seam correction без агрессивного correction style**.

---

**Что за что отвечает**

**1. Датасет / corruptions**  
Файлы:
- [src/data/corruptions.py](/Users/andreyev-a/pet_projects/unet_seam/src/data/corruptions.py:1)
- [src/data/synthetic_strip_dataset.py](/Users/andreyev-a/pet_projects/unet_seam/src/data/synthetic_strip_dataset.py:1)

Они отвечают за то, на каких synthetic mismatch’ах учится модель.

Тут есть 4 группы corruptions:
- `A`: базовые color/tonal shifts
- `B`: tone-curve style shifts
- `C`: явные spatial fields
- `D`: degradation-like ops

Именно здесь задаётся:
- сколько ops в sample
- сколько uniform vs spatial
- насколько spatial fields плавные, granular, seam-biased или neutral

Если этот блок слишком слабый:
- модель недоучивается на сложных non-uniform seam cases

Если слишком агрессивный:
- модель начинает учиться synthetic circus, а не реальному harmonization

---

**2. Модель**  
Файл:
- [src/models/harmonizer.py](/Users/andreyev-a/pet_projects/unet_seam/src/models/harmonizer.py:1)

Она предсказывает:
- corrected strip
- low-frequency correction fields
- detail correction
- confidence/gate related maps

Это “двигатель”, который реально исправляет seam.

---

**3. Loss**  
Файл:
- [src/losses/harmonizer_losses.py](/Users/andreyev-a/pet_projects/unet_seam/src/losses/harmonizer_losses.py:1)

Loss отвечает за то, **какой стиль исправления считается хорошим**.

Условно:
- `rec`, `seam`, `lab`, `profile` отвечают за качество совмещения
- `low`, `overcorr`, `field`, `detail`, `matrix` отвечают за аккуратность correction style
- `conf_align`, `conf_metric`, `conf_budget`, `gate` пытаются дисциплинировать confidence/gating behavior

---

**4. Выбор лучшего checkpoint**  
Файл:
- [scripts/train_harmonizer.py](/Users/andreyev-a/pet_projects/unet_seam/scripts/train_harmonizer.py:1)

Функция `_quality(...)` решает:
- какой checkpoint считать “best quality”

Это критично, потому что можно обучить хороший run, но выбрать не тот epoch.

---

**5. Dashboard / оценка**  
Файлы:
- [scripts/analyze_tfevents_harmonizer.py](/Users/andreyev-a/pet_projects/unet_seam/scripts/analyze_tfevents_harmonizer.py:1)
- [scripts/harmonizer_metrics_dashboard.py](/Users/andreyev-a/pet_projects/unet_seam/scripts/harmonizer_metrics_dashboard.py:1)

Они не обучают модель, а помогают понять:
- реально ли она улучшается
- где bottleneck
- чем платим за улучшение seam quality

---

**Что сейчас происходит**

По последнему run:
- seam quality хорошая, но не лучшая из всех run’ов
- `lowfreq`, `overcorrection`, `confidence_alignment`, `gain_abs_log_mean`, `detail_abs_mean` стали лучше
- но `confidence_mean` почти не сдвигается
- а `boundary_ciede2000_16` и `boundary_mae_16` оказались хуже, чем в лучшем прошлом run

Это означает:
- мы уже довольно хорошо научились делать модель аккуратнее
- но текущая попытка задавить `confidence_mean` через loss **не даёт сильного эффекта**
- при этом начинает стоить нам seam-quality

Самый важный практический вывод:
- мы **не упираемся только в датасет**
- и уже **не упираемся только в силу loss weight**
- мы начинаем упираться в то, **как сама confidence/gating часть сформулирована и что она вообще учит**

---

**Почему score всё ещё далёк от 95**

Потому что интегральная оценка наказывает не только за seam error, но и за risk profile.

Сейчас у нас:
- seam correction уже сильная
- часть risk metrics уже хорошая
- но `confidence_mean` остаётся высоким
- а `ΔE` всё ещё не там, где хочется

То есть мы уже не в фазе “научить модель чинить шов”.  
Мы в фазе:
- **сделать её исправления более безопасными и более управляемыми**

---

**Что делать дальше**

Я бы делал не ещё один грубый тюнинг весов, а следующий план.

**1. Вернуться к лучшему checkpoint как базе**  
Не брать последний run как абсолютный base, если его seam metrics хуже.  
Базой должен быть тот checkpoint, где лучший компромисс:
- `boundary_mae_16`
- `boundary_ciede2000_16`
- `lowfreq`
- `overcorrection`
- `confidence_alignment`
- `gain/detail`

С высокой вероятностью это один из мягких `stage6`/`stage6-soft` checkpoints, а не последний `stage7`.

Почему:
- нельзя строить “идеальную модель” от уже деградировавшей seam-quality точки

---

**2. Не усиливать дальше current confidence penalties лобово**  
Сейчас `conf_budget` уже доминирует в loss breakdown, а `confidence_mean` почти не падает.

Это значит:
- проблема не “мало веса”
- проблема “не тот механизм”

Почему:
- если терм уже дорогой, а metric не движется, дальнейшее увеличение веса обычно только ломает другие метрики

---

**3. Переформулировать confidence supervision**  
Это сейчас главный технический долг.

Нужно проверить и, скорее всего, поменять:
- как считается target для confidence
- не пытаемся ли мы одновременно учить confidence двум разным смыслам
- не слишком ли confidence привязан к внутренней эвристике, а dashboard меряет другое

Почему:
- по факту `confidence_mean` и `confidence_alignment_mae` ведут себя как частично разные задачи
- одна улучшается, другая почти нет

Практически это значит:
- нужен аудит confidence head semantics в loss
- возможно, нужно разнести:
  - confidence as “where correction is needed”
  - confidence as “how strong correction may be”
- сейчас это, вероятно, смешано

---

**4. Улучшить selection metric для best checkpoint ещё жёстче**  
Даже если train идёт нормально, мы должны выбирать epoch так, как будто нас интересует production-safe seam model.

Почему:
- именно checkpoint selection часто даёт самый дешёвый выигрыш без переписывания архитектуры

Тут стоит сильнее учитывать:
- `boundary_ciede2000_16`
- `confidence_mean`
- `lowfreq_mae`
- `overcorrection_mae`

Но осторожно:
- не так, чтобы selection начал выбирать “слишком осторожную, но слабую” модель

---

**5. Держать датасет в balanced non-uniform regime**  
То, что мы сейчас поставили в `stage7` config, ближе к правильному:
- spatial частый, но не абсолютный
- degradation есть, но не доминирует
- не `preview-debug` режим

Почему:
- текущая проблема уже не “слишком uniform dataset”
- экстремально усиливать corruptions дальше было бы ошибкой

---

**6. Отдельно проверять inference artifacts, а не только strip metrics**  
Потому что твои реальные визуальные наблюдения:
- dark narrow seam line
- slightly overbright wider band
- subtle glare-like color artifacts

могут не полностью отражаться в текущих strip metrics.

Что делать:
- собрать небольшой fixed visual bench
- одинаковые 20-30 трудных cases
- прогонять их на каждом candidate checkpoint
- смотреть:
  - left seam
  - right seam
  - halo
  - color flare
  - overbright band

Почему:
- иначе можно оптимизировать dashboard и всё ещё иметь неприятные inference artifacts

---

**Практический следующий план**

1. Выбрать лучший общий checkpoint из уже полученных runs.  
2. Не запускать ещё один blind stage с просто новыми loss weights.  
3. Аудировать confidence branch/loss semantics.  
4. Уточнить `_quality(...)` под реальные production priorities.  
5. После этого делать уже следующий fine-tune на 3-5 эпох от лучшего checkpoint.  
6. Параллельно вести fixed visual bench, а не только tensorboard/dashboard.

---

**Коротко**

Сейчас мы уже:
- хорошо научили модель исправлять seam
- заметно улучшили часть risk metrics
- исправили много проблем в corruption engine

Но до “идеальной модели” мешает уже не один коэффициент, а **несоответствие между тем, чему учится confidence/gating часть, и тем, что мы реально считаем безопасным качеством**.

Следующий правильный шаг:
- не просто ещё сильнее крутить веса
- а переделать именно **confidence supervision + checkpoint selection logic**, сохранив текущий balanced dataset.

Если хочешь, следующим сообщением я могу превратить это в очень конкретный технический TODO:
- какие именно функции и формулы я бы менял
- в каком порядке
- и что даст каждая правка.