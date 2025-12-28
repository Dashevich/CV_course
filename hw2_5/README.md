# Synthetic Data Augmentation for COCO-Based Dataset  
### Stable Diffusion + ControlNet → Training → Evaluation

---

## Цель
Исследовать, как синтетические данные помогают модели лучше распознавать **редкие классы** в COCO-ориентированном датасете:

1️⃣ Найти классы с малым количеством примеров  
2️⃣ Сгенерировать дополнительные данные через Stable Diffusion + ControlNet  
3️⃣ Объединить реальные и синтетические данные  
4️⃣ Обучить модель:
   - без синтетики (Baseline)
   - с синтетикой (Synthetic Augmentation)

5️⃣ Сравнить метрики на одном и том же `val` датасете

---

## Структура датасета

```

dataset/
├─ train_2017/                    # реальные данные
├─ train_2017_synthetic/          # реальные + синтетические
├─ val_2017/
└─ annotations/
├─ instances_train2017.json
├─ instances_train2017_synthetic.json
└─ instances_val2017.json

```

---

## Генерация синтетики (Stable Diffusion + ControlNet)

Для увеличения редких классов использовался Stable Diffusion в сочетании с ControlNet  

Общая идея:
- берём редкие объекты
- извлекаем структуру (границы / поза / глубина)
- генерируем новые изображения в «датасет-стиле»
- добавляем их в тренировочную выборку

Синтетика используется **только для редких классов**, чтобы не испортить баланс.

---

## Обучение

Использовалась модель детекции объектов:
- Faster R-CNN ResNet50 FPN (PyTorch)
- одинаковые гиперпараметры
- одинаковый `val_2017`

### Эксперимент 1 — Baseline
Обучение только на реальных данных:

```

train_2017 + instances_train2017.json

```

### Эксперимент 2 — Synthetic Augmentation
Обучение на:
```

train_2017_synthetic + instances_train2017_synthetic.json

```

---

## Оценка
Метрики → стандартные COCO:

- mAP@[0.50:0.95]
- AP50
- AP75
- AR (Recall)
- Per-object scale metrics (small/medium/large)

---

# ✅ Результаты

## Baseline (без синтетики)
```

Average Precision  (AP) @[ IoU=0.50:0.95 ] = 0.000
Average Precision  (AP) @[ IoU=0.50      ] = 0.001
Average Precision  (AP) @[ IoU=0.75      ] = 0.000

Average Precision  (AP) @[ small  ] = 0.000
Average Precision  (AP) @[ medium ] = 0.000
Average Precision  (AP) @[ large  ] = 0.000

Average Recall (AR) maxDets=  1 = 0.002
Average Recall (AR) maxDets= 10 = 0.010
Average Recall (AR) maxDets=100 = 0.027

```

Вывод: модель практически **ничего не видит**, редкие классы не учатся.

---

## Synthetic Augmentation (реальные + синтетика)

После добавления сгенерированных данных для редких классов модель значительно улучшила качество:

```

Average Precision  (AP) @[ IoU=0.50:0.95 ] = 0.018
Average Precision  (AP) @[ IoU=0.50      ] = 0.061
Average Precision  (AP) @[ IoU=0.75      ] = 0.011

Average Precision  (AP) @[ small  ] = 0.010
Average Precision  (AP) @[ medium ] = 0.024
Average Precision  (AP) @[ large  ] = 0.028

Average Recall (AR) maxDets=  1 = 0.012
Average Recall (AR) maxDets= 10 = 0.057
Average Recall (AR) maxDets=100 = 0.109

```

---

## 📌 Таблица Ablation Study

| Эксперимент | Train Data | mAP@[0.5:0.95] | AP50 | AP75 | AR100 |
|-----------|-----------|----------------|------|------|-------|
| Baseline | только реальные | 0.000 | 0.001 | 0.000 | 0.027 |
| + Synthetic | реальные + SD ControlNet | **0.018** | **0.061** | **0.011** | **0.109** |

✔️ заметен рост всех метрик  
✔️ сильнее всего вырос `AP50` и `Recall`  
✔️ редкие классы начали детектироваться

---

## Выводы

- Stable Diffusion + ControlNet помогает **решить проблему редких классов**
- Метрики значительно выросли
- Особенно улучшился `Recall`, значит модель перестала «пропускать» объекты
- AP на высоких IoU пока небольшой → требуется дальнейшее улучшение локализации

---
