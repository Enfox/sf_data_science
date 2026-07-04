# <center> Перенос стиля на мобильном устройстве
## <center> Computer Vision проект: Arbitrary Style Transfer (AdaIN) с_deploy на Android через TFLite

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4-EE4C2C?logo=pytorch&logoColor=white)
![TensorFlow Lite](https://img.shields.io/badge/TFLite-2.14-FF6F00?logo=tensorflow&logoColor=white)
![Kotlin](https://img.shields.io/badge/Kotlin-1.9-7F52FF?logo=kotlin&logoColor=white)
![CameraX](https://img.shields.io/badge/Android-CameraX-3DDC84?logo=android&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-Supported-76B900?logo=nvidia&logoColor=white)
![Task](https://img.shields.io/badge/Task-Style%20Transfer-purple)

## Оглавление
[1. Описание проекта](#описание-проекта)  
[2. Какой кейс решаем?](#какой-кейс-решаем)  
[3. Краткая информация о данных](#информация-о-данных)  
[4. Используемые технологии](#используемые-технологии)  
[5. Этапы работы над проектом](#этапы-работы-над-проектом)  
[6. Архитектура решения](#архитектура-решения)  
[7. Метрики качества](#метрики-качества)  
[8. Результаты](#результаты)  
[9. Выводы](#выводы)  

---

### Описание проекта

**Перенос художественного стиля** (neural style transfer) — это задача компьютерного зрения, в которой нейросеть берёт Content-изображение и Style-изображение (картину) и генерирует новое изображение, сохраняющее контент, но выполненное в заданном художественном стиле.

В рамках данной работы решается задача **arbitrary style transfer в реальном времени на мобильном устройстве**: на вход подаётся кадр с камеры телефона и выбранная картина, на выходе — стилизованное изображение, генерируемое нейросетью прямо на устройстве.

Ключевая особенность проекта — полная цепочка от обучения собственной модели до развёртывания в Android-приложении. Модель основана на архитектуре **AdaIN** (Adaptive Instance Normalization) и не использует готовую модель из репозитория magenta — декодер обучен с нуля.

:arrow_up:[к оглавлению](#оглавление)

---

### Какой кейс решаем?

Задача формализуется как **arbitrary style transfer**: одна обученная модель должна уметь переносить **любой** стиль на **любое** изображение без дообучения.

**Ключевые вызовы задачи:**
- **Real-time на мобилке**: модель должна работать достаточно быстро на Android-устройстве с TFLite
- **Arbitrary style**: нельзя жёстко зашить один стиль — модель должна принимать любое style-изображение
- **Баланс контента и стиля**: нужно сохранить узнаваемость объектов при сильном стилевом преобразовании
- **Конвертация PyTorch → TFLite**: прямая конвертация через ONNX ломает AdaIN-операции, потребовалась архитектурная адаптация
- **Отличие от magenta**: задание требует, чтобы результат визуально отличался от готового примера magenta

Решение этих проблем достигается через архитектуру **AdaIN** (VGG19 encoder + trainable decoder), обучение на 8000 изображениях (COCO + WikiArt) с style_weight=3.0, и развёртывание через раздельный экспорт encoder/decoder в TFLite.

:arrow_up:[к оглавлению](#оглавлению)

---

### Информация о данных

| Датасет | Назначение | Изображений | Источник |
|---|---|---|---|
| **COCO val2017** | Content-изображения | 5,000 | [cocodataset.org](http://images.cocodataset.org/zips/val2017.zip) |
| **WikiArt** | Style-изображения | 3,000 | [HuggingFace: Artificio/WikiArt](https://huggingface.co/datasets/Artificio/WikiArt) |
| **Magenta test images** | Тестирование и валидация | 4 content + 11 styles | [github.com/magenta](https://github.com/magenta/magenta) |

**Структура данных:**

```text
data/
├── coco_content/val2017/      # 5000 content-изображений COCO
└── wikiart_styles/            # 3000 картин WikiArt

test_images/
├── magenta_content/           # 4 content из репозитория magenta
├── magenta_styles/            # 11 style из репозитория magenta
├── magenta_reference/         # 15 reference результатов magenta
└── results/                   # 44 стилизованных результата + grids
```

**Особенности подготовки данных:**
- COCO используется как content-датасет (разнообразные сцены: люди, животные, объекты, города)
- WikiArt предоставляет разнообразные художественные стили (импрессионизм, кубизм, абстракция, укиё-э)
- Тестовые изображения magenta используются для сравнения качества модели с reference-примерами
- Изображения ресайзятся до 256×256 для обучения и 512×512 для инференса

:arrow_up:[к оглавлению](#оглавлению)

---

### Используемые технологии

| Категория | Инструменты |
|---|---|
| **Язык** | Python 3.11 |
| **DL Framework** | PyTorch 2.4 + CUDA 12.1 |
| **Модель** | AdaIN (VGG19 encoder + Decoder) |
| **Экспорт** | Keras → TFLite (float16), ONNX |
| **Mobile ML** | TensorFlow Lite 2.14 |
| **Android** | Kotlin, CameraX 1.3, RecyclerView |
| **Видео** | OpenCV, FFmpeg |
| **Данные** | HuggingFace Hub, COCO, WikiArt |
| **Ускорение** | RTX 4070 Ti (12 GB VRAM) |

:arrow_up:[к оглавлению](#оглавлению)

---

### Этапы работы над проектом

1. **Подготовка окружения** — создание venv, установка PyTorch+CUDA, загрузка VGG19 pretrained weights.

2. **Загрузка датасетов** — COCO val2017 (5K изображений), WikiArt с HuggingFace (3K картин), тестовые изображения из репозитория magenta.

3. **Проектирование архитектуры** — реализация VGG19 encoder (frozen, до relu4_1), AdaIN layer (без параметров), и Decoder (обучаемая зеркальная часть VGG, 10.58M параметров).

4. **Обучение модели** — 40 эпох, batch_size=8, CosineAnnealingLR, style_weight=3.0 на RTX 4070 Ti. Loss уменьшен с 45 до 11.5.

5. **Валидация качества** — инференс на 4×11=44 парах из magenta тестов, расчёт content cosine similarity (0.881), генерация comparison grids.

6. **Экспорт модели** — конвертация PyTorch → Keras (с ZeroPadding для Android-совместимости) → TFLite float16. Декодер: 21 MB, энкодер: 21 MB. Max diff vs PyTorch: 0.000067.

7. **Разработка Android-приложения** — CameraX для захвата фото, TFLite для инференса, горизонтальный RecyclerView для выбора стилей, сохранение в галерею.

8. **Сборка APK** — установка JDK 17 + Android SDK + Gradle, сборка через командную строку без Android Studio.

9. **Демонстрационное видео** — программная генерация прототип-видео (8 content×style пар, 24.5 сек), инструкции для записи мобильного видео.

10. **Подготовка GitHub-репозитория** — README, .gitignore с LFS для моделей, setup-скрипт.

:arrow_up:[к оглавлению](#оглавлению)

---

### Архитектура решения

#### AdaIN Style Transfer

```text
Content Image          Style Image
  [1, 3, 256, 256]      [1, 3, 256, 256]
        ↓                      ↓
┌─────────────────────────────────────────┐
│  VGG19 Encoder (frozen, ImageNet)       │
│  слои до relu4_1                        │
│  извлекает признаки                     │
└─────────────────────────────────────────┘
        ↓                      ↓
  Content Features     Style Features
  [1, 512, 32, 32]     [1, 512, 32, 32]
        ↓                      ↓
┌─────────────────────────────────────────┐
│  AdaIN Layer (без параметров)           │
│  t = σ_style · (f_content − μ_content)  │
│      / σ_content + μ_style              │
└─────────────────────────────────────────┘
                 ↓
         AdaIN Features [1, 512, 32, 32]
                 ↓
┌─────────────────────────────────────────┐
│  Decoder (обучаемый, 10.58M params)     │
│  4 conv-blocks + 3 upsampling           │
│  ReflectionPad2d + Conv2d + ReLU        │
└─────────────────────────────────────────┘
                 ↓
        Stylized Image [1, 3, 256, 256]
```

#### Конвейер для Android

```text
┌─────────────────────────────────────────────┐
│              Android App                     │
│                                              │
│  CameraX ImageCapture                        │
│         ↓                                    │
│  encoder_relu4_1.tflite (21 MB)             │
│         ↓ [1, 32, 32, 512]                  │
│  AdaIN (Kotlin, mean/std per channel)       │
│    + precomputed style vectors (11 стилей)  │
│         ↓ [1, 32, 32, 512]                  │
│  decoder.tflite (21 MB, float16)            │
│         ↓ [1, 256, 256, 3]                  │
│  ImageView → Save to Gallery                │
└─────────────────────────────────────────────┘
```

#### Гиперпараметры обучения

| Параметр | Значение |
|---|---|
| Модель | AdaIN (VGG19 + Decoder) |
| Предобученные веса | VGG19 ImageNet (frozen) |
| Входное разрешение | 256 × 256 (train), 512 × 512 (inference) |
| Trainable parameters | 10.58M (decoder only) |
| Batch size | 8 |
| Эпохи | 40 |
| Оптимизатор | Adam |
| Learning rate | 1e-4 (CosineAnnealingLR) |
| Weight decay | 1e-4 |
| Content weight | 1.0 |
| Style weight | 3.0 |
| Style layers | relu1_1, relu2_1, relu3_1, relu4_1 |
| Content layer | relu4_1 |

:arrow_up:[к оглавлению](#оглавлению)

---

### Метрики качества

- **Content cosine similarity** — косинусное сходство между VGG-признаками исходного контента и стилизованного результата. Чем выше — тем лучше сохранён контент.

- **Color style distance** — расстояние между цветовыми статистиками (mean per channel) результата и style-изображения. Чем ниже — тем точнее перенесён стиль.

- **Training loss** — сумма content loss (MSE между VGG-признаками результата и AdaIN-целью) и style loss (MSE mean/std на слоях relu1_1–relu4_1).

- **TFLite accuracy** — максимальная разница между выходом PyTorch и TFLite-модели на одинаковых входах.

:arrow_up:[к оглавлению](#оглавлению)

---

### Результаты

После 40 эпох обучения loss стабильно снижался:

| Эпоха | Content Loss | Style Loss | Total Loss |
|---|---:|---:|---:|
| 1 | 28.1 | 4.2 | 40.7 |
| 10 | 14.2 | 1.1 | 17.5 |
| 20 | 11.5 | 0.8 | 13.9 |
| 30 | 9.9 | 0.7 | 12.0 |
| 40 | **9.4** | **0.7** | **11.5** |

**Итоговые метрики:**

| Метрика | Значение |
|---|---:|
| Content cosine similarity (VGG relu4_1) | **0.881** |
| Avg color distance to style | 0.086 |
| PyTorch inference (GPU, 512×512) | **25.3 ms / 39.4 FPS** |
| TFLite decoder (CPU, float16) | 828 ms |
| TFLite vs PyTorch max diff | **0.000067** |
| Размер decoder.tflite | 21.2 MB |
| Размер APK | 75 MB |

**Бенчмарки на test images из magenta:**

| Контент | Стиль | Content cos sim | Color distance |
|---|---|---:|---:|
| Eiffel Tower | Camille Mauclair | 0.907 | 0.233 |
| Golden Gate | bricks | 0.873 | 0.025 |
| Statue of Liberty | black_zigzag | 0.859 | 0.063 |
| Colva Beach | Camille Mauclair | 0.915 | 0.252 |

**Визуализации:**

#### Скриншот приложения

![Скриншот Android-приложения](video/screenshot.jpg)

#### Видеодемонстрация

▶️ [Видеопример работы приложения и прототипа](video/prototype.mp4)

#### Примеры инференса (модель после 40 эпох обучения)

Каждый триплет: **Content | Style | Stylized Result**

| Content | Style | Result |
|---|---|---|
| ![content](test_images/magenta_content/eiffel_tower.jpg) | ![style](test_images/magenta_styles/Camille_Mauclair.jpg) | ![result](test_images/results/eiffel_tower_x_Camille_Mauclair.jpg) |
| ![content](test_images/magenta_content/eiffel_tower.jpg) | ![style](test_images/magenta_styles/bricks_sq.jpg) | ![result](test_images/results/eiffel_tower_x_bricks_sq.jpg) |
| ![content](test_images/magenta_content/golden_gate_sq.jpg) | ![style](test_images/magenta_styles/La_forma.jpg) | ![result](test_images/results/golden_gate_sq_x_La_forma.jpg) |
| ![content](test_images/magenta_content/statue_of_liberty_sq.jpg) | ![style](test_images/magenta_styles/red_texture_sq.jpg) | ![result](test_images/results/statue_of_liberty_sq_x_red_texture_sq.jpg) |

#### Showcase (6 пар в одном изображении)

![Style Transfer Showcase](test_images/results/readme_showcase.jpg)

- 🎯 Всего сгенерировано **44 стилизованных результата** (4 content × 11 styles)
- 📊 **44 comparison grids** (триплеты Content | Style | Result)

:arrow_up:[к оглавлению](#оглавлению)

---

### Выводы

В рамках проекта успешно реализован полный pipeline **arbitrary style transfer от обучения до Android-приложения**:

1. **Модель AdaIN обучена с нуля.** Decoder (10.58M параметров) обучен на COCO + WikiArt за 40 эпох. Content cosine similarity достигла **0.881**, что подтверждает сохранение контента при стилизации.

2. **Style transfer работает в реальном времени.** На GPU (RTX 4070 Ti) инференс занимает **25 мс (39 FPS)**. На Android через TFLite — около 1 секунды на CPU, что приемлемо для пофреймовой обработки фотографий.

3. **Решена проблема конвертации PyTorch → TFLite.** Прямой путь через ONNX→TF ломал AdaIN-операции (var/mean reshape). Решение: раздельный экспорт decoder и encoder через Keras с ZeroPadding (вместо REFLECT для Android-совместимости), precomputation style vectors.

4. **Результат визуально отличается от magenta.** Модель обучена на собственных данных с другими гиперпараметрами (style_weight=3.0, 40 эпох), что даёт отличный от magenta визуальный результат при сохранении качества.

5. **Android-приложение работает.** CameraX захватывает фото, пользователь выбирает стиль из горизонтальной ленты (11 стилей), кнопка "Apply Style" запускает инференс, результат можно сохранить в галерею.

---

Если информация по этому проекту покажется вам интересной или полезной, то я буду очень благодарен, если отметите репозиторий и профиль ⭐️⭐️⭐️

:arrow_up:[к оглавлению](#оглавлению)
