# 🔥 Дообучение Толяна (Fine-Tuning `saiga-llama3:8b`)

В этой папке лежит всё, чтобы бесплатно дообучить `saiga-llama3:8b` в Google Colab (на бесплатной видеокарте Tesla T4 16 ГБ) под юмор, сленг и мемы вашей компании.

## Что внутри:
* [`dataset.jsonl`](./dataset.jsonl) — готовый датасет из 53 одноходовых и многоходовых диалогов по всем вашим видосам и анкетам из `bratva/` (Ярик и космос/орбита/душ/Винтед/привязь, Макс без света в Доте, Никитос и Радик с `trevixe`, Вадим в Словакии, Юрчик, экскурсия по Кривому Рогу в КС).
* [`generate_dataset.py`](./generate_dataset.py) — скрипт, в который можно в любой момент дописать новые шутки и пересобрать `dataset.jsonl` командой `python finetune/generate_dataset.py`.
* [`train_tolyan_colab.ipynb`](./train_tolyan_colab.ipynb) — готовый ноутбук для Google Colab.
* [`Modelfile`](./Modelfile) — конфиг для сборки полученной `.gguf` модели в Ollama.

---

## Как обучить за 15 минут (бесплатно):

1. Зайди на [colab.research.google.com](https://colab.research.google.com/) и выбери **Файл → Загрузить блокнот** — выбери [`train_tolyan_colab.ipynb`](./train_tolyan_colab.ipynb).
2. В верхнем меню проверь, что стоит GPU: **Среда выполнения → Сменить среду выполнения → T4 GPU**.
3. Слева нажми на иконку папки 📁 и перетащи туда [`dataset.jsonl`](./dataset.jsonl).
4. Нажми **Среда выполнения → Выполнить все (Run all)**.
   * *Совет для GTX 1650 (4 ГБ VRAM):* В 6-м шаге ноутбука можешь поставить `QUANT_METHOD = "q3_k_m"` (весит ~3.8 ГБ и целиком помещается в 4 ГБ видеопамяти ноутбука) или оставить `"q4_k_m"` (~4.7 ГБ, как стандартная Сайга).
5. Когда скрипт сохранит `tolyan-saiga-llama3-8b.gguf` на твой Google Диск, скачай его на ноутбук с Linux в папку `finetune/` и выполни:
   ```bash
   cd finetune
   ollama create tolyan:8b -f Modelfile
   ```
6. Запусти бота со своей собственной моделью:
   ```bash
   TOLYAN_MODEL=tolyan:8b ./run.sh
   ```
   (или пропиши `"tolyan:8b"` по умолчанию в `config.py`).
