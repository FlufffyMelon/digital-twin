# DigitalTwin

Цифровой двойник: дообучение LLM на собственной переписке из Telegram, чтобы
модель отвечала в моём стиле.

Первая итерация (май 2025): LoRA (r=64) поверх `unsloth/Llama-3.1-8B` в 4 бит
через [unsloth](https://github.com/unslothai/unsloth) + TRL `SFTTrainer`.
Было 11 прогонов на разных вариантах датасета, `eval_loss` от 0.72 до 1.19
(между датасетами напрямую не сравним).

> **Приватность.** В репозитории только код. Переписки, собранные из них
> датасеты, выводы ноутбуков и веса моделей в git не попадают никогда:
> это закреплено `.gitignore` (белый список), git-фильтром для ноутбуков
> и pre-commit хуком.

## Структура

```
├── data/                     переписки и датасеты (не в git, кроме data/README.md)
├── notebooks/
│   ├── data_processing.ipynb   разбор экспорта Telegram и сборка всех вариантов датасета
│   ├── llm_processing.ipynb    попытка отбирать ответы на вопросы с помощью Llama-3.1-8B-Instruct
│   ├── ollama_prototype.ipynb  ранний прототип: разметка диалога в пары вопрос/ответ через Ollama
│   ├── finetune.ipynb          черновик обучения в ноутбуке
│   └── inference.ipynb         генерация ответов адаптером, интерактивный чат
├── scripts/
│   ├── create_dataset.py       интерактивная ручная разметка ответов на вопросы
│   └── nbstrip.py              очистка выводов ноутбуков (git-фильтр и проверка в хуке)
├── slurm/
│   ├── train.sh                обучение (`train.py`)
│   └── jupyter.sh              Jupyter Lab на GPU-узле
├── train.py                  обучение LoRA (основной скрипт)
├── logs/                     логи slurm (не в git)
├── outputs/, wandb/          чекпоинты и логи обучения — только на кластере
├── pyproject.toml, uv.lock   окружение
└── mutagen.yml               синхронизация с кластером
```

Что лежит в `data/` и как каждый файл получен — в [`data/README.md`](data/README.md).
Скрипты и `sbatch` запускаются из корня репозитория, ноутбуки — из `notebooks/`
(пути к данным в них вида `../data/...`).

## Окружение

Окружение собирается через [uv](https://docs.astral.sh/uv/) и только на
кластере (Linux x86_64, CUDA 13): torch, torchvision и xformers берутся из
индекса PyTorch `cu130`, точные версии зафиксированы в `uv.lock`.

```bash
uv sync --frozen
```

| | v1 (softcluster, май 2025) | v2 (mipt) |
|---|---|---|
| CUDA / torch | 12.x / 2.5.1 | 13.0 / 2.14.1 |
| unsloth | 2025.4.7 | 2026.11.3 |
| transformers / trl | 4.51.3 / 0.15.2 | 5.17.0 / 1.15.0 |
| peft / bitsandbytes | 0.15.2 / 0.45.5 | 0.21.2 / 0.50.2 |

Верхние границы torch, transformers, trl и datasets задаёт unsloth.

## Первичная настройка после клонирования

Без этих команд защита от утечки выводов ноутбуков не работает:

```bash
git config core.hooksPath .githooks
git config filter.nbstrip.clean "python3 scripts/nbstrip.py"
git config filter.nbstrip.smudge cat
git config filter.nbstrip.required true
```

## Работа с кластером

Код редактируется локально и синхронизируется на кластер через
[mutagen](https://mutagen.io): код и json уезжают туда, веса и логи
обучения остаются там.

```bash
mutagen project start      # запустить синхронизацию
mutagen sync list digitaltwin
mutagen project terminate  # остановить
```

Кластер — mipt, GPU-узлы:

- `node208-24`: A100 80GB, раздел `AMD7763-A100-CCP` — по умолчанию в `slurm/train.sh`;
- `node211-22`: RTX 4500 Ada 24GB, раздел `AMD7V12-RTX4500-CCP` — по умолчанию в `slurm/jupyter.sh`.

```bash
sbatch slurm/train.sh                              # обучение на A100
sbatch -p AMD7V12-RTX4500-CCP slurm/train.sh       # обучение на Ada
sbatch slurm/jupyter.sh                            # Jupyter Lab
```

Лучший чекпоинт v1 (`run_20250506_194631/checkpoint-2300`, eval_loss 0.720)
лежит на mipt в `outputs/v1_run_20250506_194631/` как точка сравнения.
