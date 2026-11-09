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

| Файл | Что делает |
|---|---|
| `data_proccess.ipynb` | Ранний прототип: разметка диалога в пары вопрос/ответ через Ollama |
| `data_proccesing.ipynb` | Разбор экспорта Telegram (`result.json`) и сборка вариантов датасета: alpaca, по reply-цепочкам, с контекстом |
| `llm_proccesing.ipynb` | Попытка автоматически отбирать ответы на вопросы с помощью Llama-3.1-8B-Instruct |
| `create_dataset.py` | Интерактивная ручная разметка: выбор правильных ответов на вопросы собеседника |
| `finetune.ipynb` | Черновик обучения в ноутбуке |
| `train.py` | Обучение LoRA (основной скрипт) |
| `inference.ipynb` | Генерация ответов обученным адаптером, интерактивный чат |
| `run.sh`, `jupyter.sh` | slurm-скрипты обучения и Jupyter Lab |
| `scripts/nbstrip.py` | Очистка выводов ноутбуков (git-фильтр и проверка в хуке) |
| `mutagen.yml` | Синхронизация кода и данных с кластером |

Данные лежат рядом с кодом, но в git не попадают:

- `result*.json` — экспорт чата из Telegram и его разбор;
- `dataset*.json`, `conversation_dialog.json` — собранные датасеты;
- `outputs/`, `wandb/` — чекпоинты и логи, существуют только на кластере.

## Окружение

```bash
pip install -r requirements.txt
```

`requirements.txt` фиксирует версии, с которыми прошло первое обучение.

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

Обучение на кластере:

```bash
sbatch run.sh
```
