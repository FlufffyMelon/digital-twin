# data/

Personal data lives here and never goes to git: only this file is tracked
(see `.gitignore` and `.githooks/pre-commit`).

```
data/
├── result.json          Telegram Desktop export of the chat (any name; several exports of one chat can be merged)
└── dataset/             built by python3 -m digital_twin.dataset.build
    ├── train.jsonl        training examples, shuffled
    ├── val.jsonl          the last weeks of the chat, full context only
    └── meta.json          names in the system prompt, export paths, sizes
```

Exporting a chat: in Telegram Desktop open the chat → ⋮ → *Export chat history*,
uncheck all media (photos, videos, voice messages, stickers, files…), choose
*Machine-readable JSON* and save. Only `result.json` is needed.
