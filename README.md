# Sentiment Reader — Game Review

Lightweight web app + API that predicts Positive/Negative sentiment for game reviews using a fine-tuned RoBERTa model.

## Repo structure
sentiment-app/
├── app_fastapi.py
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── static/
│ └── index.html
├── __pycache__/
│ └── app_fastapi.cpython-313.pyc
├── roberta/ # model files (tracked by LFS)
│ ├── config.json
│ ├── merges.txt
│ ├── model.safetensors
│ ├── special_tokens_map.json
│ ├── tokenizer_config.json
│ ├── tokenizer.json
│ └── vocab.json
├── .gitattributes
├── .gitignore
└── README.md