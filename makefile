.PHONY: install train

install:
    pip install -r requirements.txt

train:
    python main.py
