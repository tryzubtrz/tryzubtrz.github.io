# Tryzub Trade — деплой на хостинг (VPS)

Повний бот: дашборд, Telegram, стратегії, ML, single-file режим.

## Що потрібно

- VPS / хостинг з **Linux** (Ubuntu 22.04+) або Windows Server
- Python **3.11+**
- Відкритий порт **8080** (або інший у `.env`)
- Доступ до Bybit Testnet і Telegram API (не з US IP для Bybit)

> GitHub Pages / звичайний статичний хостинг **не підійде** — це Python-процес, потрібен VPS.

## Швидкий старт після розпакування ZIP

```bash
unzip tryzub-trade-bot-full.zip
cd tryzub-trade-bot

# 1) Створити .env з шаблону
cp .env.example .env
nano .env   # встав BYBIT_*, TELEGRAM_*, пароль дашборду

# 2) Встановити залежності
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3) (Опційно) інтерактивний setup
python setup.py

# 4а) Повна платформа (меню, dashboard, telegram, ML)
python main.py

# 4б) АБО один файл (простіший варіант)
FORCE_HTTPS=false python tryzub_trade_bot.py
```

Дашборд: `http://YOUR_SERVER_IP:8080`  
(логін за замовчуванням `admin` / пароль з `.env` або той, що задав `setup.py`)

## React UI (опційно)

```bash
cd dashboard/frontend
npm install
npm run dev    # http://localhost:5173 → проксі на :8080
```

## Автозапуск

- Linux: `bash scripts/install_autostart_linux.sh`
- Windows: `powershell -File scripts/install_autostart_windows.ps1`

## Важливо

1. `BYBIT_TESTNET=true` — спочатку тільки Testnet
2. Папку `data/experience/` **ніколи не видаляй** — там пам’ять помилок бота
3. Не заливай `.env` у публічний репозиторій
4. Для 24/7 потрібен VPS, який завжди увімкнений
