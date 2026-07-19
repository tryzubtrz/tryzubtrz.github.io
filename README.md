# Tryzub Trade — AI Trading Platform

Автоматизована торгова платформа для **Bybit V5** з ML-моделями (XGBoost + LSTM), ансамблем стратегій, Telegram-сповіщеннями, HTTPS-дашбордом і щоденним самооновленням.

> За замовчуванням працює на **Bybit Testnet**. Ключі API ніколи не хардкодяться — лише `.env`.

## Системні вимоги

| Компонент | Мінімум |
|-----------|---------|
| OS | Linux / macOS / Windows |
| Python | **3.11+** |
| RAM | 4 GB (8 GB+ для комфортного навчання) |
| Disk | 2 GB вільно |
| GPU | Опційно (CUDA) — прискорює LSTM |
| Мережа | Доступ до `api-testnet.bybit.com`, Telegram API |

## Швидкий старт

```bash
# 1. Клонувати / відкрити проєкт
cd tryzubtrz.github.io   # або ваш шлях

# 2. Інтерактивне встановлення
python3 setup.py

# 3. Активувати venv
source .venv/bin/activate

# 4. Запуск
python main.py
```

Дашборд: **https://localhost:8080** (self-signed сертифікат — підтвердіть виняток у браузері).

## Як отримати Bybit API ключі (Testnet)

1. Зареєструйтесь на [Bybit Testnet](https://testnet.bybit.com/).
2. Відкрийте **Account → API Management**.
3. Створіть ключ з правами **Read + Trade** (без Withdraw).
4. Увімкніть **IP whitelist** і додайте публічний IP цієї машини.
5. Скопіюйте **API Key** і **API Secret** у `setup.py` / `.env`.

Отримати тестові USDT можна через Testnet faucet у кабінеті Bybit.

## Як створити Telegram-бота

1. У Telegram знайдіть [@BotFather](https://t.me/BotFather).
2. Команда `/newbot` → ім’я → username.
3. Скопіюйте **Bot Token**.
4. Дізнайтесь свій **Chat ID** через [@userinfobot](https://t.me/userinfobot) або `@getidsbot`.
5. Напишіть своєму боту `/start`, щоб він міг слати повідомлення.
6. Вставте token і chat id під час `setup.py`.

## Запуск

| Команда | Опис |
|---------|------|
| `python main.py` | Повний запуск: двигун + HTTPS дашборд `:8080` |
| `python main.py --once` | Один цикл скану ринку / сигналів |
| `python main.py --train` | Перенавчання ML моделей |
| `python main.py --daily-update` | Повний нічний цикл (03:00 логіка) |
| `python main.py --watchdog` | Watchdog з автоперезапуском |
| `python main.py --no-dashboard` | Лише торговий двигун |
| `python setup.py` | Встановлення / переналаштування |

### Щоденний новий «мозок» + продовження після вимкнення

- Щоночі (~03:00) моделі перенавчаються → **новий brain** у `data/models/`
- Старі backup’и чистяться (`MODEL_KEEP_VERSIONS=1` = лишити лише вчорашній)
- Угоди, ризик, сигнали лежать у `data/trading.db` — після рестарту бот **продовжує**, а не вчиться з нуля
- При старті шле в Telegram: «продовжив роботу» + версія brain

### Автозапуск після увімкнення ПК

**Windows** (один раз у PowerShell з папки проєкту):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_autostart_windows.ps1
```

Створює задачу Task Scheduler `TryzubTradeBot` (старт при логіні через `scripts\start_bot.bat`).

**Linux:**

```bash
bash scripts/install_autostart_linux.sh
```

Активація середовища:

```bash
source .venv/bin/activate          # Linux/macOS
.venv\Scripts\activate             # Windows
```

## Архітектура

```
main.py                 # оркестратор
config.py               # налаштування з .env
setup.py                # інсталятор
core/                   # Bybit, DB, risk, positions, orders, anomalies
strategies/             # trend, mean-reversion, breakout, scalping, ensemble
ml/                     # features, LSTM, XGBoost, GA, A/B, shadow
indicators/             # технічні індикатори
dashboard/              # FastAPI + static UI (HTTPS)
telegram_bot/           # сповіщення
security/               # Fernet .env, bcrypt, JWT, rate-limit, TLS
automation/             # scheduler, healthcheck, daily update, watchdog, backup
utils/                  # JSON logging
data/                   # DB, models, backups, cache
logs/                   # trading.log, ai.log, errors.log
```

## Торгівля і ризик

- Пари за замовчуванням: `BTCUSDT`, `ETHUSDT`, `SOLUSDT`
- Плече, розмір позиції, TP/SL — у `.env`
- Денний ліміт збитку (дефолт **3%**) зупиняє нові угоди
- Денні цілі **+1% / +3% / +5%** → Telegram
- Ансамбль стратегій + ML-голос (XGBoost + LSTM)

## Щоденне самооновлення (03:00)

1. Збір угод / ринку / «новинних» шоків  
2. Аналіз збиткових угод  
3. Перенавчання ML  
4. Genetic Algorithm параметрів  
5. Feature Selection  
6. A/B тест  
7. Shadow Mode  
8. Кореляційна матриця  
9. Сезонні патерни  
10. Звіт у Telegram  
11. Checkpoint моделей (останні 30 версій)

О 08:00 — підсумок попереднього дня; о 08:05 — звіт перенавчання.

## Безпека

- API ключі лише в `.env` (+ шифрована копія `.env.enc` через **Fernet**)
- Паролі дашборду — **bcrypt**
- Session JWT — **24 години**
- Rate limiting на API дашборду
- HTTPS навіть локально (self-signed у `certs/`)
- Аудит усіх дій з timestamp у БД + JSON логи
- Запити до Bybit з поточного IP (додайте його в whitelist ключа)

## Логи і бекапи

- `logs/trading.log`, `logs/ai.log`, `logs/errors.log`
- Ротація: до **100MB** / файл, ~**30 днів**
- Щоденний backup БД у `data/backups/`
- Backup моделей перед кожним retrain

## Дашборд

- URL: `https://localhost:8080`
- Логін: з `setup.py` (дефолт `admin` / ваш пароль)
- Статус P&L, позиції, сигнали, ML метрики, аудит, графік ціни
- Кнопки: скан ринку, закриття позиції

## Надійність

- **Watchdog**: краш → Telegram → рестарт через 30с; макс 5/годину
- **Healthcheck** кожні 5 хв: Bybit, баланс, позиції, сервіси

## Важливо

Це ПЗ для автоматизації торгівлі на **Testnet**. Крипторинок ризикований — використовуйте mainnet лише свідомо, змінивши `BYBIT_TESTNET=false` і ключі production, з IP whitelist.

Якщо Bybit повертає `403` (geo / rate-limit, зокрема з частини USA IP), публічні ринкові дані автоматично пробують fallback-хост, а при повній недоступності — кеш / синтетичні свічки, щоб ML і дашборд лишались працездатними. Приватні ордери потребують доступного Testnet API з вашого IP.
