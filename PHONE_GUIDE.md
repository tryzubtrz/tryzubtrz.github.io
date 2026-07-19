# Tryzub Trade — гайд для телефону

Коротко: що це, що треба дати, як запустити, як бот вчиться.

---

## Що це

AI-бот для торгівлі на **Bybit Testnet**:
- сам шукає сигнали
- може відкривати/закривати угоди
- пише в Telegram
- щоночі донавчається і **не забуває помилки**
- після вимкнення ПК продовжує з збереженого стану

Код: https://github.com/tryzubtrz/tryzubtrz.github.io  
PR: https://github.com/tryzubtrz/tryzubtrz.github.io/pull/1  
Дашборд (коли запущено): https://localhost:8080

---

## Що треба підготувати (чекліст)

### 1) Bybit Testnet
Де: https://testnet.bybit.com → Account → API  

Дати боту:
- API Key  
- API Secret  

Налаштування ключа:
- права: Read + Trade (без Withdraw)
- IP whitelist = IP комп’ютера або VPS
- взяти тестові USDT у faucet

### 2) Telegram
- Токен бота: @BotFather → /newbot  
- Ваш Chat ID: @userinfobot  
- Написати боту /start  

Дати боту:
- TELEGRAM_BOT_TOKEN  
- TELEGRAM_CHAT_ID  

### 3) Комп’ютер або VPS
Ваш ПК (i7 / 2060 Ti / 16GB) — **вистачає**.

Або VPS, якщо хочете 24/7 без увімкненого ПК.

### 4) На комп’ютер встановити
1. Python 3.11+ — https://www.python.org/downloads/  
2. Git — https://git-scm.com/downloads  
3. Проєкт:
```text
git clone https://github.com/tryzubtrz/tryzubtrz.github.io.git
cd tryzubtrz.github.io
python setup.py
python main.py
```

---

## Як запустити

```text
python setup.py          ← один раз, ввести ключі
python main.py           ← щоденний запуск
```

Автозапуск Windows (після логіну):
```text
powershell -ExecutionPolicy Bypass -File scripts\install_autostart_windows.ps1
```

Корисні команди:
```text
python main.py --once           один скан
python main.py --train          донавчання зараз
python main.py --daily-update   повний нічний цикл
python main.py --watchdog       автоперезапуск при краші
```

---

## Чи не забуває він помилки?

НІ, не забуває.

- Досвід і уроки з угод → папка `data/experience/` (не стирається)
- Збитки запам’ятовуються сильніше
- Щоночі модель **донавчається** від учорашнього brain
- Видаляються лише зайві копії файлів на диску, не пам’ять

---

## Якщо ПК вимкнувся

1. Увімкнули ПК  
2. Бот стартує (вручну або автозапуск)  
3. Завантажує БД + brain + досвід  
4. Продовжує, ніби вчора  
5. У Telegram: «продовжив роботу»

Поки ПК вимкнений — торгівлі немає.

---

## Мінімум, що надіслати мені для повного старту

1. Bybit API Key (Testnet)  
2. Bybit API Secret (Testnet)  
3. Telegram Bot Token  
4. Telegram Chat ID  
5. (Опційно) доступ до VPS, якщо не з ПК  

---

## Важливо

- За замовчуванням **Testnet** (навчання на тестових грошах)
- Ключі тільки в `.env`, ніколи в коді
- Крипторинок ризикований — mainnet лише свідомо

Збережіть цей файл у Notes / Files на телефоні.
