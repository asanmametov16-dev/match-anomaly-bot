# CLAUDE.md

Этот файл читается Claude Code автоматически. Здесь собран контекст проекта,
правила и подсказки, чтобы каждая сессия начиналась с понимания, что и как
устроено.

## Что это за проект

Match Anomaly Detector — детектор аномалий в букмекерских коэффициентах
с уведомлениями в Telegram. Это **аналитический инструмент**, а не сигналы
для ставок: аномалия в линии ≠ договорняк, и попытки зарабатывать на таких
сигналах статистически убыточны (букмекеры режут лимиты, аннулируют ставки).
В коде и комментариях держимся этой формулировки.

## Структура

```
src/
├── main.py            # точка входа: бот + APScheduler
├── config.py          # pydantic-settings, читает .env
├── db.py              # SQLAlchemy 2.0 модели + init_db()
├── odds_client.py     # клиент The Odds API (h2h + totals + spreads)
├── results_client.py  # клиент football-data.org для результатов
├── elo.py             # Elo-рейтинг, fair_odds_1x2, нормализация имён
├── elo_updater.py     # ежедневный джоб обновления Elo по результатам
├── detectors.py       # 5 детекторов: spread, drift, synchronized, model_gap, exotic_spread
├── notifier.py        # отправка алертов в Telegram (использует общий Bot)
├── bot.py             # команды бота: /stats /recent /thresholds /elo
└── pipeline.py        # один цикл: fetch → детект → save → alert (с дедупом)
```

## Как запускать

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                # потом заполнить ключи
python -m src.main
```

## Обязательные переменные окружения

- `ODDS_API_KEY` — https://the-odds-api.com (500 req/мес бесплатно)
- `TELEGRAM_BOT_TOKEN` — от @BotFather
- `TELEGRAM_CHAT_ID` — целевой чат (бот принимает команды только из него)

Опциональные:

- `FOOTBALL_DATA_KEY` — https://www.football-data.org/client/register
  (без него Elo не обновляется → детектор `model_gap` шумит)

## Известные особенности и подводные камни

- **Лимит Odds API.** На бесплатном тарифе 500 запросов/мес. При
  `POLL_INTERVAL_MINUTES=15` это 2880/мес — превышение. Дефолт безопаснее
  ставить `90`. Это стоит проверять при любых изменениях расписания.
- **`model_gap` шумит первые 1–2 недели**, пока Elo-рейтинги не наберут
  статистику. По умолчанию порог высокий (0.20). Если Elo не обновляется —
  совсем отключи детектор, подняв порог в `.env` до 1.0.
- **Нормализация имён команд.** Odds API и football-data.org называют
  команды по-разному ("Manchester United" vs "Manchester United FC").
  В `elo.py` есть `_normalize`, в `results_client.py` — `normalize_team_name`.
  При проблемах с матчингом смотреть туда.
- **Дедупликация алертов.** В `pipeline.py` есть in-memory словарь
  `_recently_alerted` с окном 2 часа. После рестарта бот может прислать
  дубль — это known limitation MVP.
- **Telegram-приложение и шедулер.** В `main.py` строгий порядок:
  `app.initialize() → app.start() → updater.start_polling() → scheduler.start()`.
  При изменениях в этой части ничего не упрощать без проверки на запуске.
- **SQLite write contention.** Все джобы в одном процессе и используют
  `SessionLocal` синхронно — для MVP ок. Если переходим на multi-process
  или async-БД — нужен пересмотр.

## Стиль кода

- Python 3.11+, типы через `from __future__ import annotations`.
- SQLAlchemy 2.0 ORM-стиль (`select(...)`, `session.scalar(...)`),
  не legacy `Query`-API.
- Относительные импорты внутри пакета (`from .config import settings`).
- Логирование через `logging.getLogger(__name__)`, не print.
- HTML-экранирование пользовательских данных в Telegram-сообщениях
  (`html.escape`) — это уже есть в `bot.py` и `notifier.py`, не сломать.
- Авторизация команд бота: только `TELEGRAM_CHAT_ID` (см. `_is_authorized`).
- Никаких `localStorage`/`sessionStorage` — это серверный код.

## Тесты

Тестов пока нет — это первая задача после установки. Самый понятный
кандидат для покрытия — `detectors.py` (чистая логика без сети). Целевой
стек: pytest. Тесты не должны делать сетевых запросов.

## Git-гигиена

- `.env`, `data/`, `logs/`, `.venv/`, `__pycache__/` — в `.gitignore`.
- Коммиты осмысленными порциями, по одной задаче: «add tests for spread
  detector», не «misc changes».
- Перед коммитом убеждаемся, что `python -m py_compile src/*.py` проходит.

## Чего НЕ делать

- Не делать ставки по сигналам этого бота (см. дисклеймер в `notifier.py`).
- Не убирать `_is_authorized` из команд бота — иначе любой, кто узнает
  ник бота, сможет дёргать `/stats`.
- Не добавлять детекторы, которые делают сетевые запросы внутри цикла
  пайплайна без таймаутов и обработки ошибок.
- Не коммитить `.env`. Никогда.

## Идеи для следующих задач (по приоритету)

1. Тесты для `detectors.py` на синтетических `MatchOdds`.
2. Фильтр «только матчи в ближайшие N часов» в `pipeline.py` — сэкономит
   запросы к Odds API.
3. Загрузка стартовых Elo-рейтингов с clubelo.com (опционально).
4. Команда бота `/mute <hours>` — временно отключить алерты.
5. ML-слой (XGBoost / Isolation Forest) поверх правил — когда накопится
   2+ месяца истории срабатываний с разметкой результата.
