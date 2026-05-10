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
├── elo_bootstrap.py   # разовая загрузка рейтингов с clubelo.com
├── elo_updater.py     # ежедневный джоб обновления Elo по результатам
├── probability.py     # утилиты: implied_prob, remove_overround, consensus_probabilities
├── detectors.py       # 6 детекторов: spread, drift, synchronized, model_gap, exotic_spread, sharp_move
├── notifier.py        # отправка алертов в Telegram (использует общий Bot)
├── bot.py             # команды бота: /stats /recent /thresholds /elo
└── pipeline.py        # один цикл: fetch → фильтр → детект → save → alert
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

## Архитектура анализа

### Вероятности вместо коэффициентов

Все детекторы работают с **маржа-свободными вероятностями** (margin-free
probabilities), а не сырыми коэффициентами. Это позволяет честно сравнивать
котировки разных букмекеров с разным overround.

```
implied_prob = 1 / odds              # сырая вероятность
margin_free  = prob / sum(all_probs) # нормализация (remove_overround)
```

Реализация в `src/probability.py`: `implied_probability`, `remove_overround`,
`probabilities_from_match`.

### Веса букмекеров

Sharp-конторы (Pinnacle, Betfair, SBObet, Matchbook) двигают рынок первыми
и отражают реальный поток денег. Они получают вес `sharp_weight=1.0`.
Остальные (soft/followers) — `default_weight=0.4`.

`consensus_probabilities` строит **взвешенную медиану** по всем букмекерам.
Взвешенная медиана — не взвешенное среднее: одиночный выброс (устаревшая
котировка) не может сдвинуть консенсус дальше своего значения.

`detect_synchronized` срабатывает только если среди синхронно двинувшихся
контор есть хотя бы одна sharp — чисто follower-движение игнорируется.

### Временны́е корзины (time buckets)

Чем ближе к матчу, тем ниже пороги детекторов (больше чувствительность).
Настраивается через `time_buckets_hours` и `time_bucket_multipliers` в `.env`:

| До матча | Множитель | Смысл                          |
|----------|-----------|-------------------------------|
| < 6 ч    | ×0.70     | Активный рынок — ловим больше  |
| 6–24 ч   | ×0.85     | Повышенная чуткость            |
| 24–72 ч  | ×1.00     | Базовые пороги                 |
| > 72 ч   | ×1.30     | Ранние котировки — шум выше    |

### Скользящее окно дрейфа

`detect_drift` сравнивает текущие вероятности с самым старым снимком
в окне `drift_window_minutes` (по умолчанию 120 мин), а не с первым
снимком за всю историю. Это убирает накопленный шум из далёкого прошлого.

### Динамическая доля ничьих в Elo

`fair_odds_1x2` вычисляет долю ничьих как функцию разности рейтингов:
`draw_share = max(0.18, min(0.32, 0.30 - 0.0003 × |Δelo|))`.
Чем равнее команды — тем выше вероятность ничьей.

## Тесты

Тесты находятся в `tests/`. Запускать: `pytest -v`. 61 тест, 0 сетевых
запросов — всё на синтетических данных и in-memory SQLite.

```
tests/
├── conftest.py                      # dummy env vars для Settings()
├── test_probability.py              # implied_prob, remove_overround, consensus
├── test_detectors_spread.py         # detect_spread в процентных пунктах
├── test_detectors_drift.py          # скользящее окно дрейфа
├── test_detectors_synchronized.py   # sharp-фильтр для synchronized
├── test_detectors_time_bucket.py    # временны́е корзины
├── test_probability_weights.py      # веса букмекеров, weighted median
├── test_elo_bootstrap.py            # загрузка рейтингов с clubelo.com
└── test_elo_draw_share.py           # динамическая доля ничьих
```

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

1. Команда бота `/mute <hours>` — временно отключить алерты.
2. ML-слой (XGBoost / Isolation Forest) поверх правил — когда накопится
   2+ месяца истории срабатываний с разметкой результата.
3. Персистентная дедупликация алертов (сейчас in-memory, сбрасывается при рестарте).
4. Экспорт истории снимков в CSV/Parquet для офлайн-анализа.
