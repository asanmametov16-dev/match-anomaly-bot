# CLAUDE.md

Этот файл читается Claude Code автоматически. Здесь собран контекст проекта,
правила и подсказки, чтобы каждая сессия начиналась с понимания, что и как
устроено.

## Что это за проект

Match Anomaly Detector — инструмент выявления **точного рыночного сигнала**
по букмекерским коэффициентам с уведомлениями в Telegram. Фокус на
precision, а не recall: в «сигналы» эскалируются только аномалии высокого
качества, подтверждённые несколькими детекторами и CLV-калибровкой
(precision-gate, см. `classify_signal`); слабые кластеры сохраняются и
помечаются `confidence=weak`, но не шумят.

Рамка: это **аналитический сигнал, а не ставочная/инвестиционная
рекомендация** — «пока без ставок». В коде, сообщениях и комментариях
держимся этой формулировки (мягкий тон, без обещаний прибыли/убытка);
дисклеймер — в `notifier.py` (`_SIGNAL_DISCLAIMER`).

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
├── sstats_client.py   # клиент sstats.net: xG/winProb/Glicko для model_gap
├── detectors.py       # 7 детекторов: spread, drift, synchronized, model_gap, exotic_spread, sharp_move, cross_market
├── clv.py             # closing line value: оценка сигнальной ценности алертов
├── calibration.py     # CLV → множители весов детекторов (compute_score)
├── prob_calibration.py # Brier/log-loss консенсуса против реальных исходов
├── sstats_history.py  # офлайн-бэкфилл калибровки модели sstats по лигам
├── notifier.py        # отправка алертов в Telegram (использует общий Bot)
├── bot.py             # команды: /stats /accuracy /clv /calibration /modelcal /weights /recent /thresholds /elo
└── pipeline.py        # один цикл: fetch → xG-обогащение → детект → save → alert
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
- `SSTATS_API_KEY` — https://sstats.net (xG/winProb/Glicko для `model_gap`;
  без ключа детектор работает по Elo-fallback)
- `SSTATS_ENABLED` — `true` по умолчанию; `false` для аварийного отключения
  sstats-обогащения без удаления ключа

## Известные особенности и подводные камни

- **Лимит Odds API.** Аккаунт на Pro-тарифе — учёт дневной квоты убран
  полностью (нет `quota.py`, нет проверок в `pipeline.py`).
  `POLL_INTERVAL_MINUTES=15` теперь допустимо. Если тариф снова станет
  лимитированным — восстанавливать счётчик квоты заново.
- **`model_gap`: трёхуровневый fallback модели.** Приоритет (точность
  убывает): (1) sstats.net xG/winProb при наличии `SSTATS_API_KEY` и
  покрытия; (2) маржа-free консенсус sharp-контор (доступен с первого дня,
  точнее холодного Elo) — нужно ≥ `model_gap_min_sharp_books` (2) sharp-книг
  с полным 1X2; (3) Elo как последний резерв (шумит 1–2 недели). Источник
  пишется в `payload["source"]` ∈ {`sstats_xg`, `sharp_consensus`, `elo`}.
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

Снятие маржи (`remove_overround`) по умолчанию — **метод Шина**, а не
пропорциональное деление. Пропорциональный метод смещён: маржа
концентрируется на аутсайдерах (favourite-longshot bias), а деление на
сумму снимает её одинаковой долей со всех исходов → фаворит занижается,
аутсайдер и ничья завышаются. Шин подбирает долю «инсайдерских» денег z
бисекцией так, чтобы `q_i = (√(z²+4(1−z)p_i²/B) − z)/(2(1−z))`
суммировались в 1 — это точнее воспроизводит реальную структуру маржи в
1X2. Переключается `devig_method` ∈ {`shin`, `proportional`}; откат не
требует пересчёта данных.

Реализация в `src/probability.py`: `implied_probability`, `remove_overround`
(`_devig_shin` / `_devig_proportional`), `probabilities_from_match`.

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

### CLV (closing line value)

`clv.py` — единственная честная метрика прогностической ценности алертов.
Для направленных детекторов (drift↑, synchronized↓, model_gap value-side,
sharp_move) сравнивается консенсус-вероятность «ставочной стороны» в момент
алерта и в последнем снимке до старта матча. `+CLV` = рынок продолжил
двигаться в сторону прогноза.

Не-направленные детекторы (spread, exotic_spread) и кейсы без снимков пишут
sentinel-строку с NULL — `compute_pending_clv` (ежечасный джоб) идемпотентен
и не пересчитывает уже обработанные аномалии. Результат смотреть командой
`/clv`. Положительный CLV необходим, но не достаточен для прибыльности —
нужно ещё перекрыть маржу букмекера.

### Калибровка весов по CLV

`calibration.py` замыкает петлю: `compute_score` больше не суммирует чисто
ручные `DETECTOR_WEIGHTS`, а домножает их на CLV-множитель. Раз в час
`refresh_detector_weights` берёт средний CLV по детектору из `AnomalyCLV` и
считает `clamp(1 + sensitivity·mean_clv_pp, min, max)`. Детектор с числом
измерений < `clv_calibration_min_samples` (30) не калибруется (множитель 1.0,
доверяем дефолту). Множители — горячий модуль-кэш (без БД в пути детекторов),
прогреваются на старте и видны командой `/weights`. Отключается флагом
`clv_calibration_enabled=false`.

### Калибровка вероятностей по исходам (Brier)

`prob_calibration.py` отвечает на другой вопрос, чем CLV: **насколько
вообще верны наши вероятности?** Берётся маржа-free consensus закрывающей
линии (та же точка, что у CLV), нормируется и сравнивается с фактическим
1X2-исходом через многоклассовый Brier (∈[0,2]) и log-loss. Джоб
`compute_pending_calibration` (каждые 2ч) идемпотентен; матч без
результата/3-way закрытия после 96ч пишет sentinel-строку. Команда
`/calibration` показывает средний Brier, log-loss и кривую надёжности
(предсказанная p против эмпирической частоты по бинам). Ориентиры:
равномерный прогноз ⇒ Brier 0.667; информативный рынок 1X2 ≈ 0.55–0.58.
Это диагностика de-vig/консенсуса — Shin должен давать Brier ниже
пропорционального.

### Межрыночная согласованность (cross_market)

`detect_cross_market` использует **тождество**, а не модель: фора 0.0
(level ball) = Draw-No-Bet, поэтому её маржа-free вероятность по home
обязана равняться `p_home/(p_home+p_away)` из 1X2. Сравнивается консенсус
1X2 с медианой DNB по форе 0.0 (≥ `cross_market_min_books` контор);
расхождение в пп сверх `cross_market_pp_threshold` (× time-bucket) =
сигнал устаревшей линии/ошибки в одном из рынков. Ортогонален одиночным
детекторам, не-направленный (как spread/exotic). Пока **не алертится**
(нет в `ALERT_DETECTORS`, как `exotic_spread`): сохраняется и скорится,
а CLV-калибровка весов (#1) сама поднимет/опустит его вес по факту.
No-op без форы 0.0 — лучше молчать, чем шуметь.

### Precision-gate (точный сигнал)

`classify_signal(alert_hits)` решает, кластер аномалий — «точный сигнал»
или «слабое наблюдение». Сигнал требует одновременно: ≥
`signal_min_detectors` разных alert-детекторов, CLV-взвешенный
`compute_score` ≥ `signal_score_threshold`, средний CLV-множитель
сработавших детекторов ≥ `signal_min_clv_multiplier` (т.е. это не
исторически шумные детекторы), и однонаправленный консенсус ≥
`signal_min_agreement`. Пайплайн **сохраняет всё** и штампует каждой
записи `payload["signal_confidence"]` ∈ {signal, weak}; в Telegram уходят
только `signal`. `signal_gate_enabled=false` → старое поведение (алерт по
числу детекторов). `/recent` показывает 🎯signal/💤weak по каждой записи,
`/stats` — разбивку precision-gate. Cold-start безопасен: без CLV-данных множитель = 1.0,
решает счёт+согласие. Это оценка качества сигнала, не ставочный вердикт.

### Калибровка модели sstats (исторический бэкфилл)

`prob_calibration` нельзя засеять историей: ему нужны НАШИ снимки рынка,
которых для прошлых матчей нет. Поэтому `sstats_history.py` скорит
**модель sstats** (winProb из `/Games/glicko`) против факта из
`/Games/list?Ended=true` — это можно подтянуть сразу. Запуск офлайн:
`python -m scripts.backfill_sstats_history --max-games N` (идемпотентно,
rate-limit ~133/мин, НЕ в пайплайне). Таблица `SstatsModelOutcome`,
агрегат по лигам — команда `/modelcal`. Назначение: сравнение
модель-vs-рынок (`/modelcal` ↔ `/calibration`) и лиго-зависимое доверие
к `model_gap`. Brier/log-loss переиспользуются из `prob_calibration`.

**Два источника результатов.** `result_checker` тянет сыгранные матчи и
из football-data.org, и из sstats (`fetch_finished_matches_sstats` —
только `/Games/list`, без glicko, Order=-1 со стопом по окну). sstats
покрывает 200+ лиг вне football-data (MLS, Süper Lig, Eredivisie…),
которые иначе никогда не резолвились → теперь по ним считаются
CLV/Brier/outcomes. Оба источника пишут в общий `MatchResult`
(dedupe по `result_key`); работает даже без `FOOTBALL_DATA_KEY`. Импорт
в `result_checker` — локальный (рвёт цикл с `prob_calibration`).

**Лиго-зависимое доверие (#3).** `refresh_model_trust` (ежечасно + на
старте) классифицирует лиги по историческому Brier модели:
`trusted` (Brier < 0.667 − `model_trust_uniform_margin`),
`unreliable` (Brier ≥ 0.667), иначе `unknown` (или < `model_trust_min_samples`
матчей). Лига протягивается из sstats `/Games/list` (`season.league`) в
`XgPrediction.league` → в `payload` срабатывания `model_gap`
(`league`, `model_trust`). `classify_signal` **исключает** `model_gap` из
лиги `unreliable` из решения о «точном сигнале» (запись сохраняется,
`meta.dropped_untrusted_model_gap`). `unknown`/`trusted` не гейтятся —
cold-start безопасен.

## Тесты

Тесты находятся в `tests/`. Запускать: `pytest -v`. 163 теста, 0 сетевых
запросов — всё на синтетических данных, in-memory SQLite и httpx.MockTransport.

`conftest.py` нет: `Settings()` читает реальный `.env` (он gitignored, но
присутствует локально). Тесты детекторов завязаны на значения порогов из
`.env` — при их правке проверять, что тесты ещё проходят.

```
tests/
├── test_probability.py              # implied_prob, remove_overround, consensus
├── test_probability_devig.py        # метод Шина vs пропорциональный
├── test_detectors_spread.py         # detect_spread в процентных пунктах
├── test_detectors_drift.py          # скользящее окно дрейфа
├── test_detectors_synchronized.py   # sharp-фильтр для synchronized
├── test_detectors_time_bucket.py    # временны́е корзины
├── test_detectors_cross_market.py   # h2h vs фора 0.0 (DNB-тождество)
├── test_detectors_model_gap_xg.py   # model_gap: xG-путь и Elo-fallback
├── test_detectors_model_gap_sharp.py # model_gap: sharp-консенсус fallback
├── test_probability_weights.py      # веса букмекеров, weighted median
├── test_sstats_client.py            # sstats клиент (mock transport)
├── test_clv.py                      # closing line value
├── test_calibration.py              # CLV → веса детекторов
├── test_prob_calibration.py         # Brier/log-loss против исходов
├── test_signal_gate.py              # precision-gate classify_signal
├── test_sstats_history.py           # бэкфилл калибровки модели sstats
├── test_model_trust.py              # лиго-зависимое доверие модели (#3)
├── test_bot_helpers.py              # чистые хелперы bot.py
├── test_notifier_helpers.py         # gate-строка алерта
├── test_elo_bootstrap.py            # загрузка рейтингов с clubelo.com
└── test_elo_draw_share.py           # динамическая доля ничьих
```

## Git-гигиена

- `.env`, `data/`, `logs/`, `.venv/`, `__pycache__/` — в `.gitignore`.
- Коммиты осмысленными порциями, по одной задаче: «add tests for spread
  detector», не «misc changes».
- Перед коммитом убеждаемся, что `python -m py_compile src/*.py` проходит.

## Чего НЕ делать

- Не превращать сигнал в ставочную рекомендацию: никаких «ставь сюда»,
  stake-sizing, `VALID_SIGNAL`. Рамка «без ставок» и `_SIGNAL_DISCLAIMER`
  в `notifier.py` остаются в каждом алерте — не убирать.
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
