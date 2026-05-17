"""Точка входа.

Запускает три параллельные активности:
1. Telegram-бот в режиме polling (слушает команды).
2. Периодический опрос Odds API (каждые POLL_INTERVAL_MINUTES).
3. Ежедневный апдейт Elo по результатам матчей в ELO_UPDATE_HOUR_UTC.
"""
from __future__ import annotations

import asyncio
import logging
import signal
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from telegram.error import NetworkError

from .bot import build_application
from .calibration import refresh_detector_weights
from .clv import compute_pending_clv
from .prob_calibration import compute_pending_calibration
from .sstats_history import refresh_model_trust
from .config import settings
from .db import init_db
from .elo_updater import update_elo_from_results
from .notifier import send_startup_message, set_bot
from .pipeline import run_once
from .result_checker import check_anomaly_results


class _TelegramNetworkFilter(logging.Filter):
    """Сжимает сетевые ошибки Telegram polling до одной строки WARNING.

    Без фильтра каждый обрыв сети генерирует ~139 строк стек-трейса.
    Реальные ошибки (не NetworkError) пропускаются без изменений.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        if record.exc_info and isinstance(record.exc_info[1], NetworkError):
            record.levelno = logging.WARNING
            record.levelname = "WARNING "
            record.exc_info = None
            record.exc_text = None
        return True


def setup_logging() -> None:
    Path("logs").mkdir(exist_ok=True)
    handlers = [
        logging.StreamHandler(),
        RotatingFileHandler("logs/app.log", maxBytes=2_000_000,
                            backupCount=3, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=handlers,
    )
    # Убавляем шум httpx и apscheduler
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)
    # Telegram polling NetworkError → одна строка WARNING вместо стек-трейса
    logging.getLogger("telegram.ext.Updater").addFilter(_TelegramNetworkFilter())


async def main() -> None:
    setup_logging()
    log = logging.getLogger(__name__)

    init_db()
    log.info("БД инициализирована: %s", settings.db_url)

    # Прогреваем CLV-калибровку весов из накопленной истории до первого цикла,
    # чтобы compute_score сразу использовал откалиброванные веса.
    refresh_detector_weights()
    # Прогреваем лиго-зависимое доверие модели sstats (#3).
    refresh_model_trust()

    # Telegram-приложение
    app = build_application()
    set_bot(app.bot)

    # Шедулер
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        run_once,
        IntervalTrigger(minutes=settings.poll_interval_minutes),
        max_instances=1,
        coalesce=True,
        id="odds_poll",
    )
    scheduler.add_job(
        update_elo_from_results,
        IntervalTrigger(hours=2),
        next_run_time=datetime.now(),
        max_instances=1,
        coalesce=True,
        misfire_grace_time=None,  # старт-прогон не выбрасывать как misfire
        id="elo_update",
    )
    # Проверки результатов/CLV/калибровки: запуск СРАЗУ на старте
    # (next_run_time=now) И каждый час. ВАЖНО: misfire_grace_time=None —
    # next_run_time берётся в момент add_job, а джобы стартуют только
    # после scheduler.start() (через ~1–2с: app.initialize→start→
    # polling). Эта задержка > дефолтных 1с grace → старт-прогон иначе
    # считается misfire и ВЫБРАСЫВАЕТСЯ (переносится на след. час) —
    # ровно поэтому после рестарта результаты не резолвились. None =
    # «выполнить, как бы поздно ни было». Все идемпотентны
    # (ResultNotification-дедуп, compute_pending_* пропускают сделанное;
    # max_instances=1+coalesce от наложений).
    scheduler.add_job(
        check_anomaly_results,
        IntervalTrigger(hours=1),
        next_run_time=datetime.now(),
        max_instances=1,
        coalesce=True,
        misfire_grace_time=None,
        id="result_check",
    )
    scheduler.add_job(
        compute_pending_clv,
        IntervalTrigger(hours=1),
        next_run_time=datetime.now(),
        max_instances=1,
        coalesce=True,
        misfire_grace_time=None,
        id="clv_compute",
    )
    scheduler.add_job(
        refresh_detector_weights,
        IntervalTrigger(hours=1),
        max_instances=1,
        coalesce=True,
        id="weight_calibration",
    )
    scheduler.add_job(
        compute_pending_calibration,
        IntervalTrigger(hours=1),
        next_run_time=datetime.now(),
        max_instances=1,
        coalesce=True,
        misfire_grace_time=None,
        id="prob_calibration",
    )
    scheduler.add_job(
        refresh_model_trust,
        IntervalTrigger(hours=1),
        max_instances=1,
        coalesce=True,
        id="model_trust",
    )

    # Запуск всего: app.initialize / start, потом polling, потом scheduler.
    # python-telegram-bot v21+ требует именно такой последовательности.
    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)
    scheduler.start()
    log.info("Шедулер запущен: опрос каждые %d мин; result/CLV/calib — "
             "при старте и каждый час; Elo — при старте и каждые 2ч",
             settings.poll_interval_minutes)

    await send_startup_message()

    # Сразу один цикл, чтобы не ждать первый интервал
    try:
        await run_once()
    except Exception:
        log.exception("Ошибка в первом цикле, продолжаем по расписанию")

    # Ждём сигнала остановки
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            # Windows: signal handlers через add_signal_handler не поддерживаются
            pass

    log.info("Бот работает. Ctrl+C для остановки.")
    await stop_event.wait()

    log.info("Останавливаемся...")
    scheduler.shutdown(wait=False)
    await app.updater.stop()
    await app.stop()
    await app.shutdown()


if __name__ == "__main__":
    import time

    _RESTART_DELAY = 30

    while True:
        try:
            asyncio.run(main())
            break  # чистый выход
        except KeyboardInterrupt:
            break
        except Exception:
            logging.getLogger(__name__).error(
                "Критическая ошибка, перезапуск через %d с...", _RESTART_DELAY,
                exc_info=True,
            )
            time.sleep(_RESTART_DELAY)
