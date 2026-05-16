"""Разовый ретро-резолв старого бэклога аномалий.

Закрывает оговорку «sstats-окно результатов всего 3 дня»: чистит ложные
sentinel'ы ResultNotification(result_found=False), тянет результаты за
широкое окно из обоих источников (football-data + sstats), затем
до-считывает CLV / Brier-калибровку и обновляет кэши весов/доверия.

Идемпотентно. НЕ шлёт Telegram (бот не инициализирован в скрипте).

Запуск:
    python -m scripts.catch_up_results --days-back 30 --max-pages 40
"""
from __future__ import annotations

import argparse
import asyncio
import logging

from src.calibration import refresh_detector_weights
from src.clv import compute_pending_clv
from src.prob_calibration import calibration_summary, compute_pending_calibration
from src.result_checker import check_anomaly_results, clear_false_sentinels
from src.sstats_history import refresh_model_trust


def main() -> None:
    ap = argparse.ArgumentParser(description="retro-resolve anomaly backlog")
    ap.add_argument("--days-back", type=int, default=30,
                    help="окно результатов в днях (по умолчанию 30)")
    ap.add_argument("--max-pages", type=int, default=40,
                    help="лимит страниц sstats /Games/list (safety)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.ERROR)  # не светим apikey

    cleared = clear_false_sentinels()
    print(f"Очищено ложных sentinel'ов: {cleared}")

    asyncio.run(check_anomaly_results(days_back=args.days_back,
                                      sstats_max_pages=args.max_pages))

    print(f"CLV досчитано: {compute_pending_clv()}")
    print(f"Калибровка досчитана: {compute_pending_calibration()}")
    refresh_detector_weights()
    refresh_model_trust()

    cs = calibration_summary()
    if cs.get("n"):
        print(f"Калибровка рынка: n={cs['n']} Brier={cs['mean_brier']:.4f} "
              f"pending={cs['pending']}")
    print("DONE")


if __name__ == "__main__":
    main()
