"""Разовый бэкфилл исторической калибровки модели sstats.

Тянет сыгранные матчи из sstats (счёт + winProb/xG), скорит прогноз
модели против факта и пишет в таблицу sstats_model_outcome. Идемпотентно
— можно прерывать и запускать повторно, уже записанные id пропускаются.

Запуск:
    python -m scripts.backfill_sstats_history --max-games 2000
    python -m scripts.backfill_sstats_history --max-games 500 --sleep 0.5

Лимит sstats — 150 req/мин с ключом; --sleep держит запас. 2000 матчей
≈ 15 мин. SSTATS_API_KEY должен быть в .env.
"""
from __future__ import annotations

import argparse
import asyncio
import logging

from src.sstats_history import backfill, sstats_model_summary


def main() -> None:
    ap = argparse.ArgumentParser(description="sstats model calibration backfill")
    ap.add_argument("--max-games", type=int, default=1500,
                    help="максимум новых матчей за прогон (по умолчанию 1500)")
    ap.add_argument("--page-limit", type=int, default=200,
                    help="размер страницы /Games/list (по умолчанию 200)")
    ap.add_argument("--sleep", type=float, default=0.45,
                    help="пауза между glicko-запросами, сек (rate-limit)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    written = asyncio.run(backfill(
        max_games=args.max_games,
        page_limit=args.page_limit,
        sleep=args.sleep,
    ))
    print(f"\nЗаписано новых матчей: {written}")

    s = sstats_model_summary()
    if s.get("n"):
        print(f"Всего в калибровке: {s['n']}  "
              f"Brier={s['mean_brier']:.4f}  "
              f"log-loss={s['mean_log_loss']:.4f}  "
              f"(равномерный {s['uniform_brier']:.3f})")
        for lg in s["leagues"]:
            print(f"  {lg['league']:<28} n={lg['n']:<5} Brier={lg['brier']:.4f}")


if __name__ == "__main__":
    main()
