"""Выборочный сброс detector-зависимых данных под чистую калибровку.

Бэкапит data/anomalies.sqlite, затем чистит Anomaly/OddsSnapshot/
AnomalyCLV/AnomalyOutcome/ProbCalibration/ResultNotification. Сохраняет
SstatsModelOutcome / MatchResult / Elo (см. src/maintenance.py).

Необратимо → требует явного --yes. Запускать, когда фон-бэкфилл НЕ пишет
в ту же БД (иначе гонка за SQLite).

    python -m scripts.reset_signal_data --yes
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

from src.config import settings
from src.maintenance import purge_detector_data


def _backup_sqlite() -> str | None:
    url = settings.db_url
    if not url.startswith("sqlite:///"):
        return None
    db_path = Path(url.replace("sqlite:///", "", 1))
    if not db_path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    bak = db_path.with_name(f"{db_path.name}.bak-{stamp}")
    shutil.copy2(db_path, bak)
    return str(bak)


def main() -> None:
    ap = argparse.ArgumentParser(description="selective signal-data reset")
    ap.add_argument("--yes", action="store_true",
                    help="подтверждение деструктивной операции (обязательно)")
    args = ap.parse_args()

    if not args.yes:
        print("Откажусь без --yes (операция необратима). "
              "Будут вычищены: anomalies, odds_snapshots, anomaly_clv, "
              "anomaly_outcomes, prob_calibration, result_notifications.\n"
              "Сохранятся: sstats_model_outcome, match_results, team_ratings.")
        sys.exit(1)

    bak = _backup_sqlite()
    print(f"Бэкап: {bak}" if bak else "Бэкап: пропущен (не файловый sqlite)")

    deleted = purge_detector_data()
    total = sum(deleted.values())
    for name, n in deleted.items():
        print(f"  {name:<22} удалено {n}")
    print(f"Итого удалено строк: {total}. Калибровка стартует с чистого листа.")


if __name__ == "__main__":
    main()
