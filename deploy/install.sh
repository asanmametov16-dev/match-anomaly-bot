#!/usr/bin/env bash
# Установка/обновление systemd-службы бота (Ubuntu). Идемпотентно.
# НЕ запускает службу — старт отдельным шагом (после переноса данных).
set -e
APP=/opt/match-anomaly-bot

id mab >/dev/null 2>&1 || useradd --system --no-create-home \
    --shell /usr/sbin/nologin mab
mkdir -p "$APP/data" "$APP/logs"
chown -R mab:mab "$APP"
cp -f "$APP/deploy/match-anomaly-bot.service" \
    /etc/systemd/system/match-anomaly-bot.service
systemctl daemon-reload
systemctl enable match-anomaly-bot >/dev/null 2>&1 || true
echo "===SERVICE INSTALLED==="
