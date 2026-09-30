#!/usr/bin/env bash
# ops/launchd.sh — install/uninstall the weekly candidate retrain on the laptop.
#
# Runs ops/retrain.sh Mondays at 10:15am local time (after the overnight
# ingest). If the Mac is asleep then, launchd runs it once on wake.
# It only builds candidates; promoting stays manual.
#
#   bash ops/launchd.sh install
#   bash ops/launchd.sh uninstall
#   bash ops/launchd.sh status
#   bash ops/launchd.sh run          # start a retrain now
#
# Log: ~/Library/Logs/puckzone-retrain.log

set -euo pipefail
LABEL="com.puckzone.retrain"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG="$HOME/Library/Logs/puckzone-retrain.log"
RETRAIN_SH="$(cd "$(dirname "$0")" && pwd)/retrain.sh"

install() {
  mkdir -p "$(dirname "$PLIST")"
  cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array><string>/bin/bash</string><string>$RETRAIN_SH</string></array>
  <key>StartCalendarInterval</key>
  <dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>10</integer><key>Minute</key><integer>15</integer></dict>
  <key>StandardOutPath</key><string>$LOG</string>
  <key>StandardErrorPath</key><string>$LOG</string>
</dict>
</plist>
EOF
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  launchctl bootstrap "gui/$(id -u)" "$PLIST"
  echo "Installed $LABEL -> $RETRAIN_SH (Mondays 10:15; log: $LOG)"
}

case "${1:-}" in
  install)   install ;;
  uninstall) launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true; rm -f "$PLIST"; echo "Removed $LABEL" ;;
  status)    launchctl print "gui/$(id -u)/$LABEL" 2>/dev/null | grep -E "state|last exit code|runs" || echo "not loaded" ;;
  run)       launchctl kickstart "gui/$(id -u)/$LABEL" && echo "Started; tail -f $LOG" ;;
  *)         echo "usage: $0 install|uninstall|status|run"; exit 1 ;;
esac
