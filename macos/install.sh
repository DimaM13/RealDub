#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "=== Встановлення RealDub для macOS ==="

# Перевірка Python 3
if ! command -v python3 &> /dev/null; then
    echo "❌ Помилка: python3 не знайдено. Встановіть Python 3 (наприклад через brew install python)."
    exit 1
fi

# Перевірка Homebrew та BlackHole для системного звуку
if command -v brew &> /dev/null; then
    if ! brew list blackhole-2ch &> /dev/null; then
        echo "🔊 Встановлюю BlackHole 2ch для захоплення звуку конференцій..."
        brew install blackhole-2ch || echo "⚠ Встановіть BlackHole вручну: brew install blackhole-2ch"
    else
        echo "✓ BlackHole вже встановлено"
    fi
else
    echo "💡 Для захоплення звуку конференцій рекомендується встановити BlackHole:"
    echo "   brew install blackhole-2ch"
fi

if [ ! -d ".venv" ]; then
    echo "Створюю віртуальне середовище .venv..."
    python3 -m venv .venv
fi

echo "Встановлюю залежності..."
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

echo "Завантажую моделі ШІ..."
python download_models.py

chmod +x start.sh

echo ""
echo "=== Встановлення завершено! Запускайте: ./start.sh ==="
