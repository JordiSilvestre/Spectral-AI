#!/bin/bash
# Lanza el benchmark y guarda resultados

cd "$(dirname "$0")/.."

echo "=== SpectralAI Benchmark ==="
echo "Generando resultados..."

# Pequeño test (5 frases, rápido - ~30s)
python python/benchmark.py \
    --num-sentences 5 \
    --num-rays 16 \
    --output python/benchmark_5sentences.json \
    --csv python/benchmark_5sentences.csv

echo ""
echo "=== Resultados guardados ==="
echo "JSON:  python/benchmark_5sentences.json"
echo "CSV:   python/benchmark_5sentences.csv"
echo ""
echo "Para un benchmark más grande (100 frases, ~10 min):"
echo "  python python/benchmark.py --num-sentences 100 --output benchmark_full.json"
