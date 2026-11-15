"""Запуск из корня репозитория: python scripts/main.py"""
import argparse
from pathlib import Path

from pipeline import run_project

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Сравнение поз спортсменов на двух коротких видео")
    parser.add_argument("--reference", type=Path, default=ROOT / "data/reference_video.mp4")
    parser.add_argument("--input", type=Path, default=ROOT / "data/my.mp4")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    run_project(args.reference, args.input, args.output)


if __name__ == "__main__":
    main()
