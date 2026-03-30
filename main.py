import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.35
EMPTY_SECONDS_THRESHOLD = 3
OCCUPIED_SECONDS_THRESHOLD = 5
COLOR_EMPTY = (0, 200, 0)       # зеленый - стол пустой
COLOR_OCCUPIED = (0, 0, 200)    # красный - стол занят
COLOR_APPROACH = (0, 200, 255)  # желтый - подход к столу


def parse_args():
    parser = argparse.ArgumentParser(description="Детекция состояния столика по видео")
    parser.add_argument("--video", required=True, help="Путь к входному видео")
    parser.add_argument(
        "--roi",
        default=None,
        help="Координаты столика: x,y,w,h. Если не указано — выбор мышкой",
    )
    parser.add_argument("--output", default="output.mp4", help="Путь к выходному видео")
    parser.add_argument("--report", default="report.txt", help="Путь к текстовому отчёту")
    parser.add_argument("--model", default="yolov8n.pt", help="Модель YOLO")
    return parser.parse_args()


def select_roi(video_path: str) -> tuple:
    """Выбор зоны столика через cv2.selectROI."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Не удалось прочитать видео: {video_path}")

    print("Выберите зону столика мышкой и нажмите Enter/Space.")
    roi = cv2.selectROI("Выберите столик", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    if roi[2] == 0 or roi[3] == 0:
        raise RuntimeError("Зона столика не выбрана.")
    return roi


def boxes_intersect_roi(boxes, roi) -> bool:
    """Проверяет, пересекается ли хотя бы один bbox человека с зоной столика."""
    rx, ry, rw, rh = roi
    for box in boxes:
        bx1, by1, bx2, by2 = map(int, box)
        if bx1 < rx + rw and bx2 > rx and by1 < ry + rh and by2 > ry:
            return True
    return False


def main():
    args = parse_args()
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Видео: {width}x{height}, {fps:.1f} FPS, {total_frames} кадров")

    if args.roi:
        roi = tuple(map(int, args.roi.split(",")))
    else:
        cap.release()
        roi = select_roi(args.video)
        cap = cv2.VideoCapture(args.video)

    print(f"Зона столика (x, y, w, h): {roi}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    empty_frames_threshold = int(fps * EMPTY_SECONDS_THRESHOLD)
    occupied_frames_threshold = int(fps * OCCUPIED_SECONDS_THRESHOLD)
    state = "empty"
    frames_without_person = empty_frames_threshold + 1
    frames_with_person = 0
    events = []
    frame_idx = 0

    events.append({
        "frame": 0,
        "time_sec": 0.0,
        "event": "empty",
    })

    print("Обработка видео...")
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time_sec = frame_idx / fps

        results = model(frame, classes=[PERSON_CLASS_ID], conf=CONFIDENCE_THRESHOLD, verbose=False)
        person_boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) > 0 else []

        person_in_roi = boxes_intersect_roi(person_boxes, roi)

        prev_state = state

        if person_in_roi:
            frames_without_person = 0
            frames_with_person += 1
            if state == "empty":
                state = "approach"
                frames_with_person = 1
                events.append({
                    "frame": frame_idx,
                    "time_sec": round(current_time_sec, 2),
                    "event": "approach",
                })
            elif state == "approach" and frames_with_person > occupied_frames_threshold:
                state = "occupied"
                events.append({
                    "frame": frame_idx,
                    "time_sec": round(current_time_sec, 2),
                    "event": "occupied",
                })
        else:
            frames_without_person += 1
            if state in ("occupied", "approach") and frames_without_person > empty_frames_threshold:
                state = "empty"
                events.append({
                    "frame": frame_idx,
                    "time_sec": round(current_time_sec, 2),
                    "event": "empty",
                })

        rx, ry, rw, rh = roi
        if state == "empty":
            color = COLOR_EMPTY
            label = "EMPTY"
        elif state == "occupied":
            color = COLOR_OCCUPIED
            label = "OCCUPIED"
        else:
            color = COLOR_APPROACH
            label = "APPROACH"

        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), color, 3)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(frame, (rx, ry - th - 10), (rx + tw + 6, ry), color, -1)
        cv2.putText(frame, label, (rx + 3, ry - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        for box in person_boxes:
            bx1, by1, bx2, by2 = map(int, box)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 200, 0), 2)

        time_str = f"Time: {current_time_sec:.1f}s"
        cv2.putText(frame, time_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            elapsed = time.time() - start_time
            progress = frame_idx / total_frames * 100
            print(f"  [{progress:5.1f}%] Кадр {frame_idx}/{total_frames} ({elapsed:.1f}s)")

    cap.release()
    out.release()
    elapsed = time.time() - start_time
    print(f"Готово! Обработано {frame_idx} кадров за {elapsed:.1f}s")
    print(f"Выходное видео: {args.output}")

    df = pd.DataFrame(events)
    print("\n=== Таблица событий ===")
    print(df.to_string(index=False))

    delays = []
    for i, row in df.iterrows():
        if row["event"] == "empty":
            next_approach = df[(df.index > i) & (df["event"] == "approach")]
            if not next_approach.empty:
                delay = next_approach.iloc[0]["time_sec"] - row["time_sec"]
                delays.append(delay)

    if delays:
        avg_delay = np.mean(delays)
        min_delay = np.min(delays)
        max_delay = np.max(delays)
    else:
        avg_delay = min_delay = max_delay = 0.0

    report_lines = [
        "=" * 50,
        "ОТЧЁТ: Детекция состояния столика",
        "=" * 50,
        f"Видео: {args.video}",
        f"Зона столика (x, y, w, h): {roi}",
        f"Модель: {args.model}",
        f"Всего кадров: {frame_idx}",
        f"FPS: {fps:.1f}",
        f"Длительность: {frame_idx / fps:.1f} сек",
        "",
        "--- Статистика ---",
        f"Всего событий: {len(df)}",
        f"- empty:    {len(df[df['event'] == 'empty'])}",
        f"- approach:  {len(df[df['event'] == 'approach'])}",
        f"- occupied:  {len(df[df['event'] == 'occupied'])}",
        "",
        f"Переходов empty -> approach: {len(delays)}",
        f"Среднее время ожидания (empty -> approach): {avg_delay:.2f} сек",
        f"Минимальное время ожидания: {min_delay:.2f} сек",
        f"Максимальное время ожидания: {max_delay:.2f} сек",
        "",
        "--- Таблица событий ---",
        df.to_string(index=False),
    ]

    report_text = "\n".join(report_lines)
    print(f"\n{report_text}")

    with open(args.report, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nОтчёт сохранён: {args.report}")


if __name__ == "__main__":
    main()
