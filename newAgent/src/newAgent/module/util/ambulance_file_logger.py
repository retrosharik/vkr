import os
from datetime import datetime

class AmbulanceFileLogger:
    """Логгер с отдельным файлом на каждый запуск + имя карты"""
    def __init__(self, map_name: str = "unknown_map"):
        self.log_dir = "result"
        os.makedirs(self.log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"ambulance_logs_{map_name}_{timestamp}.txt")

        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("═" * 90 + "\n")
            f.write(f" СИМУЛЯЦИЯ НА КАРТЕ: {map_name.upper()}\n")
            f.write(f" Запуск: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("═" * 90 + "\n\n")

        self.start_cycle = None
        self.last_detection_cycle = None
        self.detected_count = 0

    def log_start(self, cycle: int):
        self.start_cycle = cycle
        self.last_detection_cycle = cycle
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"[START] Симуляция началась на цикле {cycle}\n\n")

    def log_human_detected(self, cycle: int, human_id: int, x: int, y: int):
        from_start = cycle - self.start_cycle if self.start_cycle is not None else 0
        from_last = cycle - self.last_detection_cycle if self.last_detection_cycle is not None else 0

        line = (f"[DETECT] ЖЕРТВА {human_id:6d} | "
                f"Цикл: {cycle:4d} | "
                f"С начала: {from_start:3d} тиков | "
                f"С предыдущей: {from_last:3d} тиков | "
                f"Позиция: ({x:5d}, {y:5d})\n")

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line)

        self.last_detection_cycle = cycle
        self.detected_count += 1

    def log_end(self, cycle: int):
        total_time = cycle - self.start_cycle if self.start_cycle is not None else 0
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write("\n" + "═" * 90 + "\n")
            f.write(f"[END] СИМУЛЯЦИЯ ЗАВЕРШЕНА на цикле {cycle}\n")
            f.write(f"Общее время симуляции: {total_time} тиков\n")
            f.write(f"Всего обнаружено жертв: {self.detected_count}\n")
            f.write("═" * 90 + "\n")
