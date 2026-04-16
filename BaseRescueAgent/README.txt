BaseRescueAgent

Запуск агента
1. Установить зависимости окружения ADF/RCRS.
2. Из корня проекта выполнить: python main.py

Итоговая конфигурация
- поиск: гибридный модуль с ML
- детектор: гибридный модуль с ML
- планирование пути: устойчивый эвристический A*

Финальные модели в поставке
- поиск: models/search/model.joblib
- детектор: models/detector/model.joblib
- путь: ML в итоговой версии отключена, используется эвристический A*

Переобучение на новых логах
Новые логи нужно положить в папку runtime/raw_logs, чтобы внутри были директории вида manual__YYYYMMDD_HHMMSS.
Из корня проекта: 

export PYTHONPATH="$PWD/src"

1) Detector
python -m BaseRescueAgent.ml.build_detector_dataset_v3 \
  --input runtime/raw_logs \
  --output-dir runtime/datasets/retrain_detector_v3 \
  --allowed-detector-modes heuristic,heuristic_fallback,hybrid \
  --min-candidate-count 2 \
  --min-decisions-per-run 2

python -m BaseRescueAgent.ml.train_detector_v3 \
  --dataset runtime/datasets/retrain_detector_v3/detector_v3_dataset.csv \
  --output-dir models/retrained_detector_v3 \
  --group-field run_id \
  --test-size 0.25 \
  --random-state 42

2) Search
python -m BaseRescueAgent.ml.build_search_dataset_v2 \
  --input runtime/raw_logs \
  --output-dir runtime/datasets/retrain_search_v2 \
  --allowed-search-modes heuristic,heuristic_fallback,hybrid \
  --min-candidate-count 3 \
  --min-decisions-per-run 8

python -m BaseRescueAgent.ml.train_search_v2 \
  --dataset runtime/datasets/retrain_search_v2/search_v2_dataset.csv \
  --output-dir models/retrained_search_v2 \
  --group-field run_id \
  --test-size 0.25 \
  --random-state 42

3) Path (исследовательский пайплайн, не используется в итоговой конфигурации)
python -m BaseRescueAgent.ml.build_path_edge_dataset_v3 \
  --runtime-raw-logs runtime/raw_logs \
  --output-dir runtime/datasets/retrain_path_v3

python -m BaseRescueAgent.ml.train_path_edge_risk_v3 \
  --dataset runtime/datasets/retrain_path_v3/path_edge_risk_v3_dataset.csv \
  --output-dir models/retrained_path_v3 \
  --group-field run_id \
  --test-size 0.25 \
  --random-state 42
