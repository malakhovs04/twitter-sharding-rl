# Карта оценки проекта

| Что проверяется | Доказательства |
|---|---|
| Происхождение, формат и точность ключей | 01_data_overview.ipynb, data_pipeline.py, data/processed.json |
| Причинность, TTL, реальные модельные байты переноса | simulator.py, data_pipeline.py, tests/test_project.py |
| Разделение train/validation/test | results/protocol.json, experiments.py |
| Простой baseline и выбор его параметров | 02_simulation_and_baselines.ipynb, results/threshold_search.csv |
| Табличное Q-learning | 03_q_learning.ipynb, agents.py, results/q/ |
| Настоящая нейросеть и подбор гиперпараметров | 04_neural_dqn.ipynb, agents.py, results/dqn/ |
| Сравнение без подмены целевой функции | 05_comparison.ipynb, results/comparison.csv |
| Дообучение и границы достижимого | 06_safety_finetuning.ipynb, safe_finetuning.py, results/safety/ |
| Проверяемость реализации | 13 тестов: python -m unittest discover -s tests -v |
| Вклад и ограничения авторства | CONTRIBUTIONS.md |

## Ключевые ограничения

Одна недельная выборка не доказывает обобщение. Повторное использование test в дообучении отмечено явно. Результаты показывают компромисс стоимости и перегрузки, а не безусловное превосходство ИИ. Мощность и цены условны. Нейросеть получает более подробное представление состояния, чем табличный агент. Идеальная offline-граница знает будущее и не является реальной политикой.
