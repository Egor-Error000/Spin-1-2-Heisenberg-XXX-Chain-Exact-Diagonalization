---
name: maintain-ed-workflow
description: Maintains the Dockerized exact-diagonalization code, independent tests, Jupyter report, and resource measurements. Use when changing numerical algorithms, supported chain sizes, validation, runtime configuration, or notebook outputs.
---

# Работа с численным расчётом ED

1. **Сохраняйте одну реализацию модели.** Численный алгоритм находится в `src/xxx_chain.py`; notebook импортирует его. Для вычислительного ядра используйте NumPy, Numba и SciPy, для графиков Matplotlib, для длительных этапов `tqdm.auto`. Не копируйте вычислительные функции в notebook.
2. **Проверяйте физические инварианты.** Обмен сохраняет число спинов вверх, матрица эрмитова, (S^z_{\rm tot}=0) содержит основное состояние при чётном (N), локальные диагонали равны (pm1/4), обменный элемент равен (+1/2), (E_0(4,\mathrm{ПГУ})=-2). Любое изменение базиса или границ сопровождайте независимой проверкой.
3. **Запускайте проверку в Docker.** Из корня репозитория: `docker compose run --rm notebook python -m unittest discover -s tests -v`. Эталон из тензорных произведений сравнивает элементы и спектр при (N\leq8); для (N\leq16) проверяются невязки и повторяемость eigsh. Для больших (N) контролируйте невязку и пиковую память отдельно.
4. **Измеряйте расширение диапазона.** Перед добавлением большего (N) прогоните `docker compose run --rm -T notebook python scripts/profile_resources.py N PBC` в отдельном контейнере. Сравните измеренный пик с фактической памятью Docker, оставьте запас для Jupyter и промежуточных массивов. Предыдущие 15,34 ГиБ — результат конкретной конфигурации, не универсальный лимит.
5. **Сохраняйте воспроизводимый отчёт.** После изменения расчёта выполните `docker compose run --rm notebook jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 notebooks/XXX_chain_ED.ipynb`. Проверьте отсутствие ошибочных ячеек, таблицу и оба графика. Локальные CSV/PNG пишутся в игнорируемую `data/`; notebook должен запускаться и после клонирования без этой папки.

Параллелизм выбирайте по замеру полного времени и памяти, а не только по числу доступных ядер.
