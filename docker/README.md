# Docker: сборка и запуск

Все команды выполняйте из корня репозитория в PowerShell. Сначала установите и запустите Docker Desktop. Конфигурация Docker собрана в этой папке: `Dockerfile`, `compose.yaml` и `Dockerfile.dockerignore`.

Compose запускается с параметром `-f docker/compose.yaml`. Контекст сборки и подключаемая папка проекта указывают на корень репозитория, поэтому исходники, тесты, ноутбуки и локальная папка `data/` доступны в `/workspace`.

## Сборка и запуск Jupyter

```powershell
docker compose -f docker/compose.yaml build notebook
docker compose -f docker/compose.yaml up -d notebook
docker compose -f docker/compose.yaml logs -f notebook
```

Откройте `http://127.0.0.1:8888` в браузере. Токен доступа Jupyter выводится в журнал. Порт доступен только с локального компьютера. Чтобы остановить сервис:

```powershell
docker compose -f docker/compose.yaml down
```

## Тесты и расчёты

Запустите независимые проверки и сверку нечётных/чётных секторов:

```powershell
docker compose -f docker/compose.yaml run --rm -T notebook python -m unittest discover -s tests -v
docker compose -f docker/compose.yaml run --rm -T notebook python scripts/check_partners.py
```

Вычислите полную сетку ОГУ/ПГУ для $N=3,\ldots,26$ (PowerShell 7). Случаи $N=27,28$ в этот параллельный диспетчер не входят:

```powershell
pwsh -NoProfile -File scripts/run_grid.ps1
```

Для Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_grid.ps1
```

Диспетчер собирает Docker-образ, подбирает предел параллельности по замерам (если не задан `-MaxParallel`) и запускает отдельный контейнер для каждого случая. Он планирует вычисления в пределах 70% памяти Docker и резервирует 0,5 ГиБ на рабочий контейнер. Например, `-MaxParallel 4` задаёт предел в четыре контейнера и пропускает предварительный подбор. Объём памяти Docker Engine можно узнать командой `docker info --format '{{.MemTotal}}'`; он отличается от объёма свободной памяти компьютера.

Для $N=27$ и $N=28$ запускайте по одному случаю. Файл `docker/compose.16g.yaml` ограничивает контейнер 16 ГиБ и не добавляет swap:

```powershell
docker compose -f docker/compose.yaml -f docker/compose.16g.yaml run --rm -T notebook python scripts/run_case.py 27 PBC
```

Замените `27` и `PBC` на нужные $N$ и границу. Сравнение новых методов с CSR при $N\leq26$:

```powershell
docker compose -f docker/compose.yaml run --rm -T notebook python scripts/compare_methods.py
```

Сравните замеры полной плотной диагонализации и разрежённого метода:

```powershell
docker compose -f docker/compose.yaml run --rm -T notebook python scripts/compare_dense.py --case-timeout 900
```

Команда сохраняет таблицы и график в `data/`, а также автоматически создаёт `data/dense_comparison.md`. Чтобы перестроить Markdown-отчёт по уже сохранённому JSON без повторного замера:

```powershell
docker compose -f docker/compose.yaml run --rm -T notebook python scripts/compare_dense.py --report-only
```

Выполните все ячейки ноутбука и сохраните результаты в нём:

```powershell
docker compose -f docker/compose.yaml run --rm -T notebook jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=14400 notebooks/XXX_chain_ED.ipynb
```

Для измерения ресурсов одного случая:

```powershell
docker compose -f docker/compose.yaml run --rm -T notebook python scripts/profile_resources.py 22 PBC
```

Результаты замеров и графики сохраняются в `data/`, эта папка исключена из Git. Jupyter можно оставить в фоне командой `up -d`; адрес и токен будут доступны в журнале контейнера.
