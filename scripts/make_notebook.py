"""Regenerate the source cells of the XXX report before executing it."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "XXX_chain_ED.ipynb"


def markdown(cell_id: str, source: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {},
            "source": source.strip().replace("\\\\", "\\").splitlines(keepends=True)}


def code(cell_id: str, source: str) -> dict:
    return {"cell_type": "code", "id": cell_id, "metadata": {}, "execution_count": None,
            "outputs": [], "source": source.strip().replace("\\\\", "\\").splitlines(keepends=True)}


cells = [
    markdown("intro", r"""
# XXX-цепочка спинов 1/2: энергия основного состояния

Для всех целых $N=3,\ldots,26$ рассчитываем открытые (ОГУ) и периодические
(ПГУ) границы. Полагаем $J=1$ и

$$H=\sum_{(i,j)\in B}\mathbf S_i\cdot\mathbf S_j
=\sum_{(i,j)\in B}\left[S_i^z S_j^z+
\frac12(S_i^+S_j^-+S_i^-S_j^+)\right],\qquad S^\alpha=\frac{\sigma^\alpha}{2}.$$

Для ОГУ $B=\{(1,2),\ldots,(N-1,N)\}$; для ПГУ добавляется $(N,1)$.
Искомая удельная энергия $e_0(N)=E_0(N)/N$ сравнивается с
$e_\infty=1/4-\ln2$. В обозначениях приложенного конспекта
$H=-J_{\rm конспект}\sum\mathbf S_i\cdot\mathbf S_j$ это соответствует
$J_{\rm конспект}=-1$.
"""),
    markdown("method", r"""
## Базис и метод

Состояние кодируется битовой строкой: бит 1 означает спин вверх.
Оператор $H$ сохраняет $S^z_{\rm total}=n_\uparrow-N/2$. Для чётного $N$
используем $n_\uparrow=N/2$; для нечётного $n_\uparrow=(N-1)/2$,
то есть $S^z_{\rm total}=-1/2$. Сектор $+1/2$ имеет ту же энергию из-за
одновременного переворота всех спинов.

Для каждой связи параллельных спинов диагональный вклад равен $+1/4$;
антипараллельных — $-1/4$. Перестановка антипараллельных спинов даёт
внедиагональный элемент $+1/2$. Numba формирует разреженную матрицу CSR
блоками; `scipy.sparse.linalg.eigsh(k=1, which="SA")` ищет нижний уровень.
Проверяем невязку $\|H\psi-E_0\psi\|_2$, норму $\psi$ и повторный запуск
с другим начальным вектором. Подробный вывод — в `docs/theory.md`.
"""),
    code("setup", r"""
import csv
import sys
from math import comb
from pathlib import Path

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
DATA = ROOT / "data"
CASES = DATA / "cases"
DATA.mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display
from tqdm.auto import tqdm
from result_store import SOURCE_HASH, load_case, load_grid, write_case
from xxx_chain import ground_state

SIZES = tuple(range(3, 27))
E_INFINITY = 0.25 - np.log(2.0)
print(f"48 случаев; аналитический предел: {E_INFINITY:.12f}")
"""),
    code("calculation", r"""
cached, missing = load_grid(CASES, SIZES)
print(f"Проверенных контейнерных результатов: {len(cached)}; требуется вычислить: {len(missing)}")
for n, periodic in tqdm(missing, desc="Недостающие цепочки"):
    row = ground_state(n, periodic, progress=True, repeat=True)
    write_case(CASES, row)

results = [load_case(CASES, n, boundary)
           for n in SIZES for boundary in ("OBC", "PBC")]
assert len(results) == 48 and all(row is not None for row in results)
assert {(int(row["N"]), str(row["boundary"])) for row in results} == {
    (n, boundary) for n in SIZES for boundary in ("OBC", "PBC")}
fields = ("N", "boundary", "n_up", "sz_total", "dimension", "nnz", "E0", "e0",
          "residual", "norm_error", "repeat_delta", "csr_gib")
with (DATA / "results.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(results)
print("Сохранено: data/results.csv")
"""),
    code("table", r"""
lines = [r"| N | ГУ | $n_\\uparrow$ | Размер блока | $E_0$ | $E_0/N$ | Невязка | $|\\Delta E|$ повтора |",
         "|---:|:---:|---:|---:|---:|---:|---:|---:|"]
for row in results:
    lines.append(f"| {row['N']} | {row['boundary']} | {row['n_up']} | "
                 f"{row['dimension']:,} | {row['E0']:.10f} | {row['e0']:.10f} | "
                 f"{row['residual']:.2e} | {row['repeat_delta']:.2e} |")
display(Markdown("\n".join(lines)))
"""),
    markdown("finite-size", r"""
## Сходимость и чётность

Для четырёх серий отдельно подгоняем последние четыре точки. Для ОГУ
используем $e_0\approx a+b/N$ (вклад краёв), для ПГУ —
$e_0\approx a+c/N^2$ (без краёв). Вторая модель для нечётных колец
также служит только диагностикой: фрустрация меняет коэффициент поправки.
Изотропная критическая цепочка может иметь логарифмические поправки;
интерсепты на малых $N$ не являются строгим доказательством предела.
"""),
    code("fits", r"""
fits = {}
fit_lines = [r"| ГУ | Чётность | Использованные N | Оценка $a$ | $a-e_\\infty$ |",
             "|:---:|:---:|:---:|---:|---:|"]
for boundary in ("OBC", "PBC"):
    for parity in (0, 1):
        subset = [row for row in results if row["boundary"] == boundary and row["N"] % 2 == parity][-4:]
        n = np.array([row["N"] for row in subset], dtype=float)
        energy = np.array([row["e0"] for row in subset], dtype=float)
        x = 1/n if boundary == "OBC" else 1/n**2
        slope, intercept = np.polyfit(x, energy, 1)
        fits[(boundary, parity)] = (slope, intercept)
        fit_lines.append(f"| {boundary} | {'чётные' if parity == 0 else 'нечётные'} | "
                         f"{', '.join(map(str, n.astype(int)))} | {intercept:.10f} | "
                         f"{intercept-E_INFINITY:+.3e} |")
display(Markdown("\n".join(fit_lines)))
"""),
    code("energy-plots", r"""
colors = {"OBC": "#2563eb", "PBC": "#dc2626"}
markers = {0: "o", 1: "^"}
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
for boundary in ("OBC", "PBC"):
    for parity in (0, 1):
        subset = [row for row in results if row["boundary"] == boundary and row["N"] % 2 == parity]
        n = np.array([row["N"] for row in subset])
        x = 1 / n
        y = np.array([row["e0"] for row in subset])
        label = f"{boundary}, {'чётные' if parity == 0 else 'нечётные'}"
        axes[0].plot(x, y, marker=markers[parity], color=colors[boundary],
                     linestyle="-" if parity == 0 else "--", label=label)
        axes[1].plot(x, y-E_INFINITY, marker=markers[parity], color=colors[boundary],
                     linestyle="-" if parity == 0 else "--", label=label)
        axes[0].scatter([0], [fits[(boundary, parity)][1]], marker="x", color=colors[boundary])
axes[0].axhline(E_INFINITY, color="black", linestyle=":", label=r"$1/4-\ln2$")
axes[1].axhline(0, color="black", linestyle=":")
axes[0].set_ylabel(r"$e_0(N)=E_0(N)/N$")
axes[1].set_ylabel(r"$e_0(N)-(1/4-\ln2)$")
for ax in axes:
    ax.set_xlabel(r"$1/N$")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
axes[0].set_title("Удельная энергия")
axes[1].set_title("Отклонение от аналитического предела")
fig.savefig(DATA / "energies.png", dpi=180)
plt.show()
"""),
    code("conclusions", r"""
lines = ["### Наблюдаемые закономерности", ""]
for boundary in ("OBC", "PBC"):
    for parity in (0, 1):
        branch = [row for row in results if row["boundary"] == boundary and row["N"] % 2 == parity]
        last = branch[-1]
        delta = last["e0"] - E_INFINITY
        side = "выше" if delta > 0 else "ниже"
        lines.append(f"- {boundary}, {'чётные' if parity == 0 else 'нечётные'}: "
                     f"при $N={last['N']}$ энергия {side} предела на {abs(delta):.6g}; "
                     f"экстраполяция даёт {fits[(boundary, parity)][1]:.9f}.")
lines += ["", "Открытые концы и замкнутые кольца имеют общий объёмный предел, "
          "но различаются краевыми поправками. Нечётное периодическое кольцо "
          "фрустрировано, поэтому его конечномерная ветвь отличается от чётной. "
          "Наблюдаемое приближение к пределу и приведённые подгонки имеют численный характер."]
display(Markdown("\n".join(lines)))
"""),
    markdown("validation", r"""
## Проверка уравнения и численного решения

`tests/validate.py` независимо строит гамильтониан из тензорных произведений
матриц Паули при $N=3,\ldots,8$ и сравнивает его со всеми элементами
битового блока и нижним уровнем. Для $N\leq16$ тест проходит все секторы
намагниченности; для каждого расчёта в таблице проверены две стартовые точки
`eigsh`, невязка и нормировка. Точное решение: $E_0(3,\mathrm{ОГУ})=-1$,
$E_0(3,\mathrm{ПГУ})=-3/4$, $E_0(4,\mathrm{ПГУ})=-2$.
"""),
    code("diagnostics", r"""
by_case = {(row["N"], row["boundary"]): row for row in results}
assert abs(by_case[(3, "OBC")]["E0"] + 1) < 1e-10
assert abs(by_case[(3, "PBC")]["E0"] + 0.75) < 1e-10
assert abs(by_case[(4, "PBC")]["E0"] + 2) < 1e-10
print("Максимальная невязка:", max(row["residual"] for row in results))
print("Максимальная ошибка нормы:", max(row["norm_error"] for row in results))
print("Максимальное расхождение повторов:", max(row["repeat_delta"] for row in results))
partner_path = DATA / "partner_checks.json"
if partner_path.exists():
    import json
    partners = json.loads(partner_path.read_text(encoding="utf-8"))
    if partners.get("source_sha256") == SOURCE_HASH:
        print(f"Партнёрские сектора: {partners['checked_cases']} случаев, "
              f"максимальное расхождение {partners['max_energy_delta']:.3e}")
    else:
        print("Проверка партнёрских секторов устарела; перезапустите scripts/check_partners.py.")
else:
    print("Партнёрские сектора: запустите scripts/check_partners.py для отдельной проверки.")
"""),
    markdown("resources", r"""
## Ресурсы Docker

Каждый результат диспетчера измерен в отдельном контейнере: время —
`perf_counter`, CPU — `process_time`, пик памяти процесса — `ru_maxrss`.
Время включает **два** запуска `eigsh` для проверки повторяемости; 100% CPU
соответствует одному занятому ядру. При запуске notebook без диспетчера
энергии вычисляются, но сопоставимых измерений отдельного контейнера нет.
Диспетчер допускает одновременно не более 14 контейнеров и резервирует
30% памяти Docker плюс 0,5 ГиБ на каждый запущенный расчёт.
"""),
    code("resource-table", r"""
usage = [row for row in results if "peak_rss_gib" in row]
if usage:
    fields = ("N", "boundary", "dimension", "nnz", "wall_seconds", "cpu_seconds",
              "cpu_percent_one_core", "peak_rss_gib", "csr_gib", "docker_memory_gib")
    with (DATA / "resource_measurements.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(usage)
    lines = ["| N | ГУ | Время, с | CPU, с | CPU, % ядра | Пик RAM, ГиБ | CSR, ГиБ |",
             "|---:|:---:|---:|---:|---:|---:|---:|"]
    for row in usage:
        if row["N"] >= 16 and row["boundary"] == "PBC":
            lines.append(f"| {row['N']} | PBC | {row['wall_seconds']:.2f} | "
                         f"{row['cpu_seconds']:.2f} | {row['cpu_percent_one_core']:.1f} | "
                         f"{row['peak_rss_gib']:.2f} | {row['csr_gib']:.3f} |")
    display(Markdown("\n".join(lines)))
else:
    print("Изолированные замеры отсутствуют; запустите scripts/run_grid.ps1 на Windows.")
benchmark_path = DATA / "benchmark.json"
if benchmark_path.exists():
    import json
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8-sig"))
    print(f"Выбранный предел параллелизма: {benchmark['chosen_containers']} контейнеров.")
    for measurement in benchmark["measurements"]:
        print(f"лимит {measurement['containers']:2d}: {measurement['wall_seconds']:.2f} с")
"""),
    code("resource-plot", r"""
if usage:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for boundary in ("OBC", "PBC"):
        branch = [row for row in usage if row["boundary"] == boundary]
        axes[0].plot([row["N"] for row in branch], [row["peak_rss_gib"] for row in branch],
                     marker="o", color=colors[boundary], label=boundary)
        axes[1].semilogy([row["N"] for row in branch], [row["wall_seconds"] for row in branch],
                        marker="o", color=colors[boundary], label=boundary)
    docker_gib = usage[0]["docker_memory_gib"]
    axes[0].axhline(docker_gib, color="black", linestyle="--", label="Память Docker")
    axes[0].axhline(0.70*docker_gib, color="#6b7280", linestyle=":", label="Бюджет диспетчера")
    axes[0].set_ylabel("Пиковая память процесса, ГиБ")
    axes[1].set_ylabel("Время с повтором eigsh, с (лог. шкала)")
    for ax in axes:
        ax.set_xlabel("Число спинов N")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(DATA / "resource_usage.png", dpi=180)
    plt.show()
    peak26 = max(row["peak_rss_gib"] for row in usage if row["N"] == 26)
    def candidates(n):
        k = n // 2
        return comb(n, k) + 2*n*comb(n-2, k-1)
    estimate28 = peak26 * candidates(28) / candidates(26)
    print(f"Docker: {docker_gib:.2f} ГиБ; N=26: пик {peak26:.2f} ГиБ (измерено); "
          f"N=28: около {estimate28:.1f} ГиБ (оценка, не измерение).")
    print("Подтверждённый максимум текущего кода — N=26; N=27 не тестировался и пока не поддерживается.")
"""),
]

notebook = {"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(TARGET)
