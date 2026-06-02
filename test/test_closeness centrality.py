"""
Тест closeness centrality с использованием mpnetworkx_simple
Сравнивает последовательную и параллельную версии (на всех узлах графа).
"""

import urllib.request
import gzip
import networkx as nx
import matplotlib.pyplot as plt
import ssl
import time
import os
import random
import multiprocessing as mp

import mpnetworkx_simple

# ------------------------------------------------------------
# Конфигурация
# ------------------------------------------------------------
DATA_URL = "https://snap.stanford.edu/data/amazon0505.txt.gz"
FILENAME = "amazon0505.txt.gz"
SAMPLE_SIZE = None          # None = вычисляем для всех узлов
PROCESSES = 16              # число процессов для параллельной версии

# ------------------------------------------------------------
# Загрузка графа (подграф из первых max_nodes узлов)
# ------------------------------------------------------------
def download_data():
    if os.path.exists(FILENAME):
        print(f"Файл {FILENAME} уже существует, используем его.")
        return FILENAME
    try:
        print("Скачивание Amazon графа (250 МБ)...")
        req = urllib.request.Request(DATA_URL, headers={'User-Agent': 'Mozilla/5.0'})
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, context=ctx) as resp:
            with open(FILENAME, 'wb') as f:
                f.write(resp.read())
        print("Успешно!")
        return FILENAME
    except Exception as e:
        print(f"Ошибка: {e}")
        return None

def load_graph(filename, max_nodes=10000):
    G = nx.Graph()
    print("Загрузка рёбер...")
    with gzip.open(filename, 'rb') as f:
        for line in f:
            if not line.startswith(b'#'):
                try:
                    u, v = map(int, line.decode().strip().split())
                    G.add_edge(u, v)
                except ValueError:
                    continue
    # Берём подграф из первых max_nodes узлов
    nodes = list(G.nodes())[:max_nodes]
    G_sub = G.subgraph(nodes).copy()
    print(f"Загружен подграф: {G_sub.number_of_nodes():,} узлов, {G_sub.number_of_edges():,} рёбер")
    return G_sub

# ------------------------------------------------------------
# Тестовые функции
# ------------------------------------------------------------
def test_sequential_closeness(G, sample_size):
    """Последовательная closeness (оригинальный NetworkX)"""
    print(f"\n--- Последовательная closeness (выборка: {'все узлы' if sample_size is None else sample_size}) ---")
    nodes = list(G.nodes())
    if sample_size is None or sample_size >= len(nodes):
        selected = nodes
    else:
        random.seed(42)
        selected = random.sample(nodes, sample_size)
    start = time.time()
    closeness = {}
    for node in selected:
        closeness[node] = nx.closeness_centrality(G, u=node, use_parallel=False)
    elapsed = time.time() - start
    top = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"Время: {elapsed:.2f} сек")
    print("Топ-10 узлов (последовательно):")
    for i, (node, val) in enumerate(top, 1):
        print(f"  {i:2}. Узел {node:4}: {val:.6f}")
    return closeness, elapsed, top

def test_parallel_closeness(G, processes, sample_size):
    """Параллельная closeness через mpnetworkx_simple"""
    print(f"\n--- Параллельная closeness ({processes} процессов, выборка: {'все узлы' if sample_size is None else sample_size}) ---")
    start = time.time()
    closeness = nx.closeness_centrality(G, processes=processes, sample_size=sample_size)
    elapsed = time.time() - start
    top = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"Время: {elapsed:.2f} сек")
    print("Топ-10 узлов (параллельно):")
    for i, (node, val) in enumerate(top, 1):
        print(f"  {i:2}. Узел {node:4}: {val:.6f}")
    return closeness, elapsed, top

# ------------------------------------------------------------
# Визуализация подграфа (топ-10 + соседи)
# ------------------------------------------------------------
def draw_subgraph(G, top_nodes, title):
    top_set = {node for node, _ in top_nodes[:10]}
    neighbors = set()
    for node in top_set:
        neighbors.update(G.neighbors(node))
    sub_nodes = list(top_set.union(neighbors))[:200]
    H = G.subgraph(sub_nodes)
    pos = nx.spring_layout(H, k=0.8, iterations=30, seed=42)
    plt.figure(figsize=(12, 10))
    colors = ['#e74c3c' if n in top_set else '#3498db' for n in H.nodes()]
    sizes = [300 if n in top_set else 50 for n in H.nodes()]
    nx.draw_networkx_edges(H, pos, alpha=0.2, edge_color='gray')
    nx.draw_networkx_nodes(H, pos, node_size=sizes, node_color=colors, alpha=0.7)
    labels = {n: str(n) for n in H.nodes() if n in top_set}
    nx.draw_networkx_labels(H, pos, labels, font_size=8, font_weight='bold')
    plt.title(title)
    plt.axis('off')
    outfile = f"closeness_{title.replace(' ', '_')}.png"
    plt.savefig(outfile, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Сохранено: {outfile}")

# ------------------------------------------------------------
# График масштабирования
# ------------------------------------------------------------
def plot_scaling(G, sample_size, procs=[1, 2, 4, 8, 16]):
    print("\n--- Масштабирование closeness centrality ---")
    times = []
    for p in procs:
        print(f"  Процессов: {p}...", end=' ', flush=True)
        start = time.time()
        nx.closeness_centrality(G, processes=p, sample_size=sample_size)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"{elapsed:.2f} с")
    plt.figure(figsize=(8, 5))
    plt.plot(procs, times, 'bo-', label='Реальное время')
    if times[0] > 0:
        ideal = [times[0] / p for p in procs]
        plt.plot(procs, ideal, 'r--', label='Идеальное ускорение')
    plt.xlabel('Число процессов')
    plt.ylabel('Время (секунды)')
    plt.title(f'Масштабирование closeness centrality (граф {G.number_of_nodes():,} узлов)')
    plt.legend()
    plt.grid(True)
    plt.savefig('scaling_closeness.png')
    plt.show()
    print("  График сохранён: scaling_closeness.png")

# ------------------------------------------------------------
# Основная функция ------------------------------------------------------------

def main():
    # Применяем патчи вашей библиотеки
    patcher = mpnetworkx_simple.apply_patches(auto_detect_threshold=100)

    # Загружаем подграф Amazon (10 000 узлов)
    data_file = download_data()
    if not data_file:
        return
    G = load_graph(data_file, max_nodes=10000)
    if G is None or G.number_of_nodes() == 0:
        return

    # Выводим информацию о тесте
    print("=" * 70)
    print("Тест closeness centrality: сравнение последовательной и параллельной версий")
    print(f"Граф: Amazon (подграф из {G.number_of_nodes():,} узлов, {G.number_of_edges():,} рёбер)")
    print(f"Выборка: {'все узлы' if SAMPLE_SIZE is None else SAMPLE_SIZE}, процессов: {PROCESSES}")
    print("=" * 70)

    # Последовательный тест
    seq_res, seq_time, seq_top = test_sequential_closeness(G, SAMPLE_SIZE)

    # Параллельный тест
    par_res, par_time, par_top = test_parallel_closeness(G, PROCESSES, SAMPLE_SIZE)

    # Итоги
    print("\n" + "=" * 50)
    print("ИТОГОВОЕ СРАВНЕНИЕ:")
    print(f"  Последовательно: {seq_time:.2f} сек")
    print(f"  Параллельно ({PROCESSES} процессов): {par_time:.2f} сек")
    print(f"  Ускорение: {seq_time/par_time:.2f}×")

    # Совпадение топ-10 узлов
    seq_ids = [node for node, _ in seq_top]
    par_ids = [node for node, _ in par_top]
    common = len(set(seq_ids) & set(par_ids))
    print(f"\nСовпадение топ-10 узлов: {common}/10")

    # Визуализация подграфа (на основе последовательного результата)
    draw_subgraph(G, seq_top, "Closeness Sequential (top-10)")

    # График масштабирования (запускается на всех узлах, может быть долгим)
    plot_scaling(G, SAMPLE_SIZE, procs=[1, 2, 4, 8, 16])

    # Восстанавливаем оригинальный NetworkX
    patcher.restore_original()
    print("\nТест завершён.")

if __name__ == "__main__":
    mp.freeze_support()
    main()
