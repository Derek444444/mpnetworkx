
"""
test_c.py – Closeness centrality для больших графов (выборка, параллельно)
Исправлена ошибка сериализации вложенной функции для Windows.
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
from concurrent.futures import ProcessPoolExecutor
import pickle

# ============================================================
# Глобальная функция для обработки чанка в параллельном режиме
# ============================================================
def _process_closeness_chunk(task):
    """task = (chunk_nodes, graph_data)"""
    chunk_nodes, graph_data = task
    local_G = pickle.loads(graph_data)
    res = {}
    for node in chunk_nodes:
        try:
            res[node] = nx.closeness_centrality(local_G, u=node)
        except Exception:
            res[node] = 0.0
    return res

def parallel_closeness_centrality(G, processes=None, sample_size=500):
    """
    Параллельное вычисление closeness centrality на выборке узлов.
    """
    if processes is None:
        processes = max(2, mp.cpu_count() // 2)

    nodes = list(G.nodes())
    if sample_size and sample_size < len(nodes):
        random.seed(42)
        selected_nodes = random.sample(nodes, sample_size)
    else:
        selected_nodes = nodes

    chunk_size = max(1, len(selected_nodes) // processes)
    chunks = [selected_nodes[i:i+chunk_size] for i in range(0, len(selected_nodes), chunk_size)]

    graph_data = pickle.dumps(G)
    tasks = [(chunk, graph_data) for chunk in chunks]

    with ProcessPoolExecutor(max_workers=processes) as executor:
        futures = [executor.submit(_process_closeness_chunk, task) for task in tasks]
        result = {}
        for f in futures:
            result.update(f.result())
    return result

# ============================================================
# Загрузка графа (поддержка Amazon и Facebook)
# ============================================================
def download_data(url, filename):
    if os.path.exists(filename):
        print(f"Файл {filename} уже существует.")
        return filename
    print(f"Скачивание {filename}...")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, context=ctx) as resp:
        with open(filename, 'wb') as f:
            f.write(resp.read())
    print("Скачивание завершено.")
    return filename

def load_graph(filename):
    G = nx.Graph()
    with gzip.open(filename, 'rb') as f:
        for line in f:
            if not line.startswith(b'#'):
                try:
                    u, v = map(int, line.decode().strip().split())
                    G.add_edge(u, v)
                except:
                    continue
    print(f"Граф загружен: {G.number_of_nodes():,} узлов, {G.number_of_edges():,} рёбер")
    return G

# ============================================================
# Последовательная версия (выборка)
# ============================================================
def sequential_closeness(G, sample_size=500):
    print(f"\n--- Последовательная closeness (выборка {sample_size} узлов) ---")
    nodes = list(G.nodes())
    if sample_size and sample_size < len(nodes):
        random.seed(42)
        selected = random.sample(nodes, sample_size)
    else:
        selected = nodes

    start = time.time()
    closeness = {}
    for node in selected:
        closeness[node] = nx.closeness_centrality(G, u=node)
    elapsed = time.time() - start
    print(f"Время: {elapsed:.2f} сек ({elapsed/60:.2f} мин)")
    top = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:10]
    print("Топ-10 узлов (по выборочным значениям):")
    for i, (node, val) in enumerate(top, 1):
        print(f"  {i}. Узел {node}: {val:.6f}")
    return closeness, elapsed, top

# ============================================================
# Визуализация подграфа (топ-10 + соседи, не более 200 узлов)
# ============================================================
def draw_subgraph(G, top_nodes, method_name, sample_size):
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
    plt.title(f"{method_name}\nГраф с {G.number_of_nodes():,} узлами (выборка {sample_size})\nКрасные – топ-10 closeness")
    plt.axis('off')
    outfile = f"closeness_{method_name}.png"
    plt.savefig(outfile, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Сохранено: {outfile}")

# ============================================================
# График масштабирования (зависимость времени от числа процессов)
# ============================================================
def plot_scaling(G, sample_size, procs=[1,2,4,8]):
    print("\n--- Масштабирование (зависимость времени от числа процессов) ---")
    times = []
    for p in procs:
        print(f"  Процессов: {p}...", end=' ', flush=True)
        start = time.time()
        parallel_closeness_centrality(G, processes=p, sample_size=sample_size)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"{elapsed:.2f} с")
    plt.figure(figsize=(8,5))
    plt.plot(procs, times, 'bo-', label='Реальное время')
    if times[0] > 0:
        ideal = [times[0]/p for p in procs]
        plt.plot(procs, ideal, 'r--', label='Идеальное ускорение')
    plt.xlabel('Число процессов')
    plt.ylabel('Время (секунды)')
    plt.title(f'Масштабирование closeness centrality (выборка {sample_size} узлов)')
    plt.legend()
    plt.grid(True)
    plt.savefig('scaling_closeness.png')
    plt.show()
    print("  График сохранён: scaling_closeness.png")

# ============================================================
# Основная функция
# ============================================================
def main():
    print("Выберите граф:\n1 - Facebook (4k узлов)\n2 - Amazon (410k узлов)")
    choice = input().strip()
    if choice == "2":
        url = "https://snap.stanford.edu/data/amazon0505.txt.gz"
        filename = "amazon0505.txt.gz"
    else:
        url = "https://snap.stanford.edu/data/facebook_combined.txt.gz"
        filename = "facebook_combined.txt.gz"

    # Для Amazon рекомендуется выборка 200–500 узлов, иначе последовательный расчёт будет очень долгим
    SAMPLE_SIZE = 200          # можно увеличить до 500, если есть время
    PROCESSES = 4

    print("="*70)
    print("Тест closeness centrality (выборочный метод) для больших графов")
    print(f"Выборка: {SAMPLE_SIZE} узлов, процессов: {PROCESSES}")
    print("="*70)

    data_file = download_data(url, filename)
    G = load_graph(data_file)
    if G is None:
        return

    # Последовательный тест
    seq_res, seq_time, seq_top = sequential_closeness(G, SAMPLE_SIZE)

    # Параллельный тест
    print(f"\n--- Параллельная closeness ({PROCESSES} процессов, выборка {SAMPLE_SIZE} узлов) ---")
    start_par = time.time()
    par_res = parallel_closeness_centrality(G, processes=PROCESSES, sample_size=SAMPLE_SIZE)
    par_time = time.time() - start_par
    par_top = sorted(par_res.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"Время: {par_time:.2f} сек ({par_time/60:.2f} мин)")
    print("Топ-10 узлов (параллельно):")
    for i, (node, val) in enumerate(par_top, 1):
        print(f"  {i}. Узел {node}: {val:.6f}")

    # Итоги
    print("\n" + "="*50)
    print("ИТОГИ:")
    print(f"  Последовательно (выборка): {seq_time:.2f} с")
    print(f"  Параллельно (выборка):     {par_time:.2f} с")
    print(f"  Ускорение:                 {seq_time/par_time:.2f}×")
    common = len(set([n for n,_ in seq_top]) & set([n for n,_ in par_top]))
    print(f"  Совпадение топ-10 (выборка): {common}/10")

    # Визуализация
    print("\nВизуализация подграфа...")
    draw_subgraph(G, seq_top, "sequential", SAMPLE_SIZE)
    draw_subgraph(G, par_top, "parallel", SAMPLE_SIZE)

    # Масштабирование
    plot_scaling(G, SAMPLE_SIZE, procs=[1,2,4,8])

    print("\nТест завершён. Изображения сохранены в текущей папке.")

if __name__ == "__main__":
    # Необходимо для Windows multiprocessing
    mp.freeze_support()
    main()
