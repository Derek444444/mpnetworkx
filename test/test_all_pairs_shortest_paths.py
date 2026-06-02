"""
test_apsp.py – Тест all_pairs_dijkstra_path_length (многопроцессная версия)
Ручной патч, исправлена сериализация, добавлена визуализация.
"""

import sys
import os
import urllib.request
import gzip
import networkx as nx
import matplotlib.pyplot as plt
import ssl
import time
import multiprocessing as mp
import pickle
from concurrent.futures import ProcessPoolExecutor

# ==================== Глобальная функция для чанка ====================
def _process_chunk_apsp(chunk, graph_data, **kwargs):
    local_G = pickle.loads(graph_data)
    chunk_result = {}
    for source in chunk:
        chunk_result[source] = nx.single_source_dijkstra_path_length(local_G, source, **kwargs)
    return chunk_result

# ==================== Ручной патч ====================
def parallel_all_pairs_dijkstra_path_length(G, processes=None, **kwargs):
    if processes is None:
        processes = max(2, mp.cpu_count() // 2)
    nodes = list(G.nodes())
    chunk_size = max(1, len(nodes) // processes)
    chunks = [nodes[i:i+chunk_size] for i in range(0, len(nodes), chunk_size)]
    graph_data = pickle.dumps(G)

    with ProcessPoolExecutor(max_workers=processes) as executor:
        futures = [executor.submit(_process_chunk_apsp, chunk, graph_data, **kwargs) for chunk in chunks]
        final_result = {}
        for future in futures:
            final_result.update(future.result())
    return final_result

# Заменяем оригинальную функцию
nx.all_pairs_dijkstra_path_length = parallel_all_pairs_dijkstra_path_length
print("Патч для all_pairs_dijkstra_path_length применён вручную")

# ==================== Загрузка данных ====================
def download_amazon():
    url = "https://snap.stanford.edu/data/amazon0505.txt.gz"
    filename = "amazon0505.txt.gz"
    if not os.path.exists(filename):
        print("Скачивание Amazon (250 МБ)...")
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=ctx) as resp:
            with open(filename, 'wb') as f:
                f.write(resp.read())
    return filename

def load_graph(filename):
    G = nx.Graph()
    with gzip.open(filename, 'rb') as f:
        for line in f:
            if line.startswith(b'#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = map(int, parts[:2])
            G.add_edge(u, v)
    print(f"Граф: {G.number_of_nodes():,} узлов, {G.number_of_edges():,} рёбер")
    return G

# ==================== Тесты ====================
def test_sequential(G_sub):
    print("\n--- Последовательный all_pairs_dijkstra_path_length ---")
    sources = list(G_sub.nodes())
    start = time.time()
    for source in sources:
        dist = nx.single_source_dijkstra_path_length(G_sub, source)
        for d in dist.values():
            pass
    elapsed = time.time() - start
    print(f"Обработано {len(sources)} источников, время: {elapsed:.2f} сек")
    return elapsed

def test_parallel(G_sub, processes=4):
    print(f"\n--- Параллельный all_pairs_dijkstra_path_length ({processes} процессов) ---")
    start = time.time()
    result = nx.all_pairs_dijkstra_path_length(G_sub, processes=processes)
    list(result)  # материализуем
    elapsed = time.time() - start
    print(f"Время: {elapsed:.2f} сек")
    return elapsed

# ==================== Визуализация подграфа ====================
def draw_subgraph(G, top_nodes, method_name, max_nodes=200):
    # Для all_pairs_dijkstra_path_length мы не вычисляем центральность,
    # поэтому просто покажем фрагмент графа с топ-10 узлами по степени (как пример)
    # Но можно вычислить степень и взять топ-10 по степени.
    degree = dict(G.degree())
    top_degree = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:10]
    top_set = {node for node, _ in top_degree}
    neighbors = set()
    for node in top_set:
        neighbors.update(G.neighbors(node))
    sub_nodes = list(top_set.union(neighbors))[:max_nodes]
    H = G.subgraph(sub_nodes)
    pos = nx.spring_layout(H, k=0.8, iterations=30, seed=42)
    plt.figure(figsize=(12, 10))
    colors = ['#e74c3c' if n in top_set else '#3498db' for n in H.nodes()]
    sizes = [300 if n in top_set else 50 for n in H.nodes()]
    nx.draw_networkx_edges(H, pos, alpha=0.2, edge_color='gray')
    nx.draw_networkx_nodes(H, pos, node_size=sizes, node_color=colors, alpha=0.7)
    labels = {n: str(n) for n in H.nodes() if n in top_set}
    nx.draw_networkx_labels(H, pos, labels, font_size=8, font_weight='bold')
    plt.title(f"{method_name}\nГраф Amazon (подграф {len(G)} узлов)\nКрасные – топ-10 по степени")
    plt.axis('off')
    outfile = f"apsp_{method_name}.png"
    plt.savefig(outfile, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Сохранено: {outfile}")

# ==================== График масштабирования ====================
def plot_scaling(G_sub, procs=[1,2,4,8]):
    print("\n--- Масштабирование all_pairs_dijkstra_path_length ---")
    times = []
    for p in procs:
        print(f"  Процессов: {p}...", end=' ', flush=True)
        start = time.time()
        list(nx.all_pairs_dijkstra_path_length(G_sub, processes=p))
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"{elapsed:.2f} с")
    plt.figure(figsize=(8,5))
    plt.plot(procs, times, 'bo-', label='Реальное время')
    if times[0] > 0:
        ideal = [times[0] / p for p in procs]
        plt.plot(procs, ideal, 'r--', label='Идеальное ускорение')
    plt.xlabel('Число процессов')
    plt.ylabel('Время (секунды)')
    plt.title('Масштабирование all_pairs_dijkstra_path_length (Amazon, 10000 узлов)')
    plt.legend()
    plt.grid(True)
    plt.savefig('scaling_apsp.png')
    plt.show()
    print("  График сохранён: scaling_apsp.png")

# ==================== Основная функция ====================
def main():
    print("="*70)
    print("Тест all_pairs_dijkstra_path_length (все пары кратчайших путей)")
    print("Используется подграф из первых 10000 узлов Amazon")
    print("="*70)

    filename = download_amazon()
    G = load_graph(filename)
    if G is None:
        return

    N = 10000
    sub_nodes = list(G.nodes())[:N]
    G_sub = G.subgraph(sub_nodes).copy()
    print(f"Подграф: {len(G_sub)} узлов, {len(G_sub.edges())} рёбер")

    seq_time = test_sequential(G_sub)
    par_time = test_parallel(G_sub, processes=4)

    print("\n" + "="*50)
    print("ИТОГИ:")
    print(f"  Последовательно: {seq_time:.2f} с")
    print(f"  Параллельно:     {par_time:.2f} с")
    print(f"  Ускорение:       {seq_time/par_time:.2f}×")

    # Визуализация подграфа
    print("\nВизуализация подграфа (топ-10 по степени)...")
    draw_subgraph(G_sub, None, "sequential")   # передаём заглушку вместо top_nodes
    # Для параллельной версии подграф тот же, можно не рисовать заново, но для единообразия:
    draw_subgraph(G_sub, None, "parallel")

    plot_scaling(G_sub, procs=[1,2,4,8])

    print("\nТест завершён. График и визуализации сохранены.")

if __name__ == "__main__":
    mp.freeze_support()
    main()
