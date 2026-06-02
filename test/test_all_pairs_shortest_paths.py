"""
Тест all_pairs_dijkstra_path_length (APSP) с использованием mpnetworkx_simple
Сравнивает последовательную и параллельную версии, строит график масштабирования.
"""

import os
import time
import gzip
import urllib.request
import ssl
import networkx as nx
import matplotlib.pyplot as plt
import multiprocessing as mp

# Импорт вашей библиотеки
import mpnetworkx_simple

# ------------------------------------------------------------
# Загрузка графа Amazon (можно заменить на Facebook)
# ------------------------------------------------------------
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

def load_graph(filename, max_nodes=10000):
    """Загружает граф и берёт подграф из первых max_nodes узлов."""
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
    # Берём подграф для ускорения теста
    nodes = list(G.nodes())[:max_nodes]
    G_sub = G.subgraph(nodes).copy()
    print(f"Граф загружен: {G_sub.number_of_nodes():,} узлов, {G_sub.number_of_edges():,} рёбер")
    return G_sub

# ------------------------------------------------------------
# Тестовые функции (используют библиотеку mpnetworkx_simple)
# ------------------------------------------------------------
def test_sequential(G):
    """Последовательный APSP (оригинальный NetworkX)."""
    print("\n--- Последовательный all_pairs_dijkstra_path_length ---")
    start = time.time()
    # use_parallel=False – принудительно используем оригинальную версию
    paths = nx.all_pairs_dijkstra_path_length(G, use_parallel=False)
    # материализуем итератор, чтобы измерить полное время
    list(paths)
    elapsed = time.time() - start
    print(f"Время: {elapsed:.2f} сек")
    return elapsed

def test_parallel(G, processes):
    """Параллельный APSP через mpnetworkx_simple."""
    print(f"\n--- Параллельный all_pairs_dijkstra_path_length ({processes} процессов) ---")
    start = time.time()
    paths = nx.all_pairs_dijkstra_path_length(G, processes=processes)
    list(paths)
    elapsed = time.time() - start
    print(f"Время: {elapsed:.2f} сек")
    return elapsed

# ------------------------------------------------------------
# Визуализация подграфа (топ-10 узлов по степени)
# ------------------------------------------------------------
def draw_subgraph(G, method_name, max_nodes=200):
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
    plt.title(f"{method_name}\nГраф (подграф из {len(G)} узлов)\nКрасные – топ-10 по степени")
    plt.axis('off')
    outfile = f"apsp_{method_name}.png"
    plt.savefig(outfile, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Сохранено: {outfile}")

# ------------------------------------------------------------
# График масштабирования (зависимость времени от числа процессов)
# ------------------------------------------------------------
def plot_scaling(G, procs=[1, 2, 4, 8, 16]):
    print("\n--- Масштабирование all_pairs_dijkstra_path_length ---")
    times = []
    for p in procs:
        print(f"  Процессов: {p}...", end=' ', flush=True)
        start = time.time()
        # Используем параллельную версию с p процессами
        paths = nx.all_pairs_dijkstra_path_length(G, processes=p)
        list(paths)
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
    plt.title('Масштабирование all_pairs_dijkstra_path_length (Amazon, 10000 узлов)')
    plt.legend()
    plt.grid(True)
    plt.savefig('scaling_apsp.png')
    plt.show()
    print("  График сохранён: scaling_apsp.png")

# ------------------------------------------------------------
# Основная функция
# ------------------------------------------------------------
def main():
    print("=" * 70)
    print("Тест all_pairs_dijkstra_path_length (все пары кратчайших путей)")
    print("Используется подграф из первых 10000 узлов Amazon")
    print("=" * 70)

    # 1. Применяем патчи вашей библиотеки
    patcher = mpnetworkx_simple.apply_patches(auto_detect_threshold=100)

    # 2. Загружаем граф
    filename = download_amazon()
    G = load_graph(filename, max_nodes=10000)
    if G is None:
        return

    # 3. Последовательный тест
    seq_time = test_sequential(G)

    # 4. Параллельный тест (например, 4 процесса – можно изменить)
    processes = 4
    par_time = test_parallel(G, processes=processes)

    # 5. Итоги
    print("\n" + "=" * 50)
    print("ИТОГИ:")
    print(f"  Последовательно: {seq_time:.2f} с")
    print(f"  Параллельно ({processes}): {par_time:.2f} с")
    print(f"  Ускорение: {seq_time / par_time:.2f}×")

    # 6. Визуализация подграфа
    print("\nВизуализация подграфа (топ-10 по степени)...")
    draw_subgraph(G, "apsp_demo")

    # 7. График масштабирования
    plot_scaling(G, procs=[1, 2, 4, 8, 16])   # можно настроить список процессов

    # 8. Восстанавливаем оригинальный NetworkX
    patcher.restore_original()
    print("\nТест завершён. График и визуализации сохранены.")

if __name__ == "__main__":
    # Для Windows multiprocessing
    mp.freeze_support()
    main()
