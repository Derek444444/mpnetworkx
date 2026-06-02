"""
Тест betweenness centrality с использованием mpnetworkx_simple
Сравнивает последовательную версию NetworkX  mpnetworkx_simple
"""

import urllib.request
import gzip
import networkx as nx
import matplotlib.pyplot as plt
import ssl
import time
import os
from datetime import datetime

# Импортируем библиотеку (должна быть установлена через pip)
import mpnetworkx_simple

# Применяем патчи к NetworkX
patcher = mpnetworkx_simple.apply_patches(auto_detect_threshold=100)

DATA_URL = "https://snap.stanford.edu/data/facebook_combined.txt.gz"
FILENAME = "facebook_combined.txt.gz"

def download_data():
    """Скачивание графа Facebook"""
    if os.path.exists(FILENAME):
        print(f"Файл {FILENAME} уже существует, используем его.")
        return FILENAME
    try:
        print("Скачивание Facebook графа...")
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

def load_graph(filename):
    """Загрузка графа"""
    G = nx.Graph()
    with gzip.open(filename, 'rb') as f:
        for line in f:
            if not line.startswith(b'#'):
                try:
                    u, v = map(int, line.decode().strip().split())
                    G.add_edge(u, v)
                except ValueError:
                    continue
    return G

def test_networkx_all_nodes(G):
    """Последовательный NetworkX (все узлы) с принудительным отключением параллелизации"""
    print("\n" + "="*60)
    print("ТЕСТ: Чистый NetworkX (последовательно, ВСЕ узлы)")
    print("="*60)
    start = time.time()
    # use_parallel=False заставляет использовать оригинальную последовательную реализацию
    bc = nx.betweenness_centrality(G, use_parallel=False, normalized=True)
    elapsed = time.time() - start
    print(f"Время выполнения: {elapsed:.2f} сек")
    top = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nТоп-10 узлов (NetworkX):")
    for i, (node, cent) in enumerate(top, 1):
        print(f"  {i:2}. Узел {node:4}: {cent:.6f}")
    return bc, elapsed, top

def test_mpnetworkx_parallel(G, processes=16, sample_size=None):
    """Параллельная версия через mpnetworkx_simple"""
    print("\n" + "="*60)
    print(f"ТЕСТ: mpnetworkx_simple (параллельно, {processes} процессов, sample_size={sample_size})")
    print("="*60)
    start = time.time()
    # Если sample_size = None – вычисляем на всех узлах
    bc = nx.betweenness_centrality(G, processes=processes, sample_size=sample_size, normalized=True)
    elapsed = time.time() - start
    print(f"Время выполнения: {elapsed:.2f} сек")
    top = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nТоп-10 узлов (mpnetworkx):")
    for i, (node, cent) in enumerate(top, 1):
        print(f"  {i:2}. Узел {node:4}: {cent:.6f}")
    return bc, elapsed, top

def main():
    print("="*80)
    print("Сравнение betweenness centrality: NetworkX vs mpnetworkx_simple")
    print("Граф: Facebook (4039 узлов, 88234 ребра)")
    print("="*80)

    # 1. Загрузка данных
    data_file = download_data()
    if not data_file:
        return
    G = load_graph(data_file)
    if G is None or G.number_of_nodes() == 0:
        return
    print(f"\nГраф загружен: узлов={G.number_of_nodes()}, рёбер={G.number_of_edges()}")

    # 2. Последовательный NetworkX
    nx_bc, nx_time, nx_top = test_networkx_all_nodes(G)

    # 3. Параллельная версия (16 процессов, на всех узлах – sample_size=None)
    mp_bc, mp_time, mp_top = test_mpnetworkx_parallel(G, processes=16, sample_size=None)

    # 4. Сравнение времени
    print("\n" + "="*50)
    print("ИТОГОВОЕ СРАВНЕНИЕ:")
    print(f"  NetworkX (последовательно): {nx_time:.2f} сек")
    print(f"  mpnetworkx (параллельно):   {mp_time:.2f} сек")
    print(f"  Ускорение:                  {nx_time/mp_time:.2f}×")

    # 5. Совпадение топ-10 узлов
    nx_ids = [node for node, _ in nx_top]
    mp_ids = [node for node, _ in mp_top]
    common = len(set(nx_ids) & set(mp_ids))
    print(f"\nСовпадение топ-10 узлов: {common}/10")
    if common < 10:
        print("Различающиеся узлы:")
        print(f"  Только NetworkX: {set(nx_ids) - set(mp_ids)}")
        print(f"  Только mpnetworkx: {set(mp_ids) - set(nx_ids)}")

    # Восстанавливаем оригинальный NetworkX
    mpnetworkx_simple.restore_original()
    print("\nТест завершён.")

if __name__ == "__main__":
    main()
