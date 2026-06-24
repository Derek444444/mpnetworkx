import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import networkx as nx
import pickle
import random
import warnings
from collections import deque
from .performance_boost import get_cached_graph, get_shared_executor
# Вспомогательные функции для параллельной обработки 
def _process_closeness_chunk(chunk, graph_data, kwargs):
    """Обработка чанка для closeness centrality (глобальная, для pickle)."""
    local_G = pickle.loads(graph_data)
    res = {}
    kwargs_copy = {k: v for k, v in kwargs.items() if k != 'u'}
    for node in chunk:
        try:
            res[node] = nx.closeness_centrality(local_G, u=node, **kwargs_copy)
        except Exception:
            res[node] = 0.0
    return res

def _process_apsp_chunk(chunk, graph_data, kwargs):
    """Обработка чанка для all_pairs_dijkstra_path_length (глобальная)."""
    local_G = pickle.loads(graph_data)
    chunk_result = {}
    for source in chunk:
        chunk_result[source] = nx.single_source_dijkstra_path_length(local_G, source, **kwargs)
    return chunk_result

def _betweenness_chunk_worker(chunk_nodes, graph_data):
    """Обработка чанка для betweenness centrality (уже глобальная)."""
    local_G = pickle.loads(graph_data)
    results = {}
    for source in chunk_nodes:
        # BFS для поиска кратчайших путей (deque) 
        S = []
        P = {v: [] for v in local_G}
        sigma = {v: 0 for v in local_G}
        D = {}
        sigma[source] = 1
        D[source] = 0
        q = deque([source])
        while q:
            v = q.popleft()
            S.append(v)
            for w in local_G[v]:
                if w not in D:
                    D[w] = D[v] + 1
                    q.append(w)
                if D[w] == D[v] + 1:
                    sigma[w] += sigma[v]
                    P[w].append(v)
        # Обратный проход для накопления вкладов 
        delta = {v: 0 for v in S}
        while S:
            w = S.pop()
            coeff = (1 + delta[w]) / sigma[w] if sigma[w] != 0 else 0
            for v in P[w]:
                delta[v] += sigma[v] * coeff
            if w != source:
                results[w] = results.get(w, 0) + delta[w]
    return results

# Параллельные реализации 
def parallel_closeness_centrality(G, processes=None, sample_size=500, **kwargs):
    """Параллельная closeness centrality с выборкой узлов."""
    if processes is None:
        processes = max(2, mp.cpu_count() // 2)

    nodes = list(G.nodes())
    if sample_size is not None and sample_size < len(nodes):
        random.seed(42)                     # для воспроизводимости
        selected = random.sample(nodes, sample_size)
    else:
        selected = nodes

    chunk_size = max(1, len(selected) // processes)
    chunks = [selected[i:i+chunk_size] for i in range(0, len(selected), chunk_size)]
    graph_data = pickle.dumps(G)

    with ProcessPoolExecutor(max_workers=processes) as executor:
        futures = [executor.submit(_process_closeness_chunk, chunk, graph_data, kwargs) for chunk in chunks]
        result = {}
        for f in futures:
            result.update(f.result())
    return result

def parallel_degree_centrality_threads(G, processes=None, normalized=True):
    """Многопоточная версия degree centrality."""
    if processes is None:
        processes = mp.cpu_count()
    nodes = list(G.nodes())
    chunk_size = max(1, len(nodes) // processes)
    chunks = [nodes[i:i+chunk_size] for i in range(0, len(nodes), chunk_size)]
    n = len(G)

    def process_chunk(chunk):
        res = {}
        for node in chunk:
            deg = G.degree(node)
            if normalized and n > 1:
                deg = deg / (n - 1)
            res[node] = deg
        return res

    with ThreadPoolExecutor(max_workers=processes) as executor:
        results = executor.map(process_chunk, chunks)
    final = {}
    for r in results:
        final.update(r)
    return final

def parallel_eigenvector_centrality(G, processes=None, max_iter=100, tol=1.0e-6, normalized=True, **kwargs):
    """Параллельная (многопоточная) версия eigenvector centrality."""
    if processes is None:
        processes = mp.cpu_count()
    n = len(G)
    x = {v: 1.0 for v in G.nodes()}
    norm = sum(x.values())
    if norm > 0:
        x = {v: val / norm for v, val in x.items()}
    nodes = list(G.nodes())

    for _ in range(max_iter):
        chunk_size = max(1, len(nodes) // processes)
        chunks = [nodes[i:i+chunk_size] for i in range(0, len(nodes), chunk_size)]

        def process_chunk(chunk):
            new_vals = {}
            for v in chunk:
                s = 0.0
                for nb in G.neighbors(v):
                    s += x.get(nb, 0.0)
                new_vals[v] = s
            return new_vals

        with ThreadPoolExecutor(max_workers=processes) as executor:
            results = executor.map(process_chunk, chunks)
        x_new = {}
        for r in results:
            x_new.update(r)

        norm = sum(x_new.values())
        if norm == 0:
            break
        x_new = {v: val / norm for v, val in x_new.items()}
        diff = max(abs(x_new[v] - x.get(v, 0)) for v in nodes)
        x = x_new
        if diff < tol:
            break

    if normalized:
        return x
    else:
        return x

def parallel_pagerank(G, alpha=0.85, max_iter=100, tol=1e-6, num_threads=None, **kwargs):
    """Многопоточная версия PageRank."""
    if num_threads is None:
        num_threads = mp.cpu_count()
    n = len(G)
    pr = {v: 1.0 / n for v in G.nodes()}
    nodes = list(G.nodes())

    for _ in range(max_iter):
        chunk_size = max(1, len(nodes) // num_threads)
        chunks = [nodes[i:i+chunk_size] for i in range(0, len(nodes), chunk_size)]

        def process_chunk(chunk):
            new_vals = {}
            for v in chunk:
                s = 0.0
                for nb in G.neighbors(v):
                    s += pr[nb] / G.degree(nb)
                new_vals[v] = (1 - alpha) / n + alpha * s
            return new_vals

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = executor.map(process_chunk, chunks)
        pr_new = {}
        for r in results:
            pr_new.update(r)

        diff = max(abs(pr_new[v] - pr[v]) for v in nodes)
        pr = pr_new
        if diff < tol:
            break
    return pr

def parallel_all_pairs_dijkstra_path_length(G, processes=None, **kwargs):
    """Параллельная версия all_pairs_dijkstra_path_length."""
    kwargs.pop('sample_size', None)

    if processes is None:
        processes = max(2, mp.cpu_count() // 2)
    nodes = list(G.nodes())
    chunk_size = max(1, len(nodes) // processes)
    chunks = [nodes[i:i+chunk_size] for i in range(0, len(nodes), chunk_size)]
    graph_data = pickle.dumps(G)

    final_result = {}
    with ProcessPoolExecutor(max_workers=processes) as executor:
        futures = [executor.submit(_process_apsp_chunk, chunk, graph_data, kwargs) for chunk in chunks]
        for future in futures:
            final_result.update(future.result())
    return final_result

def parallel_betweenness_centrality(G, processes=None, sample_size=200, **kwargs):
    """Параллельная betweenness centrality с выборкой узлов."""
    if processes is None:
        processes = max(2, mp.cpu_count() // 2)

    nodes = list(G.nodes())
    if sample_size is not None and sample_size < len(nodes):
        random.seed(42)
        selected = random.sample(nodes, sample_size)
    else:
        selected = nodes

    chunk_size = max(1, len(selected) // processes)
    chunks = [selected[i:i+chunk_size] for i in range(0, len(selected), chunk_size)]
    graph_data = pickle.dumps(G)

    betweenness = {node: 0.0 for node in G.nodes()}
    with ProcessPoolExecutor(max_workers=processes) as executor:
        futures = [executor.submit(_betweenness_chunk_worker, chunk, graph_data) for chunk in chunks]
        for f in futures:
            for node, val in f.result().items():
                betweenness[node] += val

    if kwargs.get('normalized', True):
        n = len(G)
        if n > 2:
            if not G.is_directed():
                scale = 1.0 / ((n - 1) * (n - 2) / 2)
            else:
                scale = 1.0 / ((n - 1) * (n - 2))
            betweenness = {node: val * scale for node, val in betweenness.items()}
    return betweenness
