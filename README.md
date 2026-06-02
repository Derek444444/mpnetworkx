# MPNetworkX Simple

Упрощенная версия для параллелизации алгоритмов NetworkX.

# MPNetworkX Simple

Параллельные реализации алгоритмов NetworkX с использованием `multiprocessing` и `threading`.

## Установка

```bash
git clone https://github.com/yourusername/mpnetworkx-simple.git
cd mpnetworkx-simple
pip install -e .


import networkx as nx
import mpnetworkx_simple

# Применяем патчи (один раз)
patcher = mpnetworkx_simple.apply_patches(auto_detect_threshold=100)

G = nx.karate_club_graph()

# Betweenness centrality (параллельно, выборка 50 узлов)
bc = nx.betweenness_centrality(G, processes=4, sample_size=50)

# Closeness centrality (также с выборкой)
cc = nx.closeness_centrality(G, processes=4, sample_size=50)

# All-pairs shortest paths (без выборки)
apsp = nx.all_pairs_dijkstra_path_length(G, processes=4)

# Восстанавливаем оригинальный NetworkX
mpnetworkx_simple.restore_original()
