import networkx as nx
import time
import logging
import warnings
from functools import wraps
from typing import Callable, Dict, Any
from concurrent.futures import TimeoutError as FutureTimeoutError

from .parallel_implementations import (
    parallel_betweenness_centrality,
    parallel_closeness_centrality,
    parallel_degree_centrality_threads,
    parallel_eigenvector_centrality,
    parallel_pagerank,
    parallel_all_pairs_dijkstra_path_length
)

class NetworkXPatcher:
    
    def __init__(self, auto_detect_threshold: int = 1000):
        self.auto_detect_threshold = auto_detect_threshold
        self.original_functions = {}
        self.performance_stats = {}
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - mpnetworkx - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger('mpnetworkx')
        self.logger.info(f"Патчер инициализирован с порогом: {self.auto_detect_threshold}")

    def patch_function(self, original_func: Callable, parallel_func: Callable, func_name: str) -> Callable:
        @wraps(original_func)
        def smart_wrapper(G, *args, **kwargs):
            processes = kwargs.pop('processes', None)
            force_parallel = kwargs.pop('force_parallel', False)
            benchmark = kwargs.pop('benchmark', False)
            use_parallel = kwargs.pop('use_parallel', True)
            sample_size = kwargs.pop('sample_size', 200)
            timeout = kwargs.pop('timeout', 300)

            should_use_parallel = (
                use_parallel and
                (force_parallel or (processes and processes > 1) or len(G) >= self.auto_detect_threshold)
            )

            if should_use_parallel:
                return self._execute_parallel(
                    G, parallel_func, original_func, func_name,
                    processes, sample_size, benchmark, timeout, *args, **kwargs
                )
            else:
                return self._execute_sequential(
                    G, original_func, func_name, *args, **kwargs
                )
        return smart_wrapper

    def _execute_parallel(self, G, parallel_func, original_func, func_name,
                         processes, sample_size, benchmark, timeout, *args, **kwargs):
        self.logger.info(f"Параллельная {func_name} (граф: {len(G)} узлов, процессы: {processes}, sample_size: {sample_size})")
        start_time = time.time()
        try:
            result = parallel_func(G, processes=processes, sample_size=sample_size, *args, **kwargs)
            parallel_time = time.time() - start_time
            if benchmark:
                seq_time = self._benchmark_sequential(G, original_func, func_name, *args, **kwargs)
                speedup = seq_time / parallel_time if parallel_time > 0 else 0
                self.logger.info(f"Ускорение: {speedup:.2f}×")
                self._update_stats(func_name, 'parallel', parallel_time, speedup)
            return result
        except FutureTimeoutError:
            self.logger.warning("Таймаут, переход к последовательной версии")
            return original_func(G, *args, **kwargs)
        except Exception as e:
            self.logger.error(f"Ошибка: {str(e)[:100]}")
            warnings.warn(f"Параллельное выполнение не удалось: {str(e)[:100]}")
            return original_func(G, *args, **kwargs)

    def _execute_sequential(self, G, original_func, func_name, *args, **kwargs):
        self.logger.info(f"🐌 Последовательная {func_name} (граф: {len(G)} узлов)")
        start_time = time.time()
        result = original_func(G, *args, **kwargs)
        execution_time = time.time() - start_time
        self._update_stats(func_name, 'sequential', execution_time)
        return result

    def _benchmark_sequential(self, G, original_func, func_name, *args, **kwargs):
        self.logger.info("⏱ Бенчмаркинг последовательной версии...")
        start_time = time.time()
        original_func(G, *args, **kwargs)
        return time.time() - start_time

    def _update_stats(self, func_name: str, mode: str, time_taken: float, speedup: float = None):
        if func_name not in self.performance_stats:
            self.performance_stats[func_name] = []
        self.performance_stats[func_name].append({
            'mode': mode,
            'time': time_taken,
            'speedup': speedup,
            'timestamp': time.time()
        })

    def apply_patches(self):
        self.logger.info(" Применение патчей...")

        self.original_functions['betweenness_centrality'] = nx.betweenness_centrality
        nx.betweenness_centrality = self.patch_function(
            self.original_functions['betweenness_centrality'],
            parallel_betweenness_centrality,
            'betweenness_centrality'
        )

        self.original_functions['closeness_centrality'] = nx.closeness_centrality
        nx.closeness_centrality = self.patch_function(
            self.original_functions['closeness_centrality'],
            parallel_closeness_centrality,
            'closeness_centrality'
        )

        self.original_functions['degree_centrality'] = nx.degree_centrality
        nx.degree_centrality = self.patch_function(
            self.original_functions['degree_centrality'],
            parallel_degree_centrality_threads,
            'degree_centrality'
        )

        self.original_functions['eigenvector_centrality'] = nx.eigenvector_centrality
        nx.eigenvector_centrality = self.patch_function(
            self.original_functions['eigenvector_centrality'],
            parallel_eigenvector_centrality,
            'eigenvector_centrality'
        )

        self.original_functions['pagerank'] = nx.pagerank
        nx.pagerank = self.patch_function(
            self.original_functions['pagerank'],
            parallel_pagerank,
            'pagerank'
        )

        self.original_functions['all_pairs_dijkstra_path_length'] = nx.all_pairs_dijkstra_path_length
        nx.all_pairs_dijkstra_path_length = self.patch_function(
            self.original_functions['all_pairs_dijkstra_path_length'],
            parallel_all_pairs_dijkstra_path_length,
            'all_pairs_dijkstra_path_length'
        )

        self.logger.info("ок Патчи применены успешно")
        
    def restore_original(self):
        for func_name, original_func in self.original_functions.items():
            setattr(nx, func_name, original_func)
        self.logger.info(" Оригинальные функции восстановлены")

    def get_performance_report(self) -> Dict[str, Any]:
        report = {
            'total_calls': {},
            'average_times': {},
            'speedup_stats': {}
        }
        for func_name, stats in self.performance_stats.items():
            report['total_calls'][func_name] = len(stats)
            parallel_times = [s['time'] for s in stats if s['mode'] == 'parallel']
            seq_times = [s['time'] for s in stats if s['mode'] == 'sequential']
            speedups = [s['speedup'] for s in stats if s['speedup'] is not None]
            if parallel_times:
                report['average_times'][f'{func_name}_parallel'] = sum(parallel_times) / len(parallel_times)
            if seq_times:
                report['average_times'][f'{func_name}_sequential'] = sum(seq_times) / len(seq_times)
            if speedups:
                report['speedup_stats'][func_name] = {
                    'average': sum(speedups) / len(speedups),
                    'max': max(speedups),
                    'min': min(speedups)
                }
        return report
