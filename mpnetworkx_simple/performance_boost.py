"""
performance_boost.py
=============================================
Модуль, объединяющий два улучшения производительности:

1. Кэш сериализации – граф сериализуется один раз,
   а не при каждом вызове функции.
   ЗАЩИТА: инвалидация кэша при изменении графа.

2. Переиспользуемый пул процессов – не создаётся заново
   для каждой метрики, что экономит время на запуск.
   ЗАЩИТА: проверка живучести воркеров, пересоздание при падении.

Всё это работает прозрачно и даёт преимущество перед joblib.
"""

import pickle
import os
import sys
import atexit
import warnings
from concurrent.futures import ProcessPoolExecutor

# ------------------------------------------------------------
#  1. Кэш сериализованных графов (с защитой от изменений)
# ------------------------------------------------------------
class GraphCache:
    """
    Кэш для сериализованных графов.
    
    ЗАЩИТА:
        - Кэш инвалидируется, если изменилась структура графа
          (размер, число рёбер, ориентированность).
        - Используется блокировка для потокобезопасности.
    """
    def __init__(self):
        self._cache = {}          # key -> байты
        self._version = {}        # key -> сигнатура графа
        self._lock = threading.Lock()
        self._ref_count = {}      # key -> счётчик обращений (для отладки)

    def _get_graph_signature(self, G):
        """
        Быстрая проверка структуры графа.
        Если изменилось что-то из этого, кэш считается устаревшим.
        """
        return (
            G.number_of_nodes(),
            G.number_of_edges(),
            G.is_directed(),
            G.is_multigraph()
        )

    def get_serialized(self, G, force=False):
        """
        Возвращает сериализованный граф в виде байтов.
        Если граф уже есть в кэше и не изменился – возвращает его.
        """
        key = id(G)
        current_sig = self._get_graph_signature(G)
        
        with self._lock:
            # Проверяем, изменился ли граф
            if key in self._cache and self._version.get(key) != current_sig:
                # Инвалидируем кэш
                del self._cache[key]
                del self._version[key]
                del self._ref_count[key]
                warnings.warn(f"Кэш инвалидирован: граф изменился (id={key})")
                force = True
            
            if key not in self._cache or force:
                # Сериализуем с максимальной производительностью
                print(f"Сериализация графа ({G.number_of_nodes()} узлов) для кэша...")
                self._cache[key] = pickle.dumps(G, protocol=pickle.HIGHEST_PROTOCOL)
                self._version[key] = current_sig
                self._ref_count[key] = 0
                print(f"Граф закэширован. Размер: {len(self._cache[key]) / 1024 / 1024:.2f} МБ")
            
            self._ref_count[key] += 1
            return self._cache[key]

    def get_size(self, G):
        """Возвращает размер сериализованного графа в байтах."""
        return len(self.get_serialized(G))

    def clear(self):
        """Очищает кэш (освобождает память)."""
        with self._lock:
            self._cache.clear()
            self._version.clear()
            self._ref_count.clear()

    def stats(self):
        """Возвращает статистику использования кэша."""
        with self._lock:
            return {
                'total_graphs': len(self._cache),
                'total_bytes': sum(len(v) for v in self._cache.values()),
                'ref_counts': dict(self._ref_count)
            }

# ------------------------------------------------------------
#  2. Переиспользуемый пул процессов (с проверкой живучести)
# ------------------------------------------------------------
class SharedExecutor:
    """
    Пул процессов, который создаётся один раз и используется для всех вызовов.
    
    ЗАЩИТА:
        - Проверяет, живы ли воркеры; пересоздаёт пул, если есть мёртвые.
        - Не даёт использовать пул, если процессы умерли из-за SEGFAULT.
    """
    def __init__(self, max_workers=None):
        self.max_workers = max_workers
        self._executor = None
        self._lock = threading.Lock()

    def get_executor(self, processes=None):
        """
        Возвращает экземпляр ProcessPoolExecutor.
        Если пул повреждён, создаётся новый.
        """
        workers = processes or self.max_workers
        if workers is None:
            import multiprocessing as mp
            workers = mp.cpu_count()
        
        with self._lock:
            # Проверяем здоровье существующего пула
            if self._executor is not None:
                try:
                    # Проверяем, сколько воркеров живы
                    # (используем внутреннее API, но оно стабильно в Python 3.7+)
                    if hasattr(self._executor, '_processes'):
                        alive_count = sum(1 for p in self._executor._processes.values() if p.is_alive())
                        total_count = len(self._executor._processes)
                        
                        if alive_count < total_count:
                            warnings.warn(f" Обнаружены мёртвые воркеры ({alive_count}/{total_count}). Пересоздаём пул...")
                            self.shutdown(wait=True)
                            self._executor = None
                except Exception:
                    # Если что-то пошло не так, пересоздаём
                    self.shutdown(wait=True)
                    self._executor = None
            
            # Создаём новый пул, если нужно
            if self._executor is None or self._executor._max_workers != workers:
                if self._executor is not None:
                    self.shutdown(wait=True)
                
                print(f"Создаётся пул процессов с {workers} воркерами...")
                self._executor = ProcessPoolExecutor(max_workers=workers)
                print("Пул создан")
            
            return self._executor

    def shutdown(self, wait=True):
        """Закрывает пул процессов."""
        with self._lock:
            if self._executor is not None:
                self._executor.shutdown(wait=wait)
                self._executor = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown(wait=True)

# ------------------------------------------------------------
#  3. Глобальные экземпляры для удобства
# ------------------------------------------------------------
import threading

_global_cache = GraphCache()
_global_executor = SharedExecutor()

# Блокировка для потокобезопасности
_global_lock = threading.Lock()

def get_cached_graph(G):
    """
    Удобная функция для получения сериализованного графа из глобального кэша.
    """
    with _global_lock:
        return _global_cache.get_serialized(G)

def get_shared_executor(processes=None):
    """
    Удобная функция для получения переиспользуемого пула процессов.
    """
    with _global_lock:
        return _global_executor.get_executor(processes)

def clear_cache():
    """Очищает глобальный кэш сериализации."""
    with _global_lock:
        _global_cache.clear()
        print("🧹 Кэш очищен")

def shutdown_executor():
    """Завершает работу глобального пула процессов."""
    with _global_lock:
        _global_executor.shutdown(wait=True)
        print("🧹 Пул процессов завершён")

def performance_stats():
    """Возвращает статистику производительности (для отладки)."""
    return {
        'cache': _global_cache.stats(),
        'executor_alive': _global_executor._executor is not None
    }

# Автоматическая очистка при завершении программы
atexit.register(shutdown_executor)

# ------------------------------------------------------------
#  Для тестирования
# ------------------------------------------------------------
if __name__ == "__main__":
    print("Модуль performance_boost загружен")
    print("Кэш и пул процессов готовы к работе")
