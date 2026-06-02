"""
mpnetworkx для оптимизации NetworkX
"""

from .networkx_patcher import NetworkXPatcher

# Глобальный экземпляр патчера
_patcher = None

def apply_patches(auto_detect_threshold=1000):
    """
    Применение патчей к NetworkX
    
    Args:
        auto_detect_threshold: Минимальный размер графа для автоматической 
                              параллелизации
    
    Returns:
        NetworkXPatcher: Экземпляр патчера
    """
    global _patcher
    _patcher = NetworkXPatcher(auto_detect_threshold)
    _patcher.apply_patches()  # Исправлено: вызывает метод apply_patches() экземпляра
    print(f" mpnetworkx v1.0.0 активирован (порог: {auto_detect_threshold} узлов)")
    return _patcher

def restore_original():
    """Восстановление оригинального NetworkX"""
    global _patcher
    if _patcher:
        _patcher.restore_original()
        print(" Оригинальные функции NetworkX восстановлены")
def get_patcher():
    """Получение экземпляра патчера"""
    global _patcher
    return _patcher

# Импортируем параллельные функции для прямого доступа
from .parallel_implementations import parallel_betweenness_centrality

__version__ = "1.0.0"
__author__ = "mpnetworkx Team"


