# Package initialization file
try:
    from .utils import get_stock_data, calculate_metrics, timer_decorator, ensure_dir_exists
    
    __all__ = [
        'get_stock_data',
        'calculate_metrics',
        'timer_decorator',
        'ensure_dir_exists'
    ]
except ImportError:
    # This allows the module to be imported even if all dependencies are not available
    pass
