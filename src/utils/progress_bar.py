import sys
from tqdm import tqdm

class CustomProgressBar(tqdm):
    """
    自定义进度条类，显示格式为"===>"
    """
    def __init__(self, *args, **kwargs):
        # 设置默认参数
        kwargs.setdefault('ascii', '=')
        kwargs.setdefault('bar_format', '{l_bar}{bar}> {r_bar}')
        
        # 调用父类初始化
        super().__init__(*args, **kwargs)

def create_progress_bar(iterable=None, desc=None, total=None, unit='it', **kwargs):
    """
    创建自定义进度条的便捷函数
    
    参数:
        iterable: 可迭代对象
        desc: 进度条描述
        total: 总项目数
        unit: 单位
        **kwargs: 传递给tqdm的其他参数
    
    返回:
        CustomProgressBar实例
    """
    return CustomProgressBar(
        iterable=iterable,
        desc=desc,
        total=total,
        unit=unit,
        **kwargs
    )