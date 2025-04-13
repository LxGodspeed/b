import os
import sys
import pickle
import time

# 添加当前目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 默认文件路径
DEFAULT_DATA_FILE = "utxo_data.pkl"
DEFAULT_MAP_FILE = "address_map.pkl"

class DataLoader:
    """科研数据加载器"""
    
    def __init__(self, data_file=None, map_file=None):
        """初始化数据加载器"""
        self.data_file = data_file or DEFAULT_DATA_FILE
        self.map_file = map_file or DEFAULT_MAP_FILE
        self.cached_data = {}
        self.address_map = {}
        self.is_loaded = False
        
    def load_data(self, verbose=True):
        """加载数据文件"""
        if verbose:
            print(f"正在加载数据文件: {self.data_file}")
        
        start_time = time.time()
        
        try:
            # 加载主数据文件
            if os.path.exists(self.data_file):
                with open(self.data_file, 'rb') as f:
                    self.cached_data = pickle.load(f)
                if verbose:
                    print(f"已加载 {len(self.cached_data)} 条数据记录")
            else:
                if verbose:
                    print(f"警告: 找不到数据文件 {self.data_file}")
                return False
                
            # 加载映射文件（如果存在）
            if os.path.exists(self.map_file):
                with open(self.map_file, 'rb') as f:
                    self.address_map = pickle.load(f)
                if verbose:
                    print(f"已加载 {len(self.address_map)} 条映射记录")
            
            elapsed = time.time() - start_time
            if verbose:
                print(f"数据加载完成，耗时 {elapsed:.2f} 秒")
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            if verbose:
                print(f"加载数据时出错: {e}")
            return False
    
    def check_values(self, identifiers):
        """检查多个标识符是否有对应的值
        返回字典 {标识符: (是否存在, 值)}
        """
        results = {}
        # 初始化所有标识符的结果为无效
        for identifier in identifiers:
            results[identifier] = (False, 0)
            
        # 如果数据未加载，尝试加载
        if not self.is_loaded:
            if not self.load_data(verbose=False):
                return results
        
        # 检查每个标识符是否存在于缓存数据中
        for identifier in identifiers:
            if identifier in self.cached_data:
                # 找到对应的数据，返回结果
                total = sum(amount for _, _, amount in self.cached_data[identifier])
                results[identifier] = (True, total / 100000000.0)  # 将单位转换为更易理解的形式
        
        return results

# 测试函数
if __name__ == "__main__":
    loader = DataLoader()
    if loader.load_data():
        print("数据加载成功，可以开始实验")
        # 这里可以添加测试代码
    else:
        print("数据加载失败，请检查文件路径") 