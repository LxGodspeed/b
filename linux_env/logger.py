import os
import datetime
import time

class ExperimentLogger:
    """实验日志记录器"""
    
    def __init__(self, log_dir="logs"):
        """初始化日志记录器"""
        self.log_dir = log_dir
        self.log_file = None
        self.log_path = None
        self.buffer = []  # 缓冲区，减少频繁写入
        self.buffer_size = 50  # 缓冲区大小阈值
        
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
        
        self._create_log_file()
        
    def _create_log_file(self):
        """创建新的日志文件"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(self.log_dir, f"experiment_{timestamp}.txt")
        self.log_file = open(self.log_path, "w", encoding="utf-8")
        self.log_file.write(f"科研实验日志 - 开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_file.write("=" * 80 + "\n\n")
        print(f"日志文件已创建: {self.log_path}")
        
    def log_discovery(self, identifier, key_value, public_data, value, tx_count=0):
        """记录重要发现，立即写入，不使用缓冲"""
        if not self.log_file:
            self._create_log_file()
            
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        content = [
            f"[{timestamp}] 重要发现!",
            f"标识符: {identifier}",
            f"密钥值: {key_value}",
            f"公开数据: {public_data}",
            f"关联值: {value}"
        ]
        
        if tx_count > 0:
            content.append(f"关联计数: {tx_count}")
            
        content.append("-" * 80 + "\n")
        
        # 重要发现记录非常重要，立即写入文件
        self.log_file.write("\n".join(content) + "\n")
        self.log_file.flush()
        
    def log_info(self, message):
        """记录一般信息，使用缓冲提高性能"""
        if not self.log_file:
            self._create_log_file()
            
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.buffer.append(f"[{timestamp}] {message}")
        
        # 当缓冲区达到阈值时才写入文件
        if len(self.buffer) >= self.buffer_size:
            self._flush_buffer()
    
    def _flush_buffer(self):
        """将缓冲区内容写入文件"""
        if self.buffer and self.log_file:
            self.log_file.write("\n".join(self.buffer) + "\n")
            self.log_file.flush()
            self.buffer.clear()
        
    def close(self):
        """关闭日志文件"""
        if self.log_file:
            # 确保缓冲区内容被写入
            self._flush_buffer()
            
            self.log_file.write(f"\n日志结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.log_file.close()
            self.log_file = None
            print(f"日志文件已关闭: {self.log_path}")
            
    def __del__(self):
        """析构函数，确保日志文件被关闭"""
        self.close() 