import numpy as np
import hashlib
import base58
import time
import random
import os
import signal
import sys
import binascii
import struct
from typing import List, Tuple, Dict, Set, Optional

# 导入本地模块
from linux_env.data_loader import DataLoader
from linux_env.logger import ExperimentLogger
from linux_env.notifier import ExperimentNotifier

# 尝试导入CuPy
CUDA_AVAILABLE = False
cp = None

# GPU优化配置
CUDA_BLOCK_SIZE = 512  # CUDA块大小
DEFAULT_BATCH_SIZE = 1024000  # 批处理大小
USE_PINNED_MEMORY = True  # 启用固定内存加速
NUM_CUDA_STREAMS = 8  # CUDA流数量

def setup_cupy():
    """设置CuPy"""
    global CUDA_AVAILABLE, cp
    
    # 直接尝试导入CuPy
    try:
        import cupy as cp
        CUDA_AVAILABLE = True
        print("成功导入CuPy")
        
        # 设置内存池
        cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
        if USE_PINNED_MEMORY:
            cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)
        
    except ImportError:
        print("警告: 未安装CuPy库，将使用CPU模式")
        print("如需GPU加速，请安装CuPy: pip install cupy-cuda12x")
        CUDA_AVAILABLE = False
        cp = None

# 初始化CuPy
setup_cupy()

# 设置信号处理，以便在Ctrl+C时优雅退出
def signal_handler(sig, frame):
    print("\n检测到中断信号，正在优雅退出...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

class ComputeEngine:
    """高性能科学计算引擎"""
    
    def __init__(self, target_identifiers=None, data_file=None, use_gpu=True, notification_token=None):
        """
        初始化计算引擎
        
        参数:
        target_identifiers: 目标标识符列表
        data_file: 数据文件路径
        use_gpu: 是否使用GPU加速
        notification_token: 通知服务令牌
        """
        self.use_gpu = False
        self.target_identifiers = set(target_identifiers) if target_identifiers else set()
        self.discovered_results = {}  # 标识符 -> 密钥映射
        self.total_processed = 0
        self.batch_size = DEFAULT_BATCH_SIZE
        self.start_time = time.time()
        self.running = True  # 控制运行状态
        
        # 性能监控
        self.performance_stats = {
            'total_time': 0,
            'gpu_time': 0,
            'cpu_time': 0,
            'total_keys': 0,
            'gpu_keys': 0,
            'cpu_keys': 0,
            'last_report_time': time.time(),
            'report_interval': 10  # 每10秒报告一次性能
        }
        
        # 初始化日志记录器
        self.logger = ExperimentLogger()
        
        # 初始化通知服务
        self.notifier = ExperimentNotifier(notification_token)
        if self.notifier.enabled:
            self.logger.log_info(f"通知服务已启用")
        
        # 初始化数据加载器
        self.data_loader = DataLoader(data_file)
        self.data_loader.load_data()
        
        # 设置GPU
        if use_gpu:
            self._setup_gpu()
    
    def _setup_gpu(self):
        """初始化GPU相关资源"""
        # 检查GPU是否可用
        if not CUDA_AVAILABLE:
            print("CuPy未安装，将使用CPU进行计算")
            print("如需GPU加速，请安装CuPy: pip install cupy-cuda12x")
            self.device = None
            self.use_gpu = False
            return
            
        try:
            # 首先检查CUDA是否可用
            if not cp.cuda.is_available():
                raise RuntimeError("CUDA不可用，请确保已安装NVIDIA驱动和CUDA工具包")
                
            self.device = cp.cuda.Device()  # 使用默认设备
            self.device.use()
            
            # 获取GPU信息
            gpu_info = cp.cuda.runtime.getDeviceProperties(self.device.id)
            print(f"检测到GPU: {gpu_info['name'].decode('utf-8')}")
            print(f"CUDA计算能力: {gpu_info['major']}.{gpu_info['minor']}")
            print(f"GPU总内存: {gpu_info['totalGlobalMem'] / 1024**3:.2f} GB")
            print(f"多处理器数量: {gpu_info['multiProcessorCount']}")
            
            # 设置适合此GPU的工作大小
            max_threads = 256 * gpu_info['multiProcessorCount']
            recommended_batch = max_threads * 50  # 每个线程处理多个项目
            self.batch_size = min(1000000, recommended_batch)  # 限制上限
            print(f"为当前GPU优化的批处理大小: {self.batch_size}")
            
            # 预热GPU
            print("正在预热GPU...")
            matrix_size = 2000
            a = cp.random.rand(matrix_size, matrix_size).astype(cp.float32)
            b = cp.random.rand(matrix_size, matrix_size).astype(cp.float32)
            for _ in range(5):
                c = cp.matmul(a, b)
                a = c / cp.max(c)
            
            self.use_gpu = True
            print(f"使用GPU: CUDA Device {self.device.id}")
            
            # 测试GPU功能
            try:
                # 创建一个简单的测试数组
                test_array = cp.array([1, 2, 3])
                _ = test_array + 1
                print("GPU功能测试成功")
                
                # 预编译CUDA核函数
                if self.use_gpu:
                    self._compile_kernels()
            except Exception as e:
                print(f"GPU功能测试失败: {e}")
                print("将使用CPU进行计算")
                self.use_gpu = False
                
        except Exception as e:
            print(f"GPU初始化失败: {e}")
            print("可能的原因:")
            print("1. 未安装NVIDIA驱动")
            print("2. 未安装CUDA工具包")
            print("3. CUDA版本与CuPy不匹配")
            print("4. 环境变量PATH中未包含CUDA路径")
            print("\n将使用CPU进行计算")
            self.device = None
            self.use_gpu = False
    
    def _compile_kernels(self):
        """编译CUDA核函数"""
        if not self.use_gpu:
            return
            
        # 哈希函数1核函数
        hash1_kernel_code = r'''
        extern "C" __global__
        void hash1_kernel(const unsigned char* input, unsigned char* output, 
                          const int batch_size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= batch_size) return;
            
            // 每个线程处理32字节的输入
            const unsigned char* in = input + idx * 32;
            unsigned char* out = output + idx * 32;
            
            // 哈希计算的初始哈希值
            unsigned int hash[8] = {
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
            };
            
            // 轮常量
            const unsigned int k[64] = {
                0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
                0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
                0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
                0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
                0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
                0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
                0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
                0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
                0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
                0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
                0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
                0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
                0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
                0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
                0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
                0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
            };
            
            // 消息调度数组
            unsigned int w[64];
            unsigned int a, b, c, d, e, f, g, h, i, t1, t2;
            
            // 预处理消息 - 使用向量加载提高性能
            #pragma unroll
            for (i = 0; i < 8; i++) {
                unsigned int tmp = *((unsigned int*)(in + i*4));
                w[i] = __byte_perm(tmp, 0, 0x0123);
            }
            
            // 填充剩余部分
            w[8] = 0x80000000;
            #pragma unroll
            for (i = 9; i < 15; i++) {
                w[i] = 0;
            }
            w[15] = 256;  // 消息长度（比特）
            
            // 扩展消息调度 - 使用位操作优化
            #pragma unroll
            for (i = 16; i < 64; i++) {
                unsigned int s0 = (__byte_perm(w[i-15], 0, 0x0321) ^ 
                                 __byte_perm(w[i-15], 0, 0x0210) ^ 
                                 (w[i-15] >> 3));
                unsigned int s1 = (__byte_perm(w[i-2], 0, 0x0321) ^ 
                                 __byte_perm(w[i-2], 0, 0x0210) ^ 
                                 (w[i-2] >> 10));
                w[i] = w[i-16] + s0 + w[i-7] + s1;
            }
            
            // 工作变量
            a = hash[0];
            b = hash[1];
            c = hash[2];
            d = hash[3];
            e = hash[4];
            f = hash[5];
            g = hash[6];
            h = hash[7];
            
            // 主循环 - 使用位操作优化
            #pragma unroll
            for (i = 0; i < 64; i++) {
                t1 = h + ((__byte_perm(e, 0, 0x0321) ^ 
                          __byte_perm(e, 0, 0x0210) ^ 
                          (e >> 6)) + ((e & f) ^ (~e & g)) + k[i] + w[i]);
                t2 = ((__byte_perm(a, 0, 0x0321) ^ 
                      __byte_perm(a, 0, 0x0210) ^ 
                      (a >> 2)) + ((a & b) ^ (a & c) ^ (b & c)));
                
                h = g;
                g = f;
                f = e;
                e = d + t1;
                d = c;
                c = b;
                b = a;
                a = t1 + t2;
            }
            
            // 更新哈希值
            hash[0] += a;
            hash[1] += b;
            hash[2] += c;
            hash[3] += d;
            hash[4] += e;
            hash[5] += f;
            hash[6] += g;
            hash[7] += h;
            
            // 写入输出 - 使用向量存储提高性能
            #pragma unroll
            for (i = 0; i < 8; i++) {
                unsigned int tmp = hash[i];
                *((unsigned int*)(out + i*4)) = __byte_perm(tmp, 0, 0x0123);
            }
        }
        '''
        
        # 哈希函数2核函数
        hash2_kernel_code = r'''
        __device__ inline unsigned int ROTL(unsigned int x, unsigned int n) {
            return (x << n) | (x >> (32 - n));
        }

        __device__ inline unsigned int f1(unsigned int x, unsigned int y, unsigned int z) {
            return x ^ y ^ z;
        }

        __device__ inline unsigned int f2(unsigned int x, unsigned int y, unsigned int z) {
            return (x & y) | (~x & z);
        }

        __device__ inline unsigned int f3(unsigned int x, unsigned int y, unsigned int z) {
            return (x | ~y) ^ z;
        }

        __device__ inline unsigned int f4(unsigned int x, unsigned int y, unsigned int z) {
            return (x & z) | (y & ~z);
        }

        __device__ inline unsigned int f5(unsigned int x, unsigned int y, unsigned int z) {
            return x ^ (y | ~z);
        }

        extern "C" __global__
        void hash2_kernel(const unsigned char* input, unsigned char* output,
                            const int batch_size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= batch_size) return;
            
            // 每个线程处理32字节的输入
            const unsigned char* in = input + idx * 32;
            unsigned char* out = output + idx * 20;
            
            // 初始哈希值
            unsigned int h0 = 0x67452301;
            unsigned int h1 = 0xEFCDAB89;
            unsigned int h2 = 0x98BADCFE;
            unsigned int h3 = 0x10325476;
            unsigned int h4 = 0xC3D2E1F0;
            
            // 常量
            const unsigned int K1 = 0x5A827999;
            const unsigned int K2 = 0x6ED9EBA1;
            const unsigned int K3 = 0x8F1BBCDC;
            const unsigned int K4 = 0xA953FD4E;
            
            // 消息块 - 使用向量加载提高性能
            unsigned int X[16];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                unsigned int tmp = *((unsigned int*)(in + i*4));
                X[i] = __byte_perm(tmp, 0, 0x0123);
            }
            
            // 填充
            X[8] = 0x80000000;
            #pragma unroll
            for (int i = 9; i < 15; i++) {
                X[i] = 0;
            }
            X[15] = 256;  // 消息长度（比特）
            
            unsigned int a1 = h0;
            unsigned int b1 = h1;
            unsigned int c1 = h2;
            unsigned int d1 = h3;
            unsigned int e1 = h4;
            unsigned int a2 = h0;
            unsigned int b2 = h1;
            unsigned int c2 = h2;
            unsigned int d2 = h3;
            unsigned int e2 = h4;
            unsigned int t;
            
            // 主循环 - 使用位操作优化
            #pragma unroll
            for (int j = 0; j < 80; j++) {
                t = ROTL(a1 + X[j & 0x0f], j < 16 ? j + 7 : 
                                                    j < 32 ? (j + 4) & 0x1f : 
                                                    j < 48 ? (j + 13) & 0x1f : 
                                                    j < 64 ? (j + 6) & 0x1f : 
                                                            (j + 15) & 0x1f);
                if (j < 16)
                    t += f1(b1, c1, d1) + K1;
                else if (j < 32)
                    t += f2(b1, c1, d1) + K2;
                else if (j < 48)
                    t += f3(b1, c1, d1) + K3;
                else if (j < 64)
                    t += f4(b1, c1, d1) + K4;
                else
                    t += f5(b1, c1, d1) + K1;
                
                a1 = e1;
                e1 = d1;
                d1 = ROTL(c1, 10);
                c1 = b1;
                b1 = t;
                
                t = ROTL(a2 + X[(80-j-1) & 0x0f], j < 16 ? j + 7 : 
                                                           j < 32 ? (j + 4) & 0x1f : 
                                                           j < 48 ? (j + 13) & 0x1f : 
                                                           j < 64 ? (j + 6) & 0x1f : 
                                                                   (j + 15) & 0x1f);
                if (j < 16)
                    t += f5(b2, c2, d2) + K2;
                else if (j < 32)
                    t += f4(b2, c2, d2) + K3;
                else if (j < 48)
                    t += f3(b2, c2, d2) + K4;
                else if (j < 64)
                    t += f2(b2, c2, d2) + K1;
                else
                    t += f1(b2, c2, d2) + K2;
                    
                a2 = e2;
                e2 = d2;
                d2 = ROTL(c2, 10);
                c2 = b2;
                b2 = t;
            }
            
            // 最终处理
            t = h1 + c1 + d2;
            h1 = h2 + d1 + e2;
            h2 = h3 + e1 + a2;
            h3 = h4 + a1 + b2;
            h4 = h0 + b1 + c2;
            h0 = t;
            
            // 写入输出 - 使用向量存储提高性能
            unsigned int* out_ptr = (unsigned int*)out;
            out_ptr[0] = __byte_perm(h0, 0, 0x0123);
            out_ptr[1] = __byte_perm(h1, 0, 0x0123);
            out_ptr[2] = __byte_perm(h2, 0, 0x0123);
            out_ptr[3] = __byte_perm(h3, 0, 0x0123);
            out_ptr[4] = __byte_perm(h4, 0, 0x0123);
        }
        '''
        
        # 编译核函数
        self.hash1_kernel = cp.RawKernel(hash1_kernel_code, 'hash1_kernel')
        self.hash2_kernel = cp.RawKernel(hash2_kernel_code, 'hash2_kernel')
        
        # 创建CUDA流
        self.streams = [cp.cuda.Stream() for _ in range(NUM_CUDA_STREAMS)]
        
        print("CUDA核函数编译完成") 

    def _generate_random_data(self, count: int) -> List[bytes]:
        """生成随机数据，使用NumPy加速生成过程"""
        try:
            # 使用NumPy生成随机数组，比循环生成更快
            random_bytes = np.random.randint(0, 256, size=(count, 32), dtype=np.uint8)
            
            # 转换为字节列表
            data_list = [bytes(row) for row in random_bytes]
            return data_list
        except Exception:
            # 如果NumPy方法失败，回退到传统方法
            data_list = []
            for _ in range(count):
                data = bytes(random.getrandbits(8) for _ in range(32))
                data_list.append(data)
            return data_list
    
    def _transform_key_to_identifier(self, private_key: bytes) -> str:
        """将密钥转换为标识符"""
        try:
            # 计算第一个哈希
            digest1 = hashlib.sha256(private_key).digest()
            
            # 计算第二个哈希
            h = hashlib.new('ripemd160')
            h.update(digest1)
            digest2 = h.digest()
            
            # 添加版本前缀 (0x00)
            versioned_hash = b'\x00' + digest2
            
            # 计算校验和
            checksum = hashlib.sha256(hashlib.sha256(versioned_hash).digest()).digest()[:4]
            
            # 组合得到最终的标识符字节
            binary_identifier = versioned_hash + checksum
            
            # Base58编码
            return base58.b58encode(binary_identifier).decode('utf-8')
        except Exception as e:
            # print(f"转换错误: {e}")
            return None
    
    def _update_performance_stats(self, start_time, end_time, num_processed, is_gpu=True):
        """更新性能统计信息"""
        elapsed = end_time - start_time
        if is_gpu:
            self.performance_stats['gpu_time'] += elapsed
            self.performance_stats['gpu_keys'] += num_processed
        else:
            self.performance_stats['cpu_time'] += elapsed
            self.performance_stats['cpu_keys'] += num_processed
            
        self.performance_stats['total_time'] = time.time() - self.start_time
        self.performance_stats['total_keys'] += num_processed
        
        # 检查是否需要报告性能
        current_time = time.time()
        if current_time - self.performance_stats['last_report_time'] >= self.performance_stats['report_interval']:
            self._report_performance()
            self.performance_stats['last_report_time'] = current_time
    
    def _report_performance(self):
        """报告性能统计信息"""
        stats = self.performance_stats
        total_time = stats['total_time']
        
        if total_time <= 0:
            return
            
        # 计算处理速率
        total_rate = stats['total_keys'] / total_time
        gpu_rate = stats['gpu_keys'] / stats['gpu_time'] if stats['gpu_time'] > 0 else 0
        cpu_rate = stats['cpu_keys'] / stats['cpu_time'] if stats['cpu_time'] > 0 else 0
        
        # 构建性能报告
        print("\n性能报告:")
        print(f"总处理数据量: {stats['total_keys']:,}")
        print(f"总运行时间: {total_time:.2f} 秒")
        print(f"平均处理速度: {total_rate:.2f} 项/秒")
        
        # 仅在使用GPU时输出GPU相关信息
        if self.use_gpu:
            print(f"GPU处理速度: {gpu_rate:.2f} 项/秒")
            if stats['cpu_time'] > 0:
                print(f"CPU处理速度: {cpu_rate:.2f} 项/秒")
                
            # 获取GPU内存使用情况
            try:
                free_mem, total_mem = cp.cuda.runtime.memGetInfo()
                used_mem = total_mem - free_mem
                used_percent = used_mem/total_mem*100
                print(f"GPU内存: {used_mem/1024**2:.1f}/{total_mem/1024**2:.1f} MB ({used_percent:.1f}%)")
                
                # 尝试获取GPU利用率
                self._try_report_gpu_utilization()
            except Exception:
                pass
        
        print("-" * 50)
    
    def _try_report_gpu_utilization(self):
        """尝试报告GPU利用率（如果pynvml可用）"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            print(f"GPU利用率: {util.gpu}%")
            print(f"GPU内存带宽利用率: {util.memory}%")
        except:
            # 如果不可用就静默失败
            pass

    def _batch_transform_gpu(self, data_list: List[bytes]) -> Dict[str, bytes]:
        """使用GPU批量处理数据"""
        if not self.use_gpu:
            return self._batch_transform_cpu(data_list)
            
        try:
            start_time = time.time()
            batch_size = len(data_list)
            result = {}
            
            # 计算每个流处理的数据大小，确保是32的倍数
            chunk_size = (batch_size // NUM_CUDA_STREAMS // 32) * 32
            if chunk_size == 0:
                chunk_size = 32
            
            # 计算网格和块大小
            block_size = min(CUDA_BLOCK_SIZE, chunk_size)
            grid_size = (chunk_size + block_size - 1) // block_size
            
            # 预分配所有GPU内存
            total_input_array = np.zeros((batch_size, 32), dtype=np.uint8)
            for i, data in enumerate(data_list):
                total_input_array[i] = np.frombuffer(data, dtype=np.uint8)
            
            device_total_in = cp.asarray(total_input_array)
            device_total_hash1_out = cp.zeros((batch_size, 32), dtype=cp.uint8)
            device_total_hash2_out = cp.zeros((batch_size, 20), dtype=cp.uint8)
            
            # 将批次分成多个子批次处理
            for start_idx in range(0, batch_size, chunk_size):
                end_idx = min(start_idx + chunk_size, batch_size)
                current_chunk_size = end_idx - start_idx
                
                # 确保当前块大小是32的倍数
                current_chunk_size = (current_chunk_size // 32) * 32
                if current_chunk_size == 0:
                    continue
                
                stream_idx = (start_idx // chunk_size) % NUM_CUDA_STREAMS
                current_grid_size = (current_chunk_size + block_size - 1) // block_size
                
                with self.streams[stream_idx]:
                    try:
                        # 获取当前批次的输入和输出视图
                        device_in = device_total_in[start_idx:start_idx + current_chunk_size]
                        device_hash1_out = device_total_hash1_out[start_idx:start_idx + current_chunk_size]
                        device_hash2_out = device_total_hash2_out[start_idx:start_idx + current_chunk_size]
                        
                        # 运行第一个哈希核函数
                        self.hash1_kernel(
                            grid=(current_grid_size,), 
                            block=(block_size,),
                            args=(device_in, device_hash1_out, current_chunk_size)
                        )
                        
                        # 运行第二个哈希核函数
                        self.hash2_kernel(
                            grid=(current_grid_size,),
                            block=(block_size,),
                            args=(device_hash1_out, device_hash2_out, current_chunk_size)
                        )
                        
                    except Exception as e:
                        print(f"处理GPU批次时出错: {e}")
                        continue
            
            # 同步所有流
            for stream in self.streams:
                stream.synchronize()
            
            # 一次性获取所有结果
            hash2_out = device_total_hash2_out.get()
            
            # 并行处理最终标识符生成
            for i in range(batch_size):
                try:
                    # 添加版本前缀 (0x00)
                    versioned_hash = b'\x00' + bytes(hash2_out[i])
                    
                    # 计算校验和
                    checksum = hashlib.sha256(hashlib.sha256(versioned_hash).digest()).digest()[:4]
                    
                    # 组合最终的标识符字节
                    binary_identifier = versioned_hash + checksum
                    
                    # Base58编码
                    identifier = base58.b58encode(binary_identifier).decode('utf-8')
                    result[identifier] = data_list[i]
                except Exception as e:
                    continue
            
            end_time = time.time()
            self._update_performance_stats(start_time, end_time, batch_size, is_gpu=True)
            return result
            
        except Exception as e:
            print(f"GPU处理失败: {e}")
            print("回退到CPU模式")
            return self._batch_transform_cpu(data_list)
    
    def _batch_transform_cpu(self, data_list: List[bytes]) -> Dict[str, bytes]:
        """使用CPU批量处理数据，优化版本"""
        start_time = time.time()
        result = {}
        
        # 预先创建需要用到的哈希对象
        h = hashlib.new('ripemd160')
        
        for data in data_list:
            try:
                # 计算SHA256哈希
                digest1 = hashlib.sha256(data).digest()
                
                # 执行RIPEMD-160哈希
                h.update(digest1)
                digest2 = h.digest()
                h.reset()  # 重置哈希对象以便重用
                
                # 添加版本前缀 (0x00)
                versioned_hash = b'\x00' + digest2
                
                # 计算校验和
                checksum = hashlib.sha256(hashlib.sha256(versioned_hash).digest()).digest()[:4]
                
                # 组合得到最终的标识符字节
                binary_identifier = versioned_hash + checksum
                
                # Base58编码
                identifier = base58.b58encode(binary_identifier).decode('utf-8')
                result[identifier] = data
            except Exception:
                # 静默处理错误，提高性能
                continue
        
        end_time = time.time()
        self._update_performance_stats(start_time, end_time, len(data_list), is_gpu=False)
        return result
    
    def _batch_check_identifiers(self, identifier_to_data: Dict[str, bytes]) -> List[Tuple[str, bytes]]:
        """检查一批标识符是否在目标集中"""
        found = []
        
        # 检查是否在目标标识符列表中
        if self.target_identifiers:
            for identifier, data in identifier_to_data.items():
                if identifier in self.target_identifiers:
                    found.append((identifier, data))
        
        # 检查是否在数据库中有对应值
        identifiers = list(identifier_to_data.keys())
        if identifiers:
            value_results = self.data_loader.check_values(identifiers)
            
            # 处理有值的标识符
            for identifier, (has_value, value) in value_results.items():
                if has_value and value > 0:
                    data = identifier_to_data[identifier]
                    # 如果已经在found列表中则跳过
                    if any(identifier == f[0] for f in found):
                        continue
                    found.append((identifier, data))
        
        return found
    
    def run_experiment(self) -> Dict[str, str]:
        """
        运行实验
        
        返回:
        字典 {标识符: 密钥(十六进制格式)}
        """
        start_time = time.time()
        processed_count = 0
        self.running = True
        
        print(f"开始科学计算实验...")
        print(f"GPU加速: {'启用' if self.use_gpu else '禁用'}")
        print(f"批处理大小: {self.batch_size}")
        if self.target_identifiers:
            print(f"目标标识符数量: {len(self.target_identifiers)}")
        
        # 检查数据是否已加载
        print(f"数据库状态: {'已加载' if self.data_loader.is_loaded else '未加载'}")
        print("实验将无限运行，直到发现有效数据或手动停止")
        
        # 记录开始信息
        self.logger.log_info(f"开始科学计算实验")
        self.logger.log_info(f"GPU加速: {'启用' if self.use_gpu else '禁用'}")
        self.logger.log_info(f"批处理大小: {self.batch_size}")
        if self.target_identifiers:
            self.logger.log_info(f"目标标识符数量: {len(self.target_identifiers)}")
        self.logger.log_info(f"数据库状态: {'已加载' if self.data_loader.is_loaded else '未加载'}")
        self.logger.log_info("实验将无限运行，直到发现有效数据或手动停止")
        
        # 进度统计变量
        last_milestone = 0
        
        try:
            while self.running:
                # 生成一批随机数据
                data_list = self._generate_random_data(self.batch_size)
                
                # 转换为标识符 (根据GPU是否可用选择不同的实现)
                if self.use_gpu:
                    identifier_to_data = self._batch_transform_gpu(data_list)
                else:
                    identifier_to_data = self._batch_transform_cpu(data_list)
                
                # 检查是否有匹配的标识符
                matches = self._batch_check_identifiers(identifier_to_data)
                
                # 处理所有匹配的标识符
                for identifier, data in matches:
                    hex_key = data.hex()
                    self.discovered_results[identifier] = hex_key
                    
                    # 获取关联值
                    value = 0
                    value_results = self.data_loader.check_values([identifier])
                    if identifier in value_results and value_results[identifier][0]:
                        value = value_results[identifier][1]
                    
                    # 记录和显示结果
                    public_data = hashlib.sha256(data).hexdigest()
                    print(f"\n[发现] 标识符: {identifier}")
                    if value > 0:
                        print(f"[发现] 关联值: {value}")
                    print(f"[发现] 密钥: {hex_key}")
                    
                    # 记录发现信息
                    self.logger.log_discovery(identifier, hex_key, public_data, value)
                    
                    # 发送通知
                    if self.notifier.enabled:
                        self.notifier.send_discovery_notification(identifier, hex_key, public_data, value)
                        print(f"已发送通知")
                
                # 更新统计信息
                processed_count += self.batch_size
                self.total_processed += self.batch_size
                
                # 每处理100万个数据输出一次进度信息
                current_milestone = processed_count // 1000000
                if current_milestone > last_milestone:
                    # 计算总体性能
                    total_elapsed = time.time() - start_time
                    total_rate = processed_count / total_elapsed if total_elapsed > 0 else 0
                    
                    # 简洁输出格式
                    if self.use_gpu:
                        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
                        used_mem = total_mem - free_mem
                        print(f"已处理 {current_milestone * 1000000:,} 个数据，速率: {total_rate:.2f} 项/秒")
                        print(f"GPU内存使用: {used_mem/1024**2:.2f} MB / {total_mem/1024**2:.2f} MB ({used_mem/total_mem*100:.1f}%)")
                    else:
                        print(f"已处理 {current_milestone * 1000000:,} 个数据，速率: {total_rate:.2f} 项/秒")
                    
                    # 记录到日志文件
                    self.logger.log_info(f"已处理 {current_milestone * 1000000:,} 个数据，速率: {total_rate:.2f} 项/秒")
                    
                    # 更新里程碑
                    last_milestone = current_milestone
                    
        except KeyboardInterrupt:
            print("\n检测到用户中断，正在停止...")
            self.logger.log_info("检测到用户中断，正在停止...")
            self.running = False
        except Exception as e:
            print(f"运行过程中出错: {e}")
            self.logger.log_info(f"运行过程中出错: {e}")
            self.running = False
        
        # 报告结果
        elapsed = time.time() - start_time
        rate = processed_count / elapsed if elapsed > 0 else 0
        print(f"\n完成实验")
        print(f"总处理数据量: {processed_count:,}")
        print(f"发现有效结果: {len(self.discovered_results)}")
        print(f"耗时: {elapsed:.2f} 秒")
        print(f"平均速率: {rate:.2f} 项/秒")
        
        # 记录结果
        self.logger.log_info(f"完成实验")
        self.logger.log_info(f"总处理数据量: {processed_count:,}")
        self.logger.log_info(f"发现有效结果: {len(self.discovered_results)}")
        self.logger.log_info(f"耗时: {elapsed:.2f} 秒")
        self.logger.log_info(f"平均速率: {rate:.2f} 项/秒")
        
        return self.discovered_results
    
    def set_batch_size(self, size: int):
        """设置批处理大小"""
        self.batch_size = size
        print(f"批处理大小设置为: {size}")
        self.logger.log_info(f"批处理大小设置为: {size}")
    
    def stop(self):
        """停止实验"""
        self.running = False
        print("正在停止实验...")
        self.logger.log_info("正在停止实验...")
        
    def __del__(self):
        """析构函数，确保资源被正确释放"""
        if hasattr(self, 'logger'):
            self.logger.close() 