# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os
import codecs
import sys
import time
import threading
import queue
import marshal
import gzip
from collections import deque
from functools import partial

from snownlp import normal
from snownlp import seg
from snownlp.classification.bayes import Bayes, AddOneProb

# 定义SimpleFreq类替代sentiment.freq
class SimpleFreq:
    def __init__(self):
        self.d = {}
        self.total = 0

# 自定义Bayes分类器，解决加载问题
class CustomBayes(Bayes):
    def __init__(self):
        super().__init__()
        # 确保d字典包含neg和pos
        self.d = {'neg': SimpleFreq(), 'pos': SimpleFreq()}
        self.total = {'neg': 0, 'pos': 0}
    
    def load(self, fname, iszip=True):
        """优化加载模型文件：支持增量训练"""
        try:
            # 检查文件是否存在
            if not os.path.exists(fname):
                # 检查是否有压缩版本
                if iszip:
                    gz_name = fname + '.gz'
                    if os.path.exists(gz_name):
                        fname = gz_name
                    else:
                        raise FileNotFoundError(f"模型文件不存在: {fname} 或 {gz_name}")
                else:
                    raise FileNotFoundError(f"模型文件不存在: {fname}")
                
            print(f"✅ 加载模型中: {fname}")
            if fname.endswith('.gz'):
                with gzip.open(fname, 'rb') as f:
                    data = marshal.load(f)
            else:
                with open(fname, 'rb') as f:
                    data = marshal.load(f)
            
            # 更新模型数据
            self.total = data.get('total', {'neg': 0, 'pos': 0})
            
            # 确保d字典包含neg和pos
            self.d = {'neg': SimpleFreq(), 'pos': SimpleFreq()}
            
            # 从加载的数据中恢复neg和pos
            if 'd' in data:
                for category in ['neg', 'pos']:
                    if category in data['d']:
                        self.d[category].d = data['d'][category].get('d', {})
                        self.d[category].total = data['d'][category].get('total', 0)
                        
            print(f"✅ 模型加载成功！")
            print(f"  负面训练数据: {self.total['neg']:,} 条")
            print(f"  正面训练数据: {self.total['pos']:,} 条")
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            # 创建新模型结构
            self.d = {'neg': SimpleFreq(), 'pos': SimpleFreq()}
            self.total = {'neg': 0, 'pos': 0}

class Sentiment(object):
    def __init__(self, load_path=None):
        self.classifier = CustomBayes()  # 使用自定义Bayes分类器
        if load_path:
            self.load(load_path)

    def save(self, fname, iszip=True):
        self.classifier.save(fname, iszip)

    def load(self, fname, iszip=True):
        self.classifier.load(fname, iszip)

    def handle(self, doc):
        words = seg.seg(doc)
        words = normal.filter_stop(words)
        return words

    def train(self, neg_docs, pos_docs, verbose=True, num_workers=4):
        """多线程训练实现 - 优化版"""
        if num_workers is None:
            try:
                num_workers = min(16, max(2, os.cpu_count() - 1))
            except:
                num_workers = 4
                
        data = []
        total = len(neg_docs) + len(pos_docs)
        
        if verbose:
            print("\n" + "="*60)
            print(f"🚀 开始训练，总共 {total:,} 条样本")
            print(f"  新增负面样本: {len(neg_docs):,} 条")
            print(f"  新增正面样本: {len(pos_docs):,} 条")
            if hasattr(self.classifier, 'total'):
                print(f"  历史负面词条: {self.classifier.total['neg']:,}")
                print(f"  历史正面词条: {self.classifier.total['pos']:,}")
            print(f"  使用 {num_workers} 个工作线程")
            print("="*60 + "\n")
        
        start_time = time.time()
        last_update_time = start_time
        last_processed = 0
        progress_lock = threading.Lock()
        
        # 使用线程安全的队列
        job_queue = queue.Queue()
        processed_count = 0
        processed_lock = threading.Lock()
        
        # 多线程处理函数
        def process_doc(doc, doc_type):
            words = self.handle(doc)
            label = 'neg' if doc_type == 'neg' else 'pos'
            return [words, label]
        
        # 添加所有任务到队列
        for doc in neg_docs:
            job_queue.put((doc, 'neg'))
        for doc in pos_docs:
            job_queue.put((doc, 'pos'))
        
        # 进度监控线程
        def progress_monitor():
            nonlocal last_update_time, last_processed, processed_count
            start = time.time()
            last_count = 0
            speed_history = deque(maxlen=10)
            
            if verbose:
                self._print_progress(0, "初始化...", 0, 0, 0)
            
            while processed_count < total:
                time.sleep(0.2)
                
                with processed_lock:
                    current_count = processed_count
                
                if current_count > last_count:
                    current_time = time.time()
                    items_processed = current_count - last_count
                    time_elapsed = current_time - last_update_time
                    current_speed = items_processed / time_elapsed if time_elapsed > 0 else 0
                    
                    speed_history.append(current_speed)
                    avg_speed = sum(speed_history) / len(speed_history) if speed_history else 0
                    
                    percent = (current_count / total) * 100
                    elapsed = current_time - start_time
                    remaining_items = total - current_count
                    estimated_remain = remaining_items / avg_speed if avg_speed > 0 else 0
                    status = f"处理中: {current_count:,}/{total:,}"
                    
                    if verbose:
                        with progress_lock:
                            self._print_progress(
                                percent, 
                                status, 
                                elapsed, 
                                estimated_remain,
                                avg_speed
                            )
                    
                    last_update_time = current_time
                    last_count = current_count
                    
            # 最终显示100%进度
            if verbose:
                with progress_lock:
                    elapsed = time.time() - start_time
                    self._print_progress(100.0, "处理完成", elapsed, 0, avg_speed)
                    sys.stdout.write("\n")
                    sys.stdout.flush()
        
        # 工作线程函数
        def worker():
            nonlocal data, processed_count
            batch_size = 50
            batch_data = []
            local_count = 0
            
            while True:
                try:
                    doc, doc_type = job_queue.get(timeout=0.1)
                    result = process_doc(doc, doc_type)
                    batch_data.append(result)
                    local_count += 1
                    
                    if local_count >= batch_size:
                        with processed_lock:
                            data.extend(batch_data)
                            processed_count += local_count
                        batch_data = []
                        local_count = 0
                    
                    job_queue.task_done()
                except queue.Empty:
                    if local_count > 0:
                        with processed_lock:
                            data.extend(batch_data)
                            processed_count += local_count
                    break
        
        # 创建进度监控线程
        monitor_thread = threading.Thread(target=progress_monitor)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # 创建线程池
        threads = []
        for _ in range(num_workers):
            t = threading.Thread(target=worker)
            t.daemon = True
            t.start()
            threads.append(t)
        
        # 等待所有工作线程完成
        for t in threads:
            t.join()
        
        if verbose:
            elapsed = time.time() - start_time
            print(f"\n✅ 样本处理完成! 耗时: {elapsed:.2f}秒")
            print(f"训练分类器中...")
        
        # 实际训练分类器
        train_start = time.time()
        self.classifier.train(data)
        
        # 更新统计
        if hasattr(self.classifier, 'total'):
            self.classifier.total['neg'] += len(neg_docs)
            self.classifier.total['pos'] += len(pos_docs)
        
        if verbose:
            elapsed_train = time.time() - train_start
            elapsed_total = time.time() - start_time
            
            print("\n" + "="*60)
            print(f"🎉 训练完成!".center(60))
            print("-"*60)
            print(f"⏱️ 总耗时: {elapsed_total:.2f}秒")
            print(f"  样本处理耗时: {elapsed:.2f}秒")
            print(f"  模型训练耗时: {elapsed_train:.2f}秒")
            print("-"*60)
            if hasattr(self.classifier, 'total'):
                print(f"📊 累计数据统计:")
                print(f"  负面训练数据: {self.classifier.total['neg']:,} 条")
                print(f"  正面训练数据: {self.classifier.total['pos']:,} 条")
            print("="*60)

    def _print_progress(self, percent, status, elapsed, remaining, speed):
        """显示进度条"""
        bar_len = 40
        filled_len = int(bar_len * percent / 100)
        bar = '█' * filled_len + '-' * (bar_len - filled_len)
        
        elapsed_str = self.format_time(elapsed)
        remain_str = self.format_time(remaining)
        speed_str = f"{speed:.1f}条/秒" if speed > 0 else ""
        
        progress_str = f'\r[{bar}] {percent:.1f}% - {status} - 耗时: {elapsed_str} - 剩余: {remain_str} - 速度: {speed_str}'
        
        max_length = 100
        if len(progress_str) < max_length:
            progress_str += ' ' * (max_length - len(progress_str))
        else:
            progress_str = progress_str[:max_length]
        
        sys.stdout.write(progress_str)
        sys.stdout.flush()
    
    def format_time(self, seconds):
        """时间格式化辅助函数"""
        if seconds < 60:
            return f"{seconds:.1f}秒"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}分钟"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}小时"
    
    def classify(self, sent):
        words = self.handle(sent)
        ret, prob = self.classifier.classify(words)
        if ret == 'pos':
            return prob
        return 1-prob


# 导出相关函数接口
def load_sentiment(fname, iszip=True):
    """加载情感分析模型"""
    global classifier
    classifier = Sentiment(load_path=fname)

def train_sentiment(neg_file, pos_file, model_file=None, verbose=True, num_workers=8):
    """训练情感分析模型，支持增量训练"""
    global classifier
    
    # 初始化模型实例
    if model_file:
        # 检查模型文件是否存在
        if os.path.exists(model_file):
            print(f"✅ 加载现有模型: {model_file}")
            classifier = Sentiment(load_path=model_file)
        else:
            print(f"⚠️ 模型文件不存在，将创建新模型: {model_file}")
            classifier = Sentiment()
    else:
        # 使用现有全局分类器
        if 'classifier' not in globals():
            classifier = Sentiment()
    
    # 读取数据
    try:
        print(f"\n📂 读取负面数据文件: {neg_file}")
        with codecs.open(neg_file, 'r', 'utf-8', errors='replace') as f:
            neg = [line.rstrip("\r\n") for line in f]
        
        print(f"📂 读取正面数据文件: {pos_file}")
        with codecs.open(pos_file, 'r', 'utf-8', errors='replace') as f:
            pos = [line.rstrip("\r\n") for line in f]
    except Exception as e:
        print(f"❌ 读取数据失败: {e}")
        return
    
    if verbose:
        print(f"\n📊 数据统计:")
        print(f"  新增负面样本: {len(neg):,}")
        print(f"  新增正面样本: {len(pos):,}")
        if hasattr(classifier.classifier, 'total'):
            print(f"  历史负面词条: {classifier.classifier.total['neg']:,}")
            print(f"  历史正面词条: {classifier.classifier.total['pos']:,}")
        print("🚀 开始训练...")
    
    # 训练模型
    classifier.train(neg, pos, verbose, num_workers)
    
    # 保存模型
    if model_file:
        try:
            print(f"\n💾 保存模型到: {model_file}")
            classifier.save(model_file)
            print(f"✅ 模型保存成功!")
        except Exception as e:
            print(f"❌ 保存模型失败: {e}")
    
    print("\n✅ 训练完成!")

def save_sentiment(fname, iszip=True):
    """保存情感分析模型"""
    global classifier
    if 'classifier' in globals():
        classifier.save(fname, iszip)
    else:
        print("⚠️ 没有可保存的模型")

def classify_sentiment(sent):
    """情感分类接口"""
    global classifier
    if 'classifier' in globals():
        return classifier.classify(sent)
    else:
        print("⚠️ 情感模型未加载")
        return 0.5

# 初始化全局模型
classifier = Sentiment()