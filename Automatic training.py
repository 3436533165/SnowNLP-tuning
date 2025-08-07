import re
import pandas as pd
from snownlp import SnowNLP
from tqdm import tqdm

def analyze_novel_sentiment(input_file, output_file):
    # 1. 文件读取与验证
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        if not content:
            print(f"❌ 错误：文件 '{input_file}' 为空！")
            return
    except Exception as e:
        print(f"❌ 文件读取失败：{str(e)}")
        return

    # 2. 强化分句逻辑
    sentences = []
    current = ""
    splitters = {'。', '！', '？', '，', '；', '…', '~', '\n', '”', '」', '》', '！？', '？！'}
    quote_starts = {'“', '《', '「'}
    
    for char in content:
        if char in splitters and current.strip():
            current += char
            if any(q in current for q in quote_starts) and not current.count('”') % 2 == 0:
                continue
            if re.search(r'[道喊问说]：$', current[-3:]):
                continue
            sentences.append(current)
            current = ""
        else:
            current += char
    
    if current.strip():
        sentences.append(current)

    # 3. 过滤无效句子
    valid_sentences = []
    for sent in sentences:
        clean_sent = sent.strip()
        if len(clean_sent) > 2 and not all(char in splitters for char in clean_sent):
            valid_sentences.append(clean_sent)
    
    if not valid_sentences:
        print("❌ 未找到有效句子！请检查文件内容")
        return
    print(f"✅ 有效句子数：{len(valid_sentences)}")

    # 4. 情感分析处理
    results = []
    print("⏳ 情感分析中...")
    for sent in tqdm(valid_sentences, desc="进度"):
        try:
            if sent.startswith(('”', '」')) and results:
                prev_sent = results[-1].split(",", 1)[1]
                if prev_sent.endswith(('：“', '：「')) or re.search(r'[道喊问说]：$', prev_sent[-3:]):
                    sent = prev_sent + sent
                    results.pop()
            
            s = SnowNLP(sent)
            label = 1 if s.sentiments >= 0.5 else 0
            results.append(f"{label},{sent}")
        except:
            results.append(f"0,{sent}")

    # 5. 结果后处理（正则替换）
    content_str = "\n".join(results)
    
    # 规则1: 合并连续换行符
    content_str = re.sub(r'\n\n+', '\n', content_str)
    
    # 规则2: 行首无标签行添加默认标签
    content_str = re.sub(r'(^|\n)(?!0,|1,)', r'\g<1>1,', content_str)
    
    # 规则3: 删除中文引号
    content_str = re.sub(r'[“”]', '', content_str)

    # 6. 写入最终结果
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content_str)
        print(f"✅ 分析完成！结果已保存至：{output_file}")
        
        # 结果预览
        print("\n🔍 结果样例：")
        for res in content_str.split('\n')[:3]:
            print(res)
            
    except Exception as e:
        print(f"❌ 结果写入失败：{str(e)}")

if __name__ == "__main__":
    analyze_novel_sentiment("data.txt", "res.txt")