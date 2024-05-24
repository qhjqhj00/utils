import pandas as pd
import json
from nltk.tokenize import sent_tokenize
from sacremoses import MosesTokenizer, MosesDetokenizer
from typing import List, Dict
from tqdm import tqdm
from dataclasses import dataclass
import re
from multiprocessing import Pool
import nltk
from statistics import quantiles
from math import ceil
import tiktoken
from collections import defaultdict

mt, md = MosesTokenizer(lang='en'), MosesDetokenizer(lang='en')
encoder = tiktoken.get_encoding("cl100k_base")

def csv2json(file_path, sep=','):
    df = pd.read_csv(file_path, sep=sep, engine='python')
    json_data = df.to_dict(orient='records')
    return json_data

def json2csv(json_list, out_path):
    df = pd.DataFrame(json_list)
    df.to_csv(out_path, index=False)

def tok(text):
    return mt.tokenize(text)

def detok(text):
    return md.detokenize(text)

def load_jsonl(path: str) -> List:
    rtn = []
    print(f"Begin to load {path}")
    for line in tqdm(open(path)):
        line = json.loads(line)
        rtn.append(line)
    return rtn

def save_jsonl(data: list, path: str) -> None:
    with open(path, "w") as f:
        for line in data:
            f.write(
                json.dumps(
                    line, ensure_ascii=False)+"\n")

def load_json(path: str):
    with open(path, "r") as f:
        rtn = json.load(f)
    return rtn

def save_json(data, path: str):
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def mp_utils(data: list, func, n_worker:int=8, chunksize:int=1):
    succeed = []
    failed = []
    with Pool(processes=n_worker) as p:
        for line in tqdm(p.imap(func, data, chunksize=chunksize)):
            if line:
                succeed.append(line)
    print(f"obtain {len(succeed)} results out of {len(data)} samples...")
    return succeed

def narrow(text, max_length, status=False):
    text_ids = encoder.encode(text)
    if len(text_ids) > max_length:
        half = int(max_length / 2)
        new_text_ids = text_ids[:half] + text_ids[-half:]
        return encoder.decode(new_text_ids)
        if status:
            print("sequence length: ", len(text_ids))
            print("new length: ", len(new_text_ids))
    else:
        if status:
            print("sequence length: ", len(text_ids))
        return text

def txt(path):
    return open(path).read()

def balance_chunk(doc, max_length, _id):

    sentences = nltk.tokenize.sent_tokenize(doc)
    sentences = [sent.lstrip().strip() for sent in sentences]
    sentences = [sent for sent in sentences if sent]

    # 计算整个文档的编码长度
    encoded_doc = encoder.encode(doc)
    total_length = len(encoded_doc)
    num_parts = ceil(total_length / max_length)  # 分割目标数量
    target_length = ceil(total_length / num_parts)+100 # 目标平均长度

    parts = []
    current_part = []
    current_length = 0

    for sentence in sentences:
        encoded_sentence = encoder.encode(sentence)
        sentence_length = len(encoded_sentence)
        if current_length + sentence_length > target_length:

            
            parts.append(" ".join(current_part))
            current_part = [sentence]
            current_length = sentence_length
        else:
            current_part.append(sentence)
            current_length += sentence_length

    if current_part:
        parts.append(" ".join(current_part))

    # 生成输出格式
    output = [{"content": part, "length": len(encoder.encode(part)), "_id": f"{_id}<split>{idx}"} for idx, part in enumerate(parts)]
    return output
    
def passage_chunk(text, by_passage=False, max_len=512):
    if re.findall(r'(Passage \d+:)', text) and by_passage:
        split_text = re.sub(r'(Passage \d+:)', r'sss@sss', text)
        passages = split_text.split("sss@sss")
        passages = [p for p in passages if p]
    else:
        sents = nltk.sent_tokenize(text)
        passages = []
        tmp_sent = []
        tmp_length = 0
        for sent in sents:
            sent_length = len(tok(sent))
            if tmp_length + sent_length > max_len:
                passages.append(" ".join(tmp_sent))
                tmp_sent = tmp_sent[-1:]
                if tmp_sent:
                    tmp_length = len(tok(tmp_sent[0]))
                else:
                    tmp_length = 0
            
            tmp_sent.append(sent)
            tmp_length += sent_length
        if tmp_sent:
            passages.append((" ".join(tmp_sent)))
    return passages

def insert_idx(text):
    sents = nltk.sent_tokenize(text)
    sents = [f"[s{i}] {sent}" for i,sent in enumerate(sents)]
    return  " ".join(sents)

def extract_idx(text, sentence_indices):
    # Dictionary to store the sentences with their corresponding index
    sentences_dict = {}
    
    # Regular expression to find sentences starting with [sX]
    pattern = r'(\[s\d+\])\s*(.*?)(?=\[\s*s\d+\]|\Z)'
    for match in re.finditer(pattern, text, re.DOTALL):
        index = match.group(1)
        sentence = match.group(2).strip()
        sentences_dict[index] = sentence
    # Retrieve the requested sentences based on the input list of indices
    extracted_sentences = [sentences_dict.get(index) for index in sentence_indices if index in sentences_dict]
    
    return extracted_sentences

def process_reader_res(res, chunk):
    sents = nltk.sent_tokenize(chunk)
    indices = re.findall(r'\d+', res)
    indices = [int(i) for i in indices]
    rtn = []
    for i,sent in enumerate(sents):
        if i in indices:
            rtn.append(sent)
    return rtn

def get_top_k_passages(query, context, reranker, k):
    # 使用chunk函数将context分为passages
    passages = passage_chunk(context, by_passage=True, max_len=512)
    
    # 使用reranker对每个passage进行评分
    scores = reranker.compute_score([[query, passage]for passage in passages]) 
    
    # 将passages和scores打包在一起并按分数排序
    scored_passages = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
    
    # 返回评分最高的k个passages
    return [passage for passage, score in scored_passages[:k]]

def score2latex(file_dir, out_file):
    files = [f for f in os.listdir(path) if f.endswith('.json')]
    # 读取json文件并存储在dataframe中
    data = defaultdict(dict)
    for file in files:
        with open(os.path.join(path, file)) as f:
            method = file[:-5]
            method = method.replace("_", " ")
            tmp = json.load(f)  # 去掉.json后缀
            for t in tmp:
                data[t][method] = tmp[t]
    df = pd.DataFrame(data)
    # 找到每一列的最大值并标记为粗体
    for col in df.columns:
        second_max = df[col].nlargest(2).values[1]
        print()
        df[col] = df[col].apply(lambda x: f'\\textbf{{{round(x,1)}}}' if x == df[col].max() else f"{round(x,1)}")
        df[col] = df[col].apply(lambda x: f'\\underline{{{x}}}' if x == f"{round(float(second_max),1)}" else x)
        

    # 将dataframe转换为latex表格并保存到txt文件中
    with open(out_file, 'w') as f:
        f.write(df.to_latex())

@dataclass
class Percentiles:
    minimum: float
    maximum: float
    mean: float
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float

    def to_json(self) -> Dict[str, float]:
        return dict(vars(self))

    @classmethod
    def from_list(cls, values: List[float]):
        count = len(values)
        if count == 0:
            minimum = maximum = mean = p50 = p75 = p90 = p95 = p99 = 0.0
        elif count == 1:
            minimum = maximum = mean = p50 = p75 = p90 = p95 = p99 = values[0]
        else:
            mean = sum(values) / count
            minimum, maximum = float(min(values)), float(max(values))
            quants = quantiles(values, n=100, method="inclusive")
            p50 = quants[49]
            p75 = quants[74]
            p90 = quants[89]
            p95 = quants[94]
            p99 = quants[98]

        return Percentiles(
            minimum=minimum,
            maximum=maximum,
            mean=mean,
            p50=p50,
            p75=p75,
            p90=p90,
            p95=p95,
            p99=p99,
        )
