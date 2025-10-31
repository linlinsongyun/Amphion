import json
import os, glob, shutil
import tqdm
from pydub import AudioSegment
import soundfile
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

def load_json(json_path, log_name="load_json"):
    logger = logging.getLogger(log_name)
    logger.info(f"Loading json data from {json_path}")
    with open(json_path, 'r') as f:
        return json.load(f)
    

def save_json(json_path, data, log_name="save_json"):
    logger = logging.getLogger(log_name)
    logger.info(f"Saving {len(data)} data into json {json_path}")
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# 配置日志记录器
def setup_logger(log_name="my_logger", save_path="./debug.log", log_level='DEUBG'):
    log_levels = {'DEUBG' : logging.DEBUG, "INFO" : logging.INFO, "ERROR" : logging.ERROR}
    # 创建日志记录器
    logger = logging.getLogger(log_name)
    logger.setLevel(log_levels[log_level])  # 设置日志级别

    # 创建控制台处理器
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(log_levels[log_level])

    # 创建文件处理器
    file_handler = logging.FileHandler(save_path, mode="a")
    file_handler.setLevel(log_levels[log_level])

    # 设置日志格式
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 将处理器添加到日志记录器
    # logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    


def split_list(lst, n):
    if n == 1:
        return [lst]
    # 计算每份的基本大小和多余的元素数量
    base_size, remainder = divmod(len(lst), n)
    
    # 初始化结果列表
    result = []
    start = 0
    
    for i in range(n):
        # 计算当前子列表的大小
        size = base_size + (1 if i < remainder else 0)
        # 切片获取子列表
        result.append(lst[start:start + size])
        # 更新起始位置
        start += size
    
    return result


def split_interval_by_midpoints(list1, min_value, max_value):
    # 初始化结果列表
    result = []

    # 提取所有区间的中点
    midpoints = [(interval[0] + interval[1]) / 2 for interval in list1]

    # 添加起始边界
    start = min_value

    # 遍历所有中点
    for midpoint in midpoints:
        # 如果中点在当前范围内，则切分 [start, midpoint]
        if start < midpoint < max_value:
            result.append([start, midpoint])
            start = midpoint

    # 添加结束边界
    if start < max_value:
        result.append([start, max_value])

    return result

def split_interval_by_midscale(list1, min_value, max_value):
    try:
        min_value = list1[0][1]
        max_value = list1[-1][0]

        result = []

        # 提取所有区间的中点
        midpoints = [(interval[0] + interval[1]) / 2 for interval in list1]
        # 添加起始边界
        start = min_value

        # 遍历所有中点
        for midpoint in midpoints:
            # 如果中点在当前范围内，则切分 [start, midpoint]
            if start < midpoint < max_value:
                result.append([start, midpoint])
                start = midpoint
        # 添加结束边界
        if start < max_value:
            result.append([start, max_value])
        
        results = []
        for i, j in result:
            results.append([min(j-45, i+45), max(i+45, j-45)])

        if len(results) > 0:
            results[-1][-1] = result[-1][-1]
            results[0][0] = result[0][0]

        return results
    except:
        return []


def split_interval_by_midpoints_dict(d_vad_intervals):
    """
    根据vad结果，切分原始音频，并存储
    """
    res = {}
    for key, value in d_vad_intervals.items():
        # res[key] = split_interval_by_midpoints(value['inters'], value['min'], value['max'])
        res[key] = split_interval_by_midscale(value['inters'], value['min'], value['max'])
    return res

def convert_timestamp_to_intervals(intervals:list) -> list:
    min_value = 0
    result = []
    
    # 检查第一个区间
    if intervals[0][0] != 0:
        result.append([0, intervals[0][0]])
    
    # 遍历区间，找出不连续的部分
    for i in range(1, len(intervals)):
        prev_end = intervals[i-1][1]
        current_start = intervals[i][0]
        
        if current_start > prev_end:
            result.append([prev_end, current_start])
    
    return result


def get_names(base_dir, suffix=None):
    return glob.glob(os.path.join(base_dir, "*{}".format(suffix)))


def post_process_text(asr_text_whole, asr_text_seg):
    whole_text = asr_text_whole[0]['text']
    
    asr_texts = ""
    asr_text_psd = ""
    for i in asr_text_seg:
        if not i:
            continue
        if i[0]['text'].strip() == "":
            continue
        asr_texts += i[0]['text']
        asr_text_psd += i[0]['text'] + ' #3 '
        
    whole_text = whole_text.replace(' ', '').strip()
    asr_texts = asr_texts.replace(' ', '').strip()
    
    asr_text_psd = asr_text_psd.strip().rstrip('#3').strip()

    return len(whole_text) == len(asr_texts), asr_text_psd, whole_text, asr_texts


def post_process_text_multiprocess(d_asr_text, asr_whole):
    """
    
    """
    d_res = {}
    for key, value in d_asr_text.items():
        if key not in d_res.keys():
            d_res[key] = {}
            
        asr_text_psd = ""
        asr_text_psd_raw = ""
        for sub_key, sub_value in value.items():
            if sub_key == "whole":
                d_res[key][sub_key] = sub_value['text'].replace(' ', '').strip()
                d_res[key]['whole_raw'] = sub_value['text'].strip()
                # d_res[key][sub_key] = sub_value['text'].strip()
                d_res[key]['whole_timestamp'] = str(sub_value['timestamp'])
                # d_res[key]['asr_vad'] = convert_timestamp_to_intervals(sub_value['timestamp'])
                continue
            is_empty_text = False
            if sub_value['text'].strip() == "":
                #continue
                is_empty_text = True
                pass
            asr_text_psd += " <None_asr_text> " if is_empty_text else sub_value['text'].replace(' ', '').strip() + " #3 "
            asr_text_psd_raw += " <None_asr_text> " if is_empty_text else sub_value['text'].strip() + " #3 "  
            
        if 'whole' not in d_res[key].keys():
            d_res[key]['whole'] = ""
        d_res[key]['prosody_text']   = asr_text_psd.strip().rstrip('#3').strip()
        d_res[key]['prosody_text_raw']   = asr_text_psd_raw.strip().rstrip('#3').strip()
        d_res[key]['is_vad_correct'] = None if not asr_whole else len(d_res[key]['whole']) == len(d_res[key]['prosody_text'].replace('#3', '').replace(' ', ''))

    return d_res

def get_wav_names(txt_path:str) -> dict:
    """
    带音素标注的格式：path|key|lang|text|phs
    asr 标注格式：key|path|text
    """
    
    tagd = False
    index = None
    
    res = {}
    with open(txt_path, 'r') as f:
        for line in tqdm.tqdm(f.readlines()):
            if not tagd:
                if line.strip().split('|')[0].endswith('.wav'):
                    key_index = 1
                    path_index = 0
                else:
                    key_index = 0
                    path_index = 1
                tagd = True
            if os.path.isfile(line.strip().split('|')[path_index]):
                res[line.strip().split('|')[key_index]] = line.strip().split('|')[path_index]
    return res
     

def get_wav_names_processor(item):
    line, key_index, path_index = item
    if os.path.isfile(line.strip().split('|')[path_index]):
        return line.strip().split('|')[key_index], line.strip().split('|')[path_index]
    return None, None

    
def get_wav_names_multiprocess(txt_path:str, log_name="load_json") -> dict:
    """
    带音素标注的格式：path|key|lang|text|phs
    asr 标注格式：key|path|text
    """
    logger = logging.getLogger(log_name)
    tagd = False
    index = None
    key_index = None
    path_index = None
    
    
    res = {}
    lines = []
    processed_results = []
    with open(txt_path, 'r') as f:
        for line in tqdm.tqdm(f.readlines()):
            if len(line.strip().split('|')) < 2:
                logger.warning("skip : [{}]".format(line.strip()))
                continue
            if not tagd:
                if line.strip().split('|')[0].endswith('.wav'):
                    logger.info("key_index = 1, path_index = 0")
                    key_index = 1
                    path_index = 0
                else:
                    logger.info("key_index = 0, path_index = 1")
                    key_index = 0
                    path_index = 1
                tagd = True
            lines.append([line.strip(), key_index, path_index])
            
    logger.info("Load {} data".format(len(lines)))
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(get_wav_names_processor, item) for item in lines]
        for future in tqdm.tqdm(as_completed(futures), total=len(lines)):
            processed_results.append(future.result())
        # processed_results = executor.map(get_wav_names_processor, lines)

    for key, wav_path in processed_results:
        if not key:
            continue
        res[key] = wav_path
    logger.info("Load exist {} wav, drop {}".format(len(res), len(lines) - len(res)))
    return res
