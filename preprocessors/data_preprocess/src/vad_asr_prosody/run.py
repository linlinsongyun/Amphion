
import json
import os, sys
import logging
import time
import tqdm
from sil_detect import (
    vad_audio_multiprocess, 
    vad_audio_silero_multiprocess, 
    remove_vad,
    remove_vad_multiprocess
)
from asr  import asr_audio_batch, asr_audio_batch_single
import pprint

from utils import (
    get_wav_names,
    get_wav_names_multiprocess,
    split_interval_by_midpoints_dict,
    post_process_text_multiprocess,
    save_json,
    load_json, 
    setup_logger
)
from segment_audio import (
    segment_audio_v1_multiprocess,
    remove_segment,
    remove_segment_multiprocess
)

def display_time_log(time_logger, bool_do, logger):
    if not bool_do:
        return
    total = 0
    for key, value in time_logger.items():
        total += value
        
    logger.info('--' * 20)
    for key, value in time_logger.items():
        logger.info("timelog : {:10s} : {:.4f}%, {}".format(key, value / total * 100, value))
    logger.info('total : {}'.format(total))
    logger.info('--' * 20)


def dict_names(name_list:list) -> dict:
    d = {}
    for i in name_list:
        d[os.path.basename(i).replace('.wav', '')] = i
    return d
    
    

"""
检测到静音，插入#3, 遗留问题：根据静音长度插入不同的标记
给出音频路径描述文件， 获取音频路径列表。一般有两种格式
  * 带音素标注的格式：path|key|lang|text|phs
  * asr 标注格式：key|path|text
根据音频列表处理数据：
  * 1000个音频为一组，多进程作vad。
  * vad结束，获取vad信息，多进程切分音频。获取scp
  * 使用scp，单进程 batch 推理，获取asr文本
  * 多进程作文本后处理。写进结果dict。
  * 10000句存储一次结果文件。
"""
if __name__ == "__main__":
    exp_name = sys.argv[1] # "my_logger"       # 用于区分不同外部进程存储目录
    name_path = sys.argv[2]
    save_dir = sys.argv[3]
    
    logger_save_path = os.path.join(save_dir, "logs", "log_{}.log".format(exp_name))
    save_path = os.path.join(save_dir, os.path.basename(name_path).replace('.txt', "_vadasrpsd_{}.json".format(exp_name)))
    save_name_path = os.path.join(save_dir, os.path.basename(name_path).replace('.txt', "_dictname_{}.json".format(exp_name)))
    os.makedirs(os.path.dirname(logger_save_path), exist_ok=True)
    setup_logger(log_name=exp_name, save_path=logger_save_path, log_level='INFO')
    logger = logging.getLogger(exp_name)
    
    INFER_BATCH  = 1000
    SAVE_BATCH   = 3
    is_asr_whole = True
    do_denoise   = False
    use_silero   = False
    time_log     = True
    
    assert os.path.isfile(name_path), f"{name_path}"
    logger.info(f"Load wav_path list from {name_path}")
    
    # prepare namedict
    time_read_st = time.time()
    try:
        names_dict = load_json(save_name_path, exp_name)
        logger.info(f"Load {len(names_dict)} wavs (exist) from {save_name_path}, cost time {time.time() - time_read_st}")
    except:
        logger.warning(f"Load from {save_name_path} faild, trying from {name_path}")
        names_dict = get_wav_names_multiprocess(name_path, exp_name)
        save_json(save_name_path, names_dict, exp_name)
        logger.info(f"Load {len(names_dict)} wavs (exist) from {name_path}, cost time {time.time() - time_read_st}")
    
    # try to load processed data
    d = {}
    try:
        d = load_json(save_path, exp_name)
        logger.info(f"Load {len(d)} processed data from {save_path}")
    except:
        logger.warning(f"Can not load data from empty json : {save_path}")

    # processing
    vad_dict = {}
    all_len = len(names_dict)
    save_index = 0
    for idx, (key, name) in enumerate(names_dict.items()):
        # 已处理数据，直接忽略
        if key in d.keys() and d[key]['whole_text'] != "":
            continue
        time_logger = {
            "vad" : 0, "denoise" : 0, "midpoint" : 0, "segment" : 0, "asr" : 0,
            "post" : 0, "get_res" : 0, "delete" : 0
        }
        
        vad_dict[key] = name
        if len(vad_dict) < INFER_BATCH and idx < all_len-1:
            continue
        if not vad_dict:
            logger.info("There is no more wavs to processing, end!")
            sys.exit(0)
            
        save_index += 1
        time_start = time.time()
        
        # step1 : norm, denoise, vad
        logger.info(f"Step1 : VAD audios")
        if use_silero:
            d_vad_intervals = vad_audio_silero_multiprocess(vad_dict, do_denoise, exp_name)
        else:
            d_vad_intervals = vad_audio_multiprocess(vad_dict, do_denoise, exp_name)                         # {audio_key : {inters: [], min:0, max=len(audio), denoise_path : }, ...} 
        time_vad = time.time()
        time_logger['vad'] = time_vad - time_start
        
        # ste2 : get speech interval
        if use_silero:
            d_seginfo = {key : value['inters'] for key, value in d_vad_intervals.items()}
        else:
            d_seginfo = split_interval_by_midpoints_dict(d_vad_intervals)                          # {audio_key:inters, ...}
        time_mid = time.time()
        time_logger['midpoint'] = time_mid - time_vad
        
        # step3 : segment speech by interval
        logger.info(f"Step2 : Split intervals")
        l_scppaths = segment_audio_v1_multiprocess(d_seginfo, vad_dict, idx, is_asr_whole, d_vad_intervals, exp_name, INFER_BATCH)     # audio_key whole_audio_path \n audio_key_sub1 audio_subpath1 \n ...
        time_seg = time.time()
        time_logger['segment'] = time_seg - time_mid
        
        # step4 : asr batchd
        logger.info(f"Save scp file into {l_scppaths}")
        d_asr_text = asr_audio_batch(l_scppaths, exp_name)                                     # {audio_key : {whole : obj(text|timestamp), subs : list(obj(text|timestamp))}}
        time_asr = time.time()
        time_logger['asr'] = time_asr - time_seg
        
        # step5 : collect res
        d_res = post_process_text_multiprocess(d_asr_text, is_asr_whole)                       # {audio_key : {is_vad_correct:, prosody_text:, whole_text:}}
        time_post = time.time()
        time_logger['post'] = time_post - time_asr
        
        # step6 : 归档数据
        for key, value in d_res.items():
            d[key] = {
                "wav_name" : vad_dict[key],
                # "wav_dur" : d_vad_intervals[key]['max'],
                "vad" : str(d_vad_intervals[key]['inters']),
                "seg_info" : str(d_seginfo[key]),
                "whole_text" : value.get('whole', None), 
                "whole_raw_text" : value.get('whole_raw', None), 
                "prosody_text" : value['prosody_text'],
                "prosody_text_raw" : value['prosody_text_raw'],
                "is_vad_correct" : value['is_vad_correct'],
                "whole_timestamp" : value['whole_timestamp']
            }
        time_res = time.time()
        time_logger['get_res'] = time_res - time_post
        
        
        
        # step7 : 删除中间文件，及时清理空间
        logger.info("step7 : Removing tmp data")
        remove_vad_multiprocess(vad_dict, exp_name)
        remove_segment_multiprocess(vad_dict, exp_name)
        time_del = time.time()
        time_logger['delete'] = time_del - time_res
            
        display_time_log(time_logger, time_log, logger)
        vad_dict = {}
        
        # 存储当前批次数据
        save_json(save_path, d, exp_name)
        
        
        # if save_index % SAVE_BATCH == 0:
        #     save_json(save_path, d, exp_name)
            
    save_json(save_path, d, exp_name)
        
        
