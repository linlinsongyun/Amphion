import json
import os, glob, shutil
import tqdm
from pydub import AudioSegment
import soundfile
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import split_list
import pprint

tmp_asr_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_asr_audio_dir")
os.makedirs(tmp_asr_save_dir, exist_ok=True)


def segment_audio(wav_name, asr_segs):
    """
    按照给出的segments切分音频并存储
    :wav_name : 等待切分的音频路径
    :resturn : 切分之后的音频暂存目录
    """
    global tmp_asr_save_dir
    sr = 16000 # asr 默认使用16k采样率
    audio = AudioSegment.from_file(wav_name, format="wav", sr=sr)
    assert asr_segs[-1][-1] == len(audio)
    save_dir = os.path.join(tmp_asr_save_dir, os.path.basename(wav_name).replace('.wav', ''))
    os.makedirs(save_dir, exist_ok=True)
  
    res = []
    for idx, (start, end) in enumerate(asr_segs):
        segment = audio[start:end]
        save_path = os.path.join(save_dir, "{}.wav".format(idx))
        segment.export(save_path, format="wav")
        res.append(save_path)
    return res
    
    
def segment_audio_v1(wav_name, asr_segs):
    """
    按照给出的segments切分音频并存储
    :wav_name : 等待切分的音频路径
    :resturn : 切分之后的音频暂存目录
    """
    global tmp_asr_save_dir
    sr = 16000 # asr 默认使用16k采样率
    audio = AudioSegment.from_file(wav_name, format="wav", sr=sr)
    assert asr_segs[-1][-1] == len(audio)
    save_dir = os.path.join(tmp_asr_save_dir, os.path.basename(wav_name).replace('.wav', ''))
    os.makedirs(save_dir, exist_ok=True)
  
    res = []
    for idx, (start, end) in enumerate(asr_segs):
        segment = audio[start:end]
        save_path = os.path.join(save_dir, "{}_{}.wav".format(os.path.basename(wav_name).replace('.wav', ''), idx))
        segment.export(save_path, format="wav")
        res.append(save_path)
    return res


def _segment_audio_v1_multiprocess(item):
    global tmp_asr_save_dir
    key, asr_segs, logger_name, wav_name = item
    sr = 16000 # asr 默认使用16k采样率
    audio = AudioSegment.from_file(wav_name, format="wav", sr=sr)
    # assert asr_segs[-1][-1] == len(audio), f"{asr_segs}, | {len(audio)}"
    save_dir = os.path.join(tmp_asr_save_dir, logger_name, "{}.wav".format(key))    # os.path.basename(wav_name).replace('.wav', '')
    os.makedirs(save_dir, exist_ok=True)
  
    sub_names = []
    for idx, (start, end) in enumerate(asr_segs):
        segment = audio[start:end]
        save_path = os.path.join(save_dir, "{}_{}.wav".format(key, idx))    # os.path.basename(wav_name).replace('.wav', '')
        segment.export(save_path, format="wav")
        sub_names.append(save_path)
    return key, wav_name, sub_names


def segment_audio_v1_multiprocess(d_seginfo, d_names, idx, asr_whole, d_vad_intervals=None, logger_name="asr_denoised", scp_batch=1000):
    """
    按照给出的segments切分音频并存储
    :wav_name : 等待切分的音频路径
    :resturn : 切分之后的音频暂存目录
    """
    global tmp_asr_save_dir
    logger = logging.getLogger(logger_name)
    log_str = "raw" if not d_vad_intervals else "norm"
    logger.debug(f"Using {log_str} audio for asr")

    input_data = []
    for key, value in d_seginfo.items():
        # silence detection faild, skip
        if d_vad_intervals is not None and key not in  d_vad_intervals.keys():
            logger.warning(" {} lost in d_vad_intervals, skip.".format(key))
            continue
        # vad 如果做了 denoise, ASR 就使用降噪之后的音频。
        input_data.append([key, value, logger_name, d_names[key] if d_vad_intervals is None else d_vad_intervals[key]['denoise_path']])
        
    processed_results = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_segment_audio_v1_multiprocess, item) for item in input_data]
        for future in tqdm.tqdm(as_completed(futures), total=len(input_data)):
            processed_results.append(future.result())

    # 切分生成处理scp
    wav_paths = []
    for item in processed_results:
        key, whole_path, sub_paths = item
        if asr_whole:
            wav_paths.append([key, whole_path])
        for _idx, i in enumerate(sub_paths):
            wav_paths.append(["gdsub_" + key + "_" + str(_idx), i])
            
    l_wav_paths = split_list(wav_paths, max(1, len(wav_paths) // scp_batch))
    
    scp_paths = []
    for scp_idx, wav_list in enumerate(l_wav_paths):
        scp_save_path = os.path.join(tmp_asr_save_dir, logger_name, "{}_{}.scp".format(idx, scp_idx))
        os.makedirs(os.path.dirname(scp_save_path), exist_ok=True)
        with open(scp_save_path, 'w') as f:
            for item in wav_list:
                key, cur_path = item
                f.write("{}\t{}\n".format(key, cur_path))
        scp_paths.append(scp_save_path)
    return scp_paths



def segment_audio_v2(wav_name, asr_segs):
    """
    按照给出的segments切分音频并存储
    :wav_name : 等待切分的音频路径
    :resturn : 切分之后的音频数据
    """
    global tmp_asr_save_dir
    sr = 16000 # asr 默认使用16k采样率
    audio, sr = soundfile.read(wav_name)
    assert asr_segs[-1][-1] == int(len(audio) / sr * 1000), f"{asr_segs[-1]}, {len(audio)}, {len(audio) / sr}, {len(audio) / sr * 1000}"
  
    res = []
    for idx, (start, end) in enumerate(asr_segs):
        segment = audio[int(start/1000*sr):int(end/1000*sr)]
        res.append(segment)
    return res
    
    
def remove_segment(wav_path, logger_name):
    global tmp_asr_save_dir
    logger = logging.getLogger(logger_name)
    directory_path = os.path.join(tmp_asr_save_dir, logger_name, os.path.basename(wav_path).replace('.wav', ''))
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
        # logger.info(f"Remove {directory_path}")
    else:
        logger.error('[ERROR] : delete faild, not exist : {}'.format(directory_path))
        
def remove_segment_processor(item):
    global tmp_asr_save_dir
    wav_path, logger_name = item
    remove_segment(wav_path, logger_name)
           
    
def remove_segment_multiprocess(name_dict, logger_name):  
    # audio_items = list(name_dict.items())
    audio_items = []
    for key, value in name_dict.items():
        audio_items.append([value, logger_name])
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        # 使用 executor.map 处理键值对
        processed_results = executor.map(remove_segment_processor, audio_items)
