from pydub import AudioSegment
from pydub.silence import detect_silence
import librosa
import sys
import os
import logging
import tqdm, glob
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor, as_completed
import noisereduce as nr
import soundfile as sf
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps


SR=16000                        # 默认采样率16K Hz
TARGET_LOUND = -17              # audio_norm 调整到-17dB， 尽量避免清音识别成静音
MIN_SIL = 120                   # VAD 最短静音段，因为vad一般切的稍长，按照100ms标准，设置vad_sil_duration=120ms
SIL_THRESH = -40                # VAD 能量门限制
DENOISE_SCALE = 0.3             # 默认VAD降噪系数，默认不做。
ASR_DENOISE_SCALE = 0.85        # 默认ASR音频降噪系数，
meter = pyln.Meter(SR)
tmp_vad_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp_vad_audio_dir')
tmp_dns_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp_dns_audio_dir')
tmp_asr_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp_dnsasr_audio_dir')
os.makedirs(tmp_vad_dir, exist_ok=True)
os.makedirs(tmp_dns_dir, exist_ok=True)
os.makedirs(tmp_asr_dir, exist_ok=True)

# vad_silero_model = load_silero_vad()

# separator = Separator(model_file_dir='/mnt/nas1/hyc/UVR/models', output_dir=output_dir, output_single_stem="Instrumental")
# separator.load_model(model_filename='UVR-DeNoise.pth') 

def remove_vad(wav_path, log_name="vad"):
    global tmp_vad_dir, tmp_dns_dir
    logger = logging.getLogger(log_name)
    wav_name_vad = os.path.join(tmp_vad_dir, log_name, os.path.basename(wav_path))
    wav_name_dns = os.path.join(tmp_dns_dir, log_name, os.path.basename(wav_path))
    wav_name_asr = os.path.join(tmp_asr_dir, log_name, os.path.basename(wav_path))
    
    if os.path.isfile(wav_name_vad):
        os.remove(wav_name_vad)
    else:
        logger.error('[ERROR] : delete faild, not exist : {}'.format(wav_name_vad))
    if os.path.isfile(wav_name_dns):
        os.remove(wav_name_dns)
    # else:
    #     logger.error('[ERROR] : delete faild, not exist : {}'.format(wav_name_dns))
    if os.path.isfile(wav_name_asr):
        os.remove(wav_name_asr)
    # else:
    #     logger.error('[ERROR] : delete faild, not exist : {}'.format(wav_name_dns))

def remove_vad_processor(item):
    wav_path, log_name = item
    remove_vad(wav_path, log_name)
    
def remove_vad_multiprocess(name_dict, log_name="vad"):
    audio_items = []
    for key, value in name_dict.items():
        audio_items.append([value, log_name])
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        # 使用 executor.map 处理键值对
        processed_results = executor.map(remove_vad_processor, audio_items)


# min_silence_len:ms, silence_thresh:db, seek_step==hop_size: ms
def get_silence_intervals(audio_path, min_silence_len=50, silence_thresh=-40, seek_step=100, sr=24000):
    audio = AudioSegment.from_file(audio_path, format="wav", sr=sr)
    silences = detect_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    return silences, len(audio)


def norm_audio(key, wav_path, save_dir, sr=16000, target_loud=-23, meter=None, save_wav=False, log_name='vad'):
    wav, _ = librosa.core.load(wav_path, sr=sr)
    # norm loudness
    loudness = meter.integrated_loudness(wav)
    wav = pyln.normalize.loudness(wav, loudness, target_loud)
    # add start sil and end sil
    last_silence_pad = int(MIN_SIL/1000*SR)
    silence = np.zeros(last_silence_pad)
    wav = np.concatenate([silence, wav, silence], axis=0)
    # norm value
    if np.abs(wav).max() > 1:
        wav = wav / np.abs(wav).max()
    
    if save_wav:
        out_wav_tmp_path = os.path.join(save_dir, log_name, "{}.wav".format(key))    # os.path.basename(wav_path)
        os.makedirs(os.path.dirname(out_wav_tmp_path), exist_ok=True)
        sf.write(out_wav_tmp_path, wav, sr)
        return out_wav_tmp_path, None
    return wav, sr




def norm_audio_multiprocess(name_dict):
    global tmp_vad_dir, SR, TARGET_LOUND, meter
    name_dict_rev = {}
    input_data = []
    for key, value in name_dict.items():
        name_dict_rev[value] = key
        input_data.append([value, tmp_vad_dir, SR, TARGET_LOUND, meter, True])
    with ProcessPoolExecutor(max_workers=4, initializer=init_meter) as executor:
        # 使用 executor.map 处理键值对
        processed_results = executor.map(norm_audio, input_data)
        
    return name_dict_rev, [vad_path for vad_path, _ in processed_results]
 

def denoise_audio(norm_wav, sr, do_denoise, key, wav_path, log_name="vad"):
    global tmp_vad_dir, tmp_dns_dir, tmp_asr_dir, DENOISE_SCALE, ASR_DENOISE_SCALE
    # denoise for asr
    clean_audio2 = nr.reduce_noise(y=norm_wav, sr=sr, prop_decrease=ASR_DENOISE_SCALE)
    save_path2 = os.path.join(tmp_asr_dir, log_name, "{}.wav".format(key))    # os.path.basename(wav_path)
    os.makedirs(os.path.dirname(save_path2), exist_ok=True)
    sf.write(save_path2, clean_audio2, sr)
    
    if not do_denoise:
        out_wav_tmp_path = os.path.join(tmp_vad_dir, log_name, "{}.wav".format(key))    # os.path.basename(wav_path)
        os.makedirs(os.path.dirname(out_wav_tmp_path), exist_ok=True)
        sf.write(out_wav_tmp_path, norm_wav, sr)
        return out_wav_tmp_path, save_path2

    clean_audio1 = nr.reduce_noise(y=norm_wav, sr=sr, prop_decrease=DENOISE_SCALE)
    save_path1 = os.path.join(tmp_dns_dir, log_name, "{}.wav".format(key))    # os.path.basename(wav_path)
    os.makedirs(os.path.dirname(save_path1), exist_ok=True)
    sf.write(save_path1, clean_audio1, sr)
    
    return save_path1, save_path2

    
    
def _vad_audio(key, audio_path, do_denoise=False, log_name="vad"):
    global tmp_vad_dir, SR, TARGET_LOUND, MIN_SIL, SIL_THRESH, meter, meter_multi
    
    # norm audio   
    norm_wav, sr = norm_audio(key, audio_path, tmp_vad_dir, SR, TARGET_LOUND, meter, False, log_name)
    # denoise audio
    denoise_vad_audio_path, denoise_asr_audio_path = denoise_audio(norm_wav, sr, do_denoise, key, audio_path, log_name)
    # vad_audio
    silence_intervals, audio_len = get_silence_intervals(denoise_vad_audio_path, min_silence_len=MIN_SIL, silence_thresh=SIL_THRESH, sr=SR)
    
    return silence_intervals, audio_len, denoise_asr_audio_path


def vad_audio(audio_path):
    silence_intervals, audio_len, _ =  _vad_audio(audio_path)
    return silence_intervals, audio_len


def vad_audio_processor(item):
    key, audio_path, do_denoise, log_name = item
    try:
        silence_intervals, audio_len, denoise_audio_path = _vad_audio(key, audio_path, do_denoise, log_name)
    except Exception as e:
        silence_intervals, audio_len, denoise_audio_path = [], 0, ""
    
    return key, {"inters" : silence_intervals, "min" : 0, "max" : audio_len, 'denoise_path' : denoise_audio_path}


def vad_audio_silero_processor(item):
    global vad_silero_model
    # _vad_silero_model = load_silero_vad()
    key, wav_path = item

    wav = read_audio(wav_path)
    speech_timestamps = get_speech_timestamps(
      wav,
      vad_silero_model,
      return_seconds=True,  # Return speech timestamps in seconds (default is samples)
    )
    tmp = []
    for i in speech_timestamps:
        tmp.append([i['start']*1000, i['end']*1000])
    return key, tmp


def init_meter():
    global meter_multi
    meter_multi = pyln.Meter(SR)  # 每个进程只创建一次 meter
    
    
def init_silero_model():
    global vad_silero_model
    vad_silero_model = load_silero_vad()  # 每个进程只创建一次 meter
    

def vad_audio_multiprocess(name_dict:dict, do_denoise:bool=False, log_name:str="vad") -> dict:
    """
    对于给出的音频列表，多线程处理vad。
    :name_dict : {audio_key : audio_path, ...}
    :return : {audio_key : {inter:, min=0, max=len_audio}, audio_key : ..., ...}
    """
    global tmp_vad_dir, SR, TARGET_LOUND, MIN_SIL, SIL_THRESH, meter
    logger = logging.getLogger(log_name)
    
    # audio_items = list(name_dict.items())
    audio_items = []
    processed_results = []
    for key, value in name_dict.items():
        audio_items.append([key, value, do_denoise, log_name])
    
    with ProcessPoolExecutor(max_workers=4, initializer=init_meter) as executor:
        futures = [executor.submit(vad_audio_processor, item) for item in audio_items]
        for future in tqdm.tqdm(as_completed(futures), total=len(audio_items)):
            processed_results.append(future.result())
        # processed_results = executor.map(vad_audio_processor, audio_items)
        
    results = {key: result for key, result in processed_results if result['denoise_path'] != ""}
    for key, value in results.items():
        logger.info(f"Save asr audio into {value['denoise_path']} (un-splited)")
        break
    
    return results

# def vad_audio_uvr5_multiprocess(name_dict:dict, do_denoise:bool=False) -> dict:
#     """
#     对于给出的音频列表，多线程处理vad。
#     :name_dict : {audio_key : audio_path, ...}
#     :return : {audio_key : {inter:, min=0, max=len_audio}, audio_key : ..., ...}
#     """
#     global tmp_vad_dir, SR, TARGET_LOUND, MIN_SIL, SIL_THRESH, meter
    
#     name_dict_rev, vad_paths = norm_audio_multiprocess(name_dict)
#     uvr5_denoise()
#     vad_audio_uvr5()
#     return results



def vad_audio_silero_multiprocess(name_dict:dict, do_denoise:bool=False, log_name:str="vad") -> dict:
    """
    对于给出的音频列表，多线程处理vad。
    :name_dict : {audio_key : audio_path, ...}
    :return : {audio_key : {inter:, min=0, max=len_audio}, audio_key : ..., ...}
    """
    global tmp_vad_dir, tmp_asr_dir, SR, TARGET_LOUND, MIN_SIL, SIL_THRESH, meter, ASR_DENOISE_SCALE
    
    #     audio_items = list(name_dict.items())

    #     # with ProcessPoolExecutor(max_workers=4, initializer=init_silero_model) as executor:
    #     with ProcessPoolExecutor(max_workers=4) as executor:
    #         # 使用 executor.map 处理键值对
    #         processed_results = executor.map(vad_audio_silero_processor, audio_items)
    #     results = {key: result for key, result in processed_results}
    logger = logging.getLogger(log_name)
    results = {}
    for key, wav_path in name_dict.items():      
        norm_wav, _ = librosa.load(wav_path, sr=SR)
        clean_audio2 = nr.reduce_noise(y=norm_wav, sr=SR, prop_decrease=ASR_DENOISE_SCALE)
        save_path2 = os.path.join(tmp_asr_dir, log_name, os.path.basename(wav_path))
        sf.write(save_path2, clean_audio2, SR)
        wav = read_audio(wav_path)
        speech_timestamps = get_speech_timestamps(
          wav,
          vad_silero_model,
          return_seconds=True,  # Return speech timestamps in seconds (default is samples)
        )
        tmp = []
        for i in speech_timestamps:
            tmp.append([i['start']*1000, i['end']*1000])
        results[key] = {"inters" : tmp, "min" : 0, "max" : len(norm_wav), 'denoise_path' : save_path2}
        
    for key, value in results.items():
        logger.info(f"Save asr audio into {value['denoise_path']} (un-splited)")
        break
        
    return results



"""
if __name__=="__main__":
    SR = 16000
    TARGET_LOUND = -23
    MIN_SILS = [200, 250, 300, 350] # , 300, 350
    SIL_THRESH = [-45, -50, -40] # , -35, -30, -25
    
    meter = pyln.Meter(SR)
    
    wav_dir = "/mnt/nas1/guoshaotong.gst/data/corpus-pipeline/compare_MFA_netalign/podcast/wavs_16k/" # sys.argv[1] #/mnt/nas1/kaixuan/runs/demo/f23_chat_wrong_breaks
    save_dir = "/mnt/nas1/guoshaotong.gst/data/corpus-pipeline/compare_MFA_netalign/podcast/sil_detection_res_cmp/"  #sys.argv[2]
    tmp_dir = os.path.join(wav_dir, "../tmp_loud_wavs")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    wavlists = glob.glob(os.path.join(wav_dir, "*wav"))
    
    tmp_filenames = glob.glob(os.path.join(tmp_dir, "*wav"))
    for i in tmp_filenames:
        os.remove(i)
    
    # 修改音频响度
    norm_wav_list = []
    for fi in tqdm.tqdm(wavlists):
        wav_path= os.path.join(wav_dir, fi)
        wav, _ = librosa.core.load(wav_path, sr=SR)
        loudness = meter.integrated_loudness(wav)
        wav = pyln.normalize.loudness(wav, loudness, TARGET_LOUND)
        if np.abs(wav).max() > 1:
            wav = wav / np.abs(wav).max()
        out_wav_tmp_path = os.path.join(tmp_dir, os.path.basename(wav_path))
        sf.write(out_wav_tmp_path, wav, SR)
        norm_wav_list.append(out_wav_tmp_path)
    
    for min_sil in MIN_SILS:
        for sil_thresh in SIL_THRESH:
            print("[INFO] : min_sil = {}, sil_thresh = {}".format(min_sil, sil_thresh))
            filename = os.path.join(save_dir, "sil_dectection_res_minsil-{}_sil_thresh-neg{}.txt".format(min_sil, -1*sil_thresh))
            with open(filename, 'w') as f:
                for fi in tqdm.tqdm(norm_wav_list):
                    wav_path = fi
                    # 计算音频中的静音区间
                    silence_intervals = get_silence_intervals(wav_path, min_silence_len=min_sil, silence_thresh=sil_thresh, sr=SR)
                    # 打印静音区间
                    res = "{}|".format(os.path.basename(fi))
                    for interval in silence_intervals:
                        res += "[{:.2f}, {:.2f}]".format(interval[0], interval[1]) + ", "
                    f.write("{}\n".format(res.strip()))
            print("[INFO] : save result into : {}".format(filename))

"""
