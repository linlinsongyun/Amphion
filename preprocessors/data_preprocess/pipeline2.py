import os
import io
import scipy.io.wavfile
import argparse
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_nonsilent
import numpy as np
import datetime
import librosa
#import torch
import time
from tqdm import tqdm
import warnings
from spleeter.separator import Separator
import soundfile as sf

#os.environ['CUDA_VISIBLE_DEVICES']='7'

def get_time_folder():
    '''
    获取当天日期
    :return: 日期
    '''
    curr_time = datetime.datetime.now()
    folder = str(curr_time.month).rjust(2, '0') + str(curr_time.day).rjust(2, '0')
    return folder

def compute_SNR(separator, audio):
    #audio, _ = librosa.load(p, sr=sr)
    #waveform = librosa.resample(audio, sr, 44100)
    waveform = audio
    waveform = np.expand_dims(waveform, axis=1)

    prediction = separator.separate(waveform)
    vocal, no_vocal = prediction['vocals'], prediction['accompaniment']

    rms_vocal, rms_sil = np.sqrt(np.mean(vocal ** 2)), np.sqrt(np.mean(no_vocal ** 2))
    db_voice = 20 * np.log10(rms_vocal - rms_sil)
    db_sil = 20 * np.log10(rms_sil)
    # rms_sil = rms_vocal且rms_sil = 0时,出现NaN
    snr = db_voice - db_sil
    if np.isinf(snr) or np.isnan(snr): 
        snr = -100
    return snr

def process_snr(separator, p_res, p_res_snr, sep = '|', SAMPLE_RATE=44100, seg_ignore_len=320):
    print("Start processing SNR...")
    start_time = time.time()
    processed_aid = set()
    pre_name = None
    f1 = open(p_res, 'r')
    if os.path.exists(p_res_snr):
       with open(p_res_snr) as f:
           for l in f.readlines():
               processed_aid.add(l.split('|')[0].split(os.sep)[-1][:-4]) 
    f2 = open(p_res_snr, 'a+')
    duration_segs, duration_total = [], 0
    for l in tqdm(f1.readlines()):
        l = l.strip()
        # p_aid, sid, start_time, end_time, text, align
        ts = l.split(sep)
        cur_name = ts[0].split(os.sep)[-1][:-4]
        if cur_name in processed_aid: continue
        if cur_name != pre_name:
            tmp = ts[0].split(os.sep)
            tmp[-2] = '44.1k'
            p = os.sep.join(tmp)
            audio, _ = librosa.load(p, sr = SAMPLE_RATE)
            pre_name = cur_name
            duration_total += len(audio) / SAMPLE_RATE
        s = int(float(ts[2]) * SAMPLE_RATE)
        e = int(float(ts[3]) * SAMPLE_RATE)
        audio_seg = audio[s : e]
        ts.append('-')
        if ts[4] == 'single' and e - s > seg_ignore_len:
            snr = compute_SNR(separator, audio_seg)
            ts[-1] = str(snr)
            if snr >= 30:
                duration_segs.append((e - s) / SAMPLE_RATE)
        f2.write('|'.join(ts) + '\n')
    f1.close()
    f2.close()
    print("Time cost: ", time.time() - start_time, "Speed up: ", duration_total / (time.time() - start_time))
    print(f"Segment duration / Total duration: {sum(duration_segs) / 3600} h /{duration_total / 3600} / h")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-pre", "--pre_fix", type=str, default="",
                        help="Pre-fix of corpus.")
    parser.add_argument("-o", "--out_dir", type=str, default="./",
                        help="output of corpus.")
    #parser.add_argument("-i", "--input_dir", type=str,
    #                    help="Input dir.")
    parser.add_argument("-n", "--n_groups", type=int,
                        help="Step")
    parser.add_argument("-g", "--group", type=int,
                        help="Step")
    args = parser.parse_args()
    date = get_time_folder()
    #corpus_name = args.pre_fix + '_' + date 
    corpus_name = args.pre_fix
 
    p_res = os.path.join(args.out_dir, corpus_name, f'sad_split_{args.group}.txt')
    p_res_snr = os.path.join(args.out_dir, corpus_name, f'sad_split_snr_{args.group}.txt')
   
    separator = Separator('spleeter:2stems')
    process_snr(separator, p_res, p_res_snr)
