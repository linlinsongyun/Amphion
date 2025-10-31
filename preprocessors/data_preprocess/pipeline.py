import os
import io
import scipy.io.wavfile
import argparse
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_nonsilent
import numpy as np
from pyannote.audio.utils.signal import Binarize
from pyannote.audio import Model, Pipeline
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio import Inference
import datetime
import librosa
import soundfile as sf
import torch
import time
from tqdm import tqdm
import warnings
#from spleeter.separator import Separator
from modelscope.pipelines import pipeline as pipeline_modelscope
from modelscope.utils.constant import Tasks
from multiprocessing import Pool
#os.environ['CUDA_VISIBLE_DEVICES']='7'


class WaveProcessor:
    def __init__(self, input_dir, out_dir, name, transcript=False, step=1, group=1, n_groups=3):
        self.transcript = transcript

        self.MAX_VALUE = 32768.0
        self.SAMPLE_RATE_PYANNOTE = 16000
        self.SAMPLE_RATE = 44100
        self.seg_max_len = int(20 * 1000)
        #self.seg_min_len = int(0.8 * 1000)
        self.seg_ignore_len = int(0.5 * 1000)
        self.keep_silence = 100 # milisecond
        
        token = ""
        model = Model.from_pretrained("pyannote/segmentation", use_auth_token=token)
        self.pipeline = VoiceActivityDetection(segmentation=model)
        HYPER_PARAMETERS = {
        # onset/offset activation thresholds
        "onset": 0.82, "offset": 0.4,
        # remove speech regions shorter than that many seconds.
        "min_duration_on": 0.0,
        # fill non-speech regions shorter than that many seconds.
        "min_duration_off": 0.0,
        }
        self.pipeline.instantiate(HYPER_PARAMETERS)

        self.pipeline_scd = Pipeline.from_pretrained("pyannote/speaker-segmentation", use_auth_token=token) 
        self.pipeline_asr = pipeline_modelscope(
                         task=Tasks.auto_speech_recognition,
                         model='damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',)
        self.pipeline_align = pipeline_modelscope(
                         task=Tasks.speech_timestamp,
                         model='damo/speech_timestamp_prediction-v1-16k-offline',)
        
        #self.separator = Separator('spleeter:2stems')

        self.duration_total = 0
        self.duration_segs = []
        self.p_duration_segs = os.path.join(out_dir, name, f"duration_segs_{group}.npy")
        self.input_dir = input_dir
        self.wav_dir = os.path.join(out_dir, name, 'wav_dir')
        self.norm_dir = os.path.join(out_dir, name, 'norm_dir')
        self.wav_dir_44k = os.path.join(self.wav_dir, '44.1k')
        self.wav_dir_16k = os.path.join(self.wav_dir, '16k')
        self.norm_dir_44k = os.path.join(self.norm_dir, '44.1k')
        self.norm_dir_16k = os.path.join(self.norm_dir, '16k')
        self.corpus_dir = os.path.join(out_dir, name, 'corpus_dir')
        for dir in (self.wav_dir, self.norm_dir, self.corpus_dir, 
                    self.wav_dir_44k, self.wav_dir_16k, self.norm_dir_44k, self.norm_dir_16k):
            os.makedirs(dir, exist_ok=True)
        print(group, n_groups)
        assert group <= n_groups
        n_files = len(os.listdir(self.norm_dir_16k))
        n_files_per_group = n_files // n_groups
        start_ind = int((group - 1) * n_files_per_group)
        end_ind = n_files if group == n_groups else int(group * n_files_per_group)
        self.filelist = sorted(os.listdir(self.norm_dir_16k))[start_ind : end_ind]       

        self.p_res = os.path.join(out_dir, name, f'sad_split_{group}.txt')
        self.p_res_snr = os.path.join(out_dir, name, f'sad_split_snr_{group}.txt')
        p_res_snr_asr = os.path.join(out_dir, name, f'sad_split_snr_asr_align_{group}.txt')
        self.processed_aid, self.processed_sid_asr = set(), set()
        if step == 1:
            if os.path.exists(self.p_res):
                with open(self.p_res) as f:
                    for l in f.readlines():
                        self.processed_aid.add(l.split('|')[0].split(os.sep)[-1][:-4])
            self.f_res = open(self.p_res, 'a+')
        elif step == 3:
            self.pipeline_osd = Pipeline.from_pretrained("pyannote/overlapped-speech-detection", use_auth_token=token)
            self.f_res = open(self.p_res)
            self.f_res_snr = open(self.p_res_snr) 
            if os.path.exists(p_res_snr_asr):
                with open(p_res_snr_asr) as f:
                    for l in f.readlines():
                        self.processed_sid_asr.add(l.split('|')[1][:-4])
            self.f_res_snr_asr = open(p_res_snr_asr, 'a+')
        else:
            raise

        self.f_res_json = open(os.path.join(out_dir, name, 'AmapSpeech.json'), 'w')
        self.L3, self.L2, self.L1, self.L0 = 30, 20, 10, 0
        #self.f0 = open(os.path.join(out_dir, 'L0_data_list.txt'), 'w')
        #self.f1 = open(os.path.join(out_dir, 'L1_data_list.txt'), 'w')
        #self.f2 = open(os.path.join(out_dir, 'L2_data_list.txt'), 'w')
        #self.f3 = open(os.path.join(out_dir, 'L3_data_list.txt'), 'w')

        self.s = time.time()

    def speaker_osd(self, audio):
        waveform = torch.from_numpy(audio).unsqueeze(0).float()
        audio_in_memory = {"waveform": waveform, "sample_rate": 16000}
        output = self.pipeline_osd(audio_in_memory)
        #print(output, '1'*20)
        for speech in output.get_timeline().support():
            #print(speech.start, speech.end, speech)
            return True
        return False
 
    def save_statistic(self):
        np.save(self.p_duration_segs, self.duration_segs)

    def destroy(self):
        try:
            self.f_res.close()
            self.f_res_snr.close()
            self.f_res_snr_asr.close()
        except:
            pass
        self.f_res_json.close()
        #self.f0.close()
        #self.f1.close()
        #self.f2.close()
        #self.f3.close()
        print("Time cost: ", time.time() - self.s, "Speed up: ", self.duration_total / (time.time() - self.s))
        print(f"Segment duration / Total duration: {sum(self.duration_segs) / 3600} h /{self.duration_total / 3600} / h")
    
    def norm(self, inputPath, outputPath):
        '''
        归一化音频
        :param inputPath: 输入路径
        :param outputPath: 输出路径
        :return: none
        '''
        cmdline = f"sox --norm=-1 {inputPath} {outputPath}"
        #print(cmdline)
        os.system(cmdline)

    def compute_SNR(self, audio):
        #audio = np.array(audio.get_array_of_samples()) / self.MAX_VALUE
        #waveform = librosa.resample(audio, self.SAMPLE_RATE, 44100)
        waveform = audio
        waveform = np.expand_dims(waveform, axis=1)
        print(waveform, waveform.shape)
        prediction = self.separator.separate(waveform)
        vocal, no_vocal = prediction['vocals'], prediction['accompaniment']

        rms_vocal, rms_sil = np.sqrt(np.mean(vocal ** 2)), np.sqrt(np.mean(no_vocal ** 2))
        db_voice = 20 * np.log10(rms_vocal - rms_sil)
        db_sil = 20 * np.log10(rms_sil)
        # rms_sil = rms_vocal且rms_sil = 0时,出现NaN
        snr = db_voice - db_sil
        return snr
    
    def create_segments(self, audio, outputDir, speech, filename):
        '''
        根据SAD结果，切割音频
        :param audio: 读取的音频文件
        :param outputDir: 输出文件夹
        :param speech: SAD结果
        :param file: 前缀
        :return:
        '''
        print("Start force-split...")
        cur_num = 1
        for time_pair in speech.get_timeline().support():
            time_pair = list(time_pair)
            #print(time_pair[0], time_pair[1])
            start = int(time_pair[0] * 1000)
            end = int(time_pair[1] * 1000)
            start = max(0, start - self.keep_silence)
            end = min(len(audio), end + self.keep_silence)
            # if end - start >= 10000:
            #     continue
            cur_segment = audio[start : end]
            #cur_segment.export(f'{outputDir}/{outputName}.wav', format='wav')
            print(len(cur_segment))
            segs = self.force_split(cur_segment)
            #time_stamps = self.force_split(cur_segment)
            print(cur_num, len(segs))
            for seg in segs:
                post_fix = str(cur_num).rjust(5,'0')
                output_name = f'{filename}_{post_fix}.wav'
                seg = self.pad_silence(seg)
                #if self.speaker_change_detection(self.convert_audiosegment_to_arr(seg)):
                    #self.f.write()
                #    continue
                #self.log(self.compute_SNR(seg), filename, time_pair[0], time_pair[1])
                print(os.path.join(self.corpus_dir, output_name))
                seg.export(os.path.join(self.corpus_dir, output_name), format='wav', sample_width=2)
                cur_num += 1

    def create_time_stamps(self, audio_pydub, audio, outputDir, speech, filename):
        '''
        根据SAD结果，切割音频
        :param audio: 读取的音频文件
        :param outputDir: 输出文件夹
        :param speech: SAD结果
        :param file: 前缀
        :return:
        '''
        print("Start force-split...")
        cur_num = 1
        for time_pair in tqdm(speech.get_timeline().support()):
            time_pair = list(time_pair)
            #print(time_pair[0], time_pair[1])
            start = int(time_pair[0] * 1000)
            end = int(time_pair[1] * 1000)
            # if end - start >= 10000:
            #     continue
            cur_segment = audio_pydub[start : end]
            #cur_segment.export(f'{outputDir}/{outputName}.wav', format='wav')
            #print(len(cur_segment), start, end)
            #print(len(cur_segment) / 1000, start/1000, end/ 1000, '------------->')
            time_stamps = self.detect_nonsil(cur_segment, start, end)
            ##print(len(audio), [(e/1000 - s/1000, s/1000, e/1000) for (s,e) in time_stamps])
            for (s, e) in time_stamps:
                assert cur_num < 100000
                post_fix = str(cur_num).rjust(5,'0')
                output_name = f'{filename}_{post_fix}.wav'
                seg = audio_pydub[s : e]
                audio_seg = audio[int(s / 1000 * self.SAMPLE_RATE_PYANNOTE) : int(e / 1000 * self.SAMPLE_RATE_PYANNOTE)]
                #seg = self.pad_silence(seg)
                out_path = os.path.join(self.corpus_dir, output_name)
                # longaudio name, segment audio, name, start time, end time, is_multi, asr_text
                long_audio_path = os.path.join(self.norm_dir_16k, filename + '.wav')
                seg_name = output_name[:-4]
                line = [long_audio_path, seg_name, str(s / 1000), str(e / 1000), '-', '-', '-']
                if e - s > self.seg_ignore_len:   
                    has_multi_speaker = self.speaker_change_detection(audio_seg)
                    if has_multi_speaker:
                        line[4] = 'multi'
                    else:
                        line[4] = 'single'
                        if self.transcript and len(seg) > self.seg_ignore_len:
                            text = self.asr(audio_seg) 
                            if text:
                                text_align = self.alignment(audio_seg, text)
                                line[5], line[6] = str(text), str(text_align)
                self.f_res.write('|'.join(line) + '\n')
                #self.log(self.compute_SNR(seg), filename, time_pair[0], time_pair[1])
                self.duration_segs.append((e - s) / 1000)
                #print(oout_path, s / 1000, e / 1000, (e - s) / 1000)
                #seg.export(os.path.join(self.corpus_dir, output_name), format='wav')
                cur_num += 1
    
    def process_snr(self, sep = '|'):
        print("Start processing SNR...")
        pre_name = None
        for l in tqdm(self.f_res.readlines()):
            l = l.strip()
            # p_aid, sid, start_time, end_time, text, align
            ts = l.split(sep)
            cur_name = ts[0].split(os.sep)[-1]
            if cur_name != pre_name:
                p = ts[0].replace('16k', '44.1k')
                audio, _ = librosa.load(p, sr = self.SAMPLE_RATE)
                pre_name = cur_name
                self.duration_total += len(audio) / self.SAMPLE_RATE
            s = int(float(ts[2]) * self.SAMPLE_RATE)
            e = int(float(ts[3]) * self.SAMPLE_RATE)
            audio_seg = audio[s : e]
            ts.append('-')
            if e - s > self.seg_ignore_len:
                snr = self.compute_SNR(audio_seg)
                ts[-1] = str(snr)
                #if snr > 5:
                #    text = self.asr(audio_seg)
                #    if text:
                #        text_align = self.alignment(audio_seg, text)
                #        ts[5], ts[6] = str(text), str(text_align)
                if snr >= 30:
                    self.duration_segs.append((e - s) / 1000)
            self.f_res_snr.write('|'.join(ts), '\n')
 
    def process_transcript(self, sep = '|'):
        print("Start processing ASR, Alignment...")
        pre_name = None
        for l in tqdm(self.f_res_snr.readlines()):
            l = l.strip()
            # p_aid, sid, start_time, end_time, text, align
            ts = l.split(sep)
            cur_name = ts[0].split(os.sep)[-1]
            sid = ts[1][:-4]
            if sid in self.processed_sid_asr: continue 
            if cur_name != pre_name:
                p = ts[0]
                audio, _ = librosa.load(p, sr = self.SAMPLE_RATE_PYANNOTE)
                pre_name = cur_name
                self.duration_total += len(audio) / self.SAMPLE_RATE_PYANNOTE
            s = int(float(ts[2]) * self.SAMPLE_RATE_PYANNOTE)
            e = int(float(ts[3]) * self.SAMPLE_RATE_PYANNOTE)
            audio_seg = audio[s : e]
            snr = ts[-1]
            if snr != '-' and float(snr) > 20 and float(ts[3]) - float(ts[2]) > self.seg_ignore_len / 1000:
                spk_osd = self.speaker_osd(audio_seg)
                if spk_osd:
                    print("Has overlaped speaker: ", ts[1])
                    continue
                self.duration_segs.append(float(ts[3]) - float(ts[2]))
                text = self.asr(audio_seg) if ts[5] == '-' else ts[5]
                ts[5] = str(text)
                if text:
                    try:
                        text_align = self.alignment(audio_seg, text)
                        ts[6] = str(text_align)
                    except:
                        pass
            self.f_res_snr_asr.write('|'.join(ts) + '\n')

    def speaker_change_detection(self, audio):
        pad = np.array([0.] * int(0.2 * 16000))
        audio = np.hstack((pad, audio, pad))
        waveform = torch.from_numpy(audio).unsqueeze(0).float()
        audio_in_memory = {"waveform": waveform, "sample_rate": self.SAMPLE_RATE_PYANNOTE}
        output = self.pipeline_scd(audio_in_memory)
        s = set()
        for turn, _, speaker in output.itertracks(yield_label=True):
            if len(s) > 1:
                #print(turn.start, turn.end, speaker)
                #print("#"*20 + "Speaker changed!!!" + '#'*20)
                return True
            s.add(speaker)
        return False

    def log(self, snr, filename, start_time, end_time):
        line = '|'.join([filename, str(start_time), str(end_time), str(snr)])
        print(line)
        if snr > self.L3:
            self.f3.write(line + '\n')
        if snr > self.L2:
            self.f2.write(line + '\n')
        if snr > self.L1:
            self.f1.write(line + '\n')
        else:
            self.f0.write(line + '\n')

    def youtube_download(self, urls_txt, downloadDir):
        '''
        根据youtube url，下载m4a音频文件
        :param urls_txt: 存url的txt文件
        :param downloadDir: 下载文件夹
        :return: none
        '''
        urls = []
        with open(urls_txt, 'r') as f:
            for line in f:
                urls.append(line)
        for url in urls:
            cmd = f"youtube-dl -o {downloadDir}/%(title)s.%(ext)s -f 140 {url}"
            os.system(cmd)

    def rename(self, n, pre_fix):
        '''
        重命名downloadDir中的文件，方便后续处理
        :param downloadDir: 目标文件夹
        :param n: 当前编号
        :param pre_fix: 数据前缀
        :return: none
        '''
        filelist = os.listdir(self.input_dir)
        cur_num = n
        for file in filelist:
            new_name = pre_fix + str(cur_num).rjust(5, '0')
            os.rename(f'{self.input_dir}/{file}', f'{self.input_dir}/{new_name}.m4a')
            cur_num += 1
    
    def os_system_cmd(self, cmd):
        os.system(cmd)

    def conversion(self):
        '''
        把m4a文件转为16bit, 24k, 单声道wav文件
        :param downloadDir: 原始数据文件夹
        :param wavDir: wav数据文件夹
        :return: none
        '''
        s = time.time()
        filelist = os.listdir(self.input_dir)
        cmds = []
        for file in filelist:
            #assert file[-3:] in {"wav", "mp3", "mp4", "m4a"}
            #if '.m4a' in file:
            if True:
                path_in = os.path.join(self.input_dir, file)
                path_out = os.path.join(self.wav_dir_44k, file.split('.')[0] + '.wav')
                path_out_16k = os.path.join(self.wav_dir_16k, file.split('.')[0] + '.wav')
                cmd = f'ffmpeg -i {path_in} -ac 1 -ar {self.SAMPLE_RATE} -r 16 {path_out}'
                cmd2 = f'ffmpeg -i {path_in} -ac 1 -ar {self.SAMPLE_RATE_PYANNOTE} -r 16 {path_out_16k}'
                if not os.path.exists(path_out):
                    cmds.append(cmd)
                if not os.path.exists(path_out_16k):
                    cmds.append(cmd2)
        #with Pool(self.numOfPools) as p:
        #    p.imap_unordered(self.os_system_cmd, cmds)
        for cmd in cmds:
            os.system(cmd)
        print('Time cost for Conversion:', time.time() - s)

    def normalization(self):
        '''
        归一化音频
        :param wavDir: 输入文件夹
        :param normDir: 输出文件夹
        :return: none
        '''
        s = time.time()
        ts = []
        filelist = os.listdir(self.wav_dir_44k)
        for file in filelist:
            if '.wav' in file:
                path_in = os.path.join(self.wav_dir_44k, file)
                path_out = os.path.join(self.norm_dir_44k, file)
                if not os.path.exists(path_out):
                    self.norm(path_in, path_out)
        filelist = os.listdir(self.wav_dir_16k)
        for file in filelist:
            if '.wav' in file:
                path_in = os.path.join(self.wav_dir_16k, file)
                path_out = os.path.join(self.norm_dir_16k, file)
                if not os.path.exists(path_out):
                    self.norm(path_in, path_out)
        print('Time cost for Conversion:', time.time() - s)

    def process_sad_w_denoise(self, denoiseDir, corpusDir):
        '''
        根据sad结果，进行切割
        :param denoiseDir: 输入文件夹
        :param corpusDir: 输出文件夹
        :return: none
        '''
        model = Model.from_pretrained("pyannote/segmentation", use_auth_token=self.token)
        pipeline = VoiceActivityDetection(segmentation=model)
        HYPER_PARAMETERS = {
        # onset/offset activation thresholds
        "onset": 0.82, "offset": 0.82,
        # remove speech regions shorter than that many seconds.
        "min_duration_on": 0.1,
        # fill non-speech regions shorter than that many seconds.
        "min_duration_off": 0.1,
        }
        pipeline.instantiate(HYPER_PARAMETERS)
        file_dirs = os.listdir(os.path.join(denoiseDir, 'htdemucs'))
        for _dir in file_dirs:
            print(f'begin processing {_dir}')
            pre_fix = _dir
            filepath = os.path.join(denoiseDir, 'htdemucs', _dir, 'vocals.wav')
            speech = pipeline(filepath)
            #print(speech, type(speech), dir(speech))
            audio = AudioSegment.from_wav(filepath)
            self.create_segments(audio, corpusDir, speech, pre_fix)
            print(f'finish processing {_dir}')
    
    def convert_arr_to_audiosegment(self, arr):
        wav_io = io.BytesIO()
        scipy.io.wavfile.write(wav_io, self.SAMPLE_RATE_PYANNOTE, (arr * self.MAX_VALUE).astype(np.int16))
        wav_io.seek(0)
        sound = AudioSegment.from_wav(wav_io)
        #assert len(sound) / 1000 * self.SAMPLE_RATE_PYANNOTE == len(arr)
        return sound

    def convert_audiosegment_to_arr(self, seg):
        audio = np.array(seg.get_array_of_samples()).astype(np.float32) / self.MAX_VALUE
        return audio

    def process_sad(self):
        '''
        根据sad结果，进行切割
        :param normDir: 输入文件夹
        :param corpusDir: 输出文件夹
        :return: none
        '''
        #filelist = os.listdir(self.norm_dir_16k)
        for file in tqdm(self.filelist):
            print(f'Begin processing {file}')
            filepath = os.path.join(self.norm_dir_16k, file)
            filename = file.split('.')[0]
            # Continue to process 
            if filename in self.processed_aid: continue
            file_dic = {'uri': f'{file}', 'audio': filepath}
            #sad_scores = sad(file_dic)
            #binarize = Binarize(offset=0.82, onset=0.82, log_scale=True,
            #                    min_duration_off=0.1, min_duration_on=0.1)
            #speech = binarize.apply(sad_scores, dimension=1)
            audio, _ = librosa.load(filepath, sr = self.SAMPLE_RATE_PYANNOTE)
            if len(audio) < self.seg_ignore_len / 1000 * self.SAMPLE_RATE_PYANNOTE: # 小于500ms的不考虑
                continue
            self.duration_total += len(audio) / self.SAMPLE_RATE_PYANNOTE
            #print("Start resampling...")
            #waveform = librosa.resample(audio, self.SAMPLE_RATE, self.SAMPLE_RATE_PYANNOTE)
            #print("finished resampling...")
            waveform = torch.from_numpy(audio).unsqueeze(0)
            audio_in_memory = {"waveform": waveform, "sample_rate": self.SAMPLE_RATE_PYANNOTE}
            speech = self.pipeline(audio_in_memory)
            #speech = self.pipeline(file_dic)
            #print(speech, type(speech), dir(speech))
            #speech = binarize.apply(sad_scores, dimension=1)
            #audio = AudioSegment.from_wav(filepath)
            audio_pydub = self.convert_arr_to_audiosegment(audio)
            #print(len(audio), len(audio_pydub))
            #self.create_segments(audio_pydub, self.corpus_dir, speech, filename)
            self.create_time_stamps(audio_pydub, audio, self.corpus_dir, speech, filename)
            print(f'Finish processing {file}')

    def seperation_denoise(self, normDir, denoised_dir):
        for file in os.listdir(normDir):
            filepath = os.path.join(normDir, file)
            os.system(f"demucs --two-stems=vocals --out={denoised_dir} {filepath}")
    
    @classmethod
    def detect_nonsil(cls, sound, start_time, end_time, min_silence_len=500, silence_thresh=-50, delta_sil_len=50, delta_sil_thresh=5, keep_silence=100, seg_max_len=20000):
        # start_time, end_time is absolute in milisecond
        if end_time - start_time < seg_max_len:
            #self.time_stamps_valid.append([start_time, end_time])
            return [[start_time, end_time]]
        if min_silence_len <= 0:
            #warnings.warn("#"*20 + "Random split!!!" + '#'*20) 
            print("#"*20 + "Random split!!!" + '#'*20)
            return [[start_time, start_time + len(sound) // 2], [start_time + len(sound) // 2, end_time]] 
        print(start_time, end_time, len(sound) / 1000, silence_thresh, min_silence_len, '*'*10)
        time_stamps = detect_nonsilent(sound,
            # must be silent for at least 400ms
            min_silence_len = min_silence_len,#800,#100,
            # consider it silent if quieter than -50 dBFS
            silence_thresh = silence_thresh, #dBFS-100,
        )
        time_stamps_valid = []
        for i in range(len(time_stamps)):
            s, e = time_stamps[i]
            s, e = max(0, s - keep_silence), min(len(sound), e + keep_silence)
            seg = sound[s : e]
            #min_silence_len = max(1, min_silence_len - delta_sil_len)
            min_silence_len = min_silence_len - delta_sil_len
            silence_thresh = silence_thresh + delta_sil_thresh
            time_stamps_valid.extend(cls.detect_nonsil(seg, start_time + s, start_time + e, min_silence_len = min_silence_len, silence_thresh = silence_thresh))
        return time_stamps_valid

    def force_split(self, sound, min_silence_len=500, silence_thresh=-50):
        #print('Sound dBFS: ', sound.dBFS)
        if len(sound) < self.seg_min_len:
            return []
        elif len(sound) < self.seg_max_len:
            return [sound]
        if min_silence_len <= 0:
            print("#"*20 + "Random split!!!" + '#'*20)
            l = len(sound)
            return [sound[:l], sound[l:]]

        segs = split_on_silence(sound,
            # must be silent for at least 400ms
            min_silence_len = min_silence_len,#800,#100,
            # consider it silent if quieter than -50 dBFS
            silence_thresh = silence_thresh, #dBFS-100,
        )
        ids = [i for i, seg in enumerate(segs) if len(seg) > self.seg_max_len]
        segs_valid = [seg for seg in segs if self.seg_min_len <= len(seg) <= self.seg_max_len] # 小于0.8s的片段会丢弃
        for i in ids: 
            seg = segs[i]
            min_silence_len = max(10, min_silence_len - 100)
            tmp = self.force_split(seg, min_silence_len = min_silence_len, silence_thresh = silence_thresh + 10)
            segs_valid.extend(tmp)
        return segs_valid

    def pad_silence(self, audio, dur = 200):
        padded_sil = AudioSegment.silent(duration=dur)
        return padded_sil + audio + padded_sil
    
    def asr(self, audio):
        audio_pad = np.array([0.] * int(self.SAMPLE_RATE_PYANNOTE * 0.2))
        audio = np.hstack((audio_pad, audio, audio_pad))
        rec_result = self.pipeline_asr(audio_in=audio)
        return rec_result['text'] if 'text' in rec_result else None
    def is_chinese(self, char):
        if '\u4e00' <= char <= '\u9fff': # '中文'
            return True
        return False
 
    def prepare_align_text(self, line):
        # you are you know of嗯like a minor incident
        line = line.strip()
        res = ''
        for i, char in enumerate(line[:-1]):
            char_next = line[i+1]
            res += char
            if self.is_chinese(char):
                res += ' '
            elif self.is_chinese(char_next):
                res += ' '
        return res + line[-1]

    def alignment(self, audio, text):
        #text_in = ' '.join(tuple(text))
        text_in = self.prepare_align_text(text)
        pcm = (audio * self.MAX_VALUE).astype(np.int16)
        rec_result = self.pipeline_align(audio_in=pcm.tobytes(), text_in=text_in)
        return rec_result['text']

def get_time_folder():
    '''
    获取当天日期
    :return: 日期
    '''
    curr_time = datetime.datetime.now()
    folder = str(curr_time.month).rjust(2, '0') + str(curr_time.day).rjust(2, '0')
    return folder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--urls_txt", type=str, default="urls.txt",
                        help="The text file containing urls to download.")
    parser.add_argument("-offset", "--offset", type=float, default=0.82,
                        help="SAD parameters.")
    parser.add_argument("-onset", "--onset", type=float, default=0.82,
                        help="SAD parameters.")
    parser.add_argument("-pre", "--pre_fix", type=str, default="",
                        help="Pre-fix of corpus.")
    parser.add_argument("-o", "--out_dir", type=str, default="./",
                        help="output of corpus.")
    parser.add_argument("-i", "--input_dir", type=str,
                        help="Input dir.")
    parser.add_argument("-s", "--step", type=int,
                        help="Step")
    parser.add_argument("-n", "--n_groups", type=int,
                        help="Step")
    parser.add_argument("-g", "--group", type=int,
                        help="Step")
    args = parser.parse_args()
    date = get_time_folder()
    #corpus_name = args.pre_fix + '_' + date
    corpus_name = args.pre_fix
    # 转格式后文件夹
    #wav_dir = os.path.join(args.save_dir, folder, 'wav_data')
    # 归一化后文件夹
    #norm_dir = os.path.join(args.save_dir, folder, 'norm_data')
    # 声源分离后文件夹
    #denoiseDir = os.path.join(args.save_dir, folder, 'denoise_data')
    # 存语料的文件夹
    #corpus_dir = os.path.join(args.save_dir, folder, f'{date}_corpus')
    wave_processor = WaveProcessor(args.input_dir, args.out_dir, corpus_name, step=args.step, group = args.group, n_groups = args.n_groups)
    if args.step == 1:
        # 转换音频格式 m4a, mp3 --> wav
        #wave_processor.conversion()
        # 归一化音频
        #wave_processor.normalization()
        # 声源分离
        # seperation_denoise(normDir, denoiseDir)
        # 根据sad结果，切割音频
        # pyannote.audio: 核心的音频分析工具。
        # VoiceActivityDetection: 用于从长音频中找出“哪里有人说话”。
        # speaker-segmentation: 用于判断一个片段内有几个说话人，实现了说话人变化检测。
        # overlapped-speech-detection: 用于检测是否存在声音重叠（多人同时说话）

        wave_processor.process_sad()
        wave_processor.save_statistic()
        # SNR检测
        #wave_processor.process_snr()
    else:
        # ASR转录, 强制对齐
        wave_processor.process_transcript()

    wave_processor.destroy()   

if __name__ == '__main__':
    main()


