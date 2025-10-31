from funasr import AutoModel
import sys, os
import tqdm
import logging
import soundfile

asr_mode = "seaco_paraformer" # 'sensevoice' / 'seaco_paraformer'

if asr_mode == "seaco_paraformer":
    model = AutoModel(model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                      model_revision="v2.0.4",
                      disable_update=True
                      )
    model_en = AutoModel(model="iic/speech_paraformer-large-vad-punc_asr_nat-en-16k-common-vocab10020",
                      model_revision="v2.0.4",
                      disable_update=True
                      )  
elif asr_mode == "seaco_paraformer_straming":

    model = AutoModel(model="paraformer-zh-streaming", model_revision="v2.0.4", disable_update=True)
else:
    model = AutoModel(
        model="iic/SenseVoiceSmall",
        trust_remote_code=True,
        remote_code="./model.py",
        disable_update=True
        # vad_model="fsmn-vad",
        # vad_kwargs={"max_single_segment_time": 30000},
        # device="cuda:1",
    )

def _get_key(key_str):
    if key_str.startswith('gdsub'):
        key_str = key_str.replace('gdsub_', '')
        return "_".join(key_str.split("_")[:-1]), key_str
    return key_str, key_str

def dict_res(res_list:list) -> dict:
    d_res = {}
    for res in res_list:
        key, sub_key = _get_key(res['key'])
        if key not in d_res.keys():
            d_res[key] = {}

        text = res['text'] if asr_mode == 'seaco_paraformer' else res['text'].replace("<|", "").replace("|>", "|").split('|')[-1]
        if sub_key == key:
            d_res[key]['whole'] = {'text' : text, 'timestamp' : str(res.get('timestamp', None))}
        else:
            d_res[key][sub_key] = {'text' : text, 'timestamp' : str(res.get('timestamp', None))}
    return d_res


def asr_audio(wav_name, audio_paths):
    global model
    text = model.generate(input=wav_name, 
            batch_size_s=300)
    res = []
    for wav_path in audio_paths:
        try:
            _res = model.generate(input=wav_path, 
                batch_size_s=300)
        except:
            _res = None
        res.append(_res)
        
    return text, res


def asr_audio_v2(wav_name, audio_paths):
    global model
    text = model.generate(input=wav_name, 
            batch_size_s=300)
    res = []
    for wav_path in audio_paths:
        try:
            _res = model.generate(input=wav_path, 
                batch_size_s=300)
        except:
            _res = None
        res.append(_res)
        
    return text, res


def _asr_audio_batch(scp_path, log_name, lang='zh'):
    logger = logging.getLogger(log_name)
    global model, model_en
    res = None
    try:
        if asr_mode == 'seaco_paraformer':
            if lang == 'en':
                res = []
                logger.info('asr model_en executed for : {}'.format(scp_path))
                with open(scp_path, 'r') as f:
                    for line in f.readlines():
                        try:
                            key, wav_path = line.strip().replace('\t', '|').split('|')
                            _res = model_en.generate(input=wav_path)
                            _res[0]['key'] = key
                            res.append(_res[0])
                        except Exception as e:
                            logger.error('asr model_en executed error : [{}] for [{}]'.format(e, wav_path))
                            pass
            else:
                logger.info('asr model_zh executed for : {}'.format(scp_path))
                res = model.generate(input=scp_path, batch_size=6)
        else:
            res = model.generate(input=scp_path, cache={}, batch_size=4, merge_length_s=15)
    except Exception as e:
        logger.error('asr model executed error : [{}] for [{}]'.format(e, scp_path))
        
    if not res:
        logger.error('asr model executed error : {}'.format(scp_path))
        res = []
        with open(scp_path, 'r') as f:
            for line in f.readlines():
                key, _ = line.strip().replace('\t', '|').split('|')
                res.append({"key" : key, 'text': '', 'timestamp': []})

    return res


def _asr_audio_stream(scp_path, log_name):
    assert asr_mode == "seaco_paraformer_straming"
    logger = logging.getLogger(log_name)
    global model
    
    res = []
    with open(scp_path, 'r') as f:
        d_res = {}
        for line in f.readlines():
            key, wav_file = line.strip().replace('\t', '|').split('|')
            speech, sample_rate = soundfile.read(wav_file)
            
            chunk_size = [0, 10, 5] #[0, 10, 5] 600ms, [0, 8, 4] 480ms
            encoder_chunk_look_back = 4 #number of chunks to lookback for encoder self-attention
            decoder_chunk_look_back = 1 #number of encoder chunks to lookback for decoder cross-attention
            chunk_stride = chunk_size[1] * 960 # 600ms

            cache = {}
            total_chunk_num = int(len((speech)-1)/chunk_stride+1)
            for i in range(total_chunk_num):
                speech_chunk = speech[i*chunk_stride:(i+1)*chunk_stride]
                is_final = i == total_chunk_num - 1
                stream_res = model.generate(input=speech_chunk, cache=cache, is_final=is_final, chunk_size=chunk_size, encoder_chunk_look_back=encoder_chunk_look_back, decoder_chunk_look_back=decoder_chunk_look_back)
                d_res['text'] += stream_res['text']
                d_res['timestamp'] = []
        res.append(d_res)
    

    return res


def asr_audio_batch(scp_paths:list, log_name:str, lang='zh', streaming:bool=False):
    import pprint
    
    # streaming=True
    res = []
    for scp_path in scp_paths:
        if streaming:
            res.extend(_asr_audio_stream(scp_path, log_name))
        else:
            res.extend(_asr_audio_batch(scp_path, log_name, lang))
    d_res = dict_res(res)

    return d_res


def asr_audio_batch_single(scp_path):
    global model
    import time
    st = time.time()
    with open(scp_path, 'r') as f:
        for line in f.readlines():
            key, wav_path = line.strip().replace('\t', '|').split('|')
            res = model.generate(input=wav_path,
                         batch_size=300,
                        )
    ends = time.time()
    print(ends - st)
    return res
