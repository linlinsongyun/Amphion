# conda activate /mnt/nas1/guoshaotong.gst/tools/anaconda-v2/envs/py38-emo
# python test_sencevoice.py /mnt/nas1/guoshaotong.gst/data/corpus-pipeline/10w-hoours/segment_amap_data/seaco_asr_result/youtube/seaco_paraformer_0.scp result/ 

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

import sys, os
import tqdm


def _check_scp_format(scp_path, save_dir):
    """检查输入的scp文件，
         如果格式正确，字典化数据格式
         否则，把错误信息写到error_log
       返回所有格式正确的数据
    """
    print("[INFO] : Checking input scp file [{}]....".format(scp_path))
    lines = []
    with open(scp_path, 'r') as f:
        for line in f.readlines():
            lines.append(line.strip())
       
    d_info = {}
    d_error = {}
    idx = 0
    for line in tqdm.tqdm(lines):
        idx += 1
        # num\tpath
        try:
            num, _path = line.strip().replace('\t', "|").split('|')
            if not os.path.isfile(_path.strip()):
                d_error[num] = "path not exist : {}".format(_path)
                continue
            d_info[num] = _path
        except:
            d_error[str(idx)] = "split error : {}".format(line)
        
    if len(list(d_error.keys())):
        print("[WARNING] : There is error in scp, cleaning...")
        save_error_log_path = os.path.join(save_dir, "error_log", "{}.scp_error".format(os.path.basename(scp_path)))
        os.makedir(os.path.dirname(save_error_log_path.rstrip('/')), exist_ok = True)
        with open(save_error_log_path, 'w') as f:
            for key, value in d_error.items():
                f.write("{}|{}\n".format(key, value))
    
        save_scp_log_path = os.path.join(save_dir, "{}.correct.scp".format(os.path.basename(scp_path)))
        with open(save_scp_log_path, 'w') as f:
            for key, value in tqdm.tqdm(d_info.items()):
                f.write("{}\t{}\n".format(key, value))
        return d_info, save_scp_log_path
    
    return d_info, scp_path


def delete_file(file_path):
    try:
        os.remove(file_path)
        print("[INFO] : 临时文件 {} 已成功删除。".format(file_path))
    except FileNotFoundError:
        print(f"[ERROR] : 文件 {file_path} 不存在。")
    except PermissionError:
        print(f"[ERROR] : 没有权限删除文件 {file_path}。")
    except Exception as e:
        print(f"[ERROR] : 删除文件时发生错误: {e}")


def get_res(res_path):
    d = {}
    with open(res_path, 'r') as f:
        for line in f.readlines():
            key = line.strip().split('|')[0]
            d[key] = line.strip()
    return d


def prune_excud(bak_res_dict, scp_path):
    d = {}
    with open(scp_path, 'r') as f:
        for line in f.readlines():
            key, wav_path = line.strip().replace('\t', '|').split('|')
            if key not in bak_res_dict.keys():
                d[key] = wav_path
    return d


def check_and_excute(scp_dir, cur_index):
    scp_path = os.path.join(scp_dir, "post_asr_{}.scp".format(cur_index))
    save_path = os.path.join(scp_dir, "{}_sense.res".format(os.path.basename(scp_path)))
    
    if not os.path.isfile(save_path):
        return False, scp_path, {}, save_path
    
    bak_res_dict = get_res(save_path)
    tmp_scp_dict = prune_excud(bak_res_dict, scp_path)
    if len(tmp_scp_dict.keys()) == 0:
        print('[INFO] : data of [{}] has been excuted, ending...'.format(scp_path))
        return True, scp_path, {}, save_path
    
    scp_path = os.path.join(scp_dir, "tmp_post_asr_{}.scp".format(cur_index))
    with open(scp_path, 'w') as f:
        for key, wav_path in tmp_scp_dict.items():
            f.write("{}\t{}\n".format(key, wav_path))
    print("[INFO] : {} data excuted and {} ready for asr, excuting from tmp scp [{}]\
             ".format(len(bak_res_dict), len(tmp_scp_dict), scp_path))
    
    return False, scp_path, bak_res_dict, save_path
    

def get_asr_model(scp_path):
    """获取模型，初始化失败直接退出"""
    print("[INFO] : Initializing models...")
    try:
        model = AutoModel(
            model="iic/SenseVoiceSmall",
            trust_remote_code=True,
            remote_code="./model.py",  
            # vad_model="fsmn-vad",
            # vad_kwargs={"max_single_segment_time": 30000},
            # device="cuda:1",
        )
    except Exception as e:
        print("[ERROR] : Init model faild, {}, {}".format(scp_path, e))
        sys.exit(0)
        
    return model


def asr_punc_vad(model, scp_path, save_dir):
    # 先检查数据是否正确, 不正确的过滤
    scp_info, scp_path = _check_scp_format(scp_path, save_dir)

    print("[INFO] : Eatracting asr text for sensevoice...")
    res = None
    # 识别
    try: 
        res = model.generate(input=scp_path,
                             cache={},
                             batch_size=2,
                             merge_length_s=15,
                        )
    except Exception as e:
        print('[ERROR] : asr model executed error : [{}] for [{}]'.format(e, scp_path))
    if not res:
        print('[ERROR] : asr model executed error : {}'.format(scp_path))
        
    return res, scp_info


def save_res(res, scp_info, save_path, d_processed_data):
    os.makedirs(os.path.dirname(save_path.rstrip('/')), exist_ok=True)
    
    print("[INFO] : Saving asr result into : [{}]...".format(save_path))
    with open(save_path, "w") as f:
        for key, value in d_processed_data.items():
            f.write("{}\n".format(value.strip()))
        for resi in tqdm.tqdm(res):
            try:
                if resi['key'] not in scp_info.keys():
                    print("[ERROR] : lost data : {}".format(resi['key']))
                    continue
                # key|wav_path|text_asr|timestamp
                f.write("{}|{}|{}\n".format(resi['key'], scp_info[resi['key']], resi['text'].replace("<|", "").replace("|>", "|")))
            except:
                print("[ERROR] : Can not write [{}] into result files".format(resi))


if __name__ == "__main__":
    scp_dir = sys.argv[1]
    cur_index = sys.argv[2]

    bool_is_excud, scp_path, d_processed_data, res_path = check_and_excute(scp_dir, cur_index)
    assert os.path.isfile(scp_path), "[ERROR] : scp file [{}] not exist".format(scp_path)
    if bool_is_excud:
        sys.exit() 
        
    # 获取资源：
    model = get_asr_model(scp_path)
    
    # 识别
    res, scp_info = asr_punc_vad(model, scp_path, scp_dir)
    
    save_res(res, scp_info, res_path, d_processed_data)

    if os.path.basename(scp_path).startswith('tmp_'):
        delete_file(scp_path)

