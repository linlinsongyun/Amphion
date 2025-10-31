# conda activate /mnt/nas1/guoshaotong.gst/tools/anaconda-v2/envs/py38-emo
# python test_seaco_paraformer.py /mnt/nas1/guoshaotong.gst/data/corpus-pipeline/10w-hoours/segment_amap_data/seaco_asr_result/youtube/seaco_paraformer_0.scp result/ 


from funasr import AutoModel
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
        save_error_log_path = os.path.join(save_dir, \
                                           "error_log", \
                                           "{}.scp_error".format(os.path.basename(scp_path)))
        os.makedir(os.path.dirname(save_error_log_path.rstrip('/')), exist_ok = True)
        with open(save_error_log_path, 'w') as f:
            for key, value in d_error.items():
                f.write("{}|{}\n".format(key, value))
    
        save_scp_log_path = os.path.join(save_dir, \
                                         "{}.correct.scp".format(os.path.basename(scp_path)))
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
    

def get_infer_scp_file(scp_dir, cur_index, infer_type, stage):
    """
    获取asr model 读取的scp文件
      如果stage == 'raw'， 返回原始scp文件
      如果stage == 'excu'， 返回需要创建的scp文件路径
    """
    prefix = "" if stage == 'raw' else 'tmp_'
    if infer_type == "wo_punc":
        scp_path = os.path.join(scp_dir, "{}post_asr_{}.scp".format(prefix, cur_index))
    else:
        scp_path = os.path.join(scp_dir, "{}post_asr_clean_{}.scp".format(prefix, cur_index))
        if not os.path.isfile(scp_path):
            print("[INFO] : Can not find scp files for double checkd, get raw asr scp")
            scp_path = os.path.join(scp_dir, "{}post_asr_{}.scp".format(prefix, cur_index))
    
    return scp_path

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

    
def get_asr_model(infer_type, scp_path):
    """获取模型，初始化失败直接退出"""
    print("[INFO] : Initializing models for [{}]".format(scp_path))
    try:
        print('[INFO] : infer type for seaco : [{}]'.format(infer_type))
        if infer_type == "wo_punc":
            model = AutoModel(model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                              model_revision="v2.0.4",
                              )
        else:
            model = AutoModel(model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                              model_revision="v2.0.4",
                              # vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                              # vad_model_revision="v2.0.4",
                              # punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                              # punc_model_revision="v2.0.4",
                              # spk_model="damo/speech_campplus_sv_zh-cn_16k-common",
                              # spk_model_revision="v2.0.2",
                              # device=device_info
                              )
    except Exception as e:
        print("[Error] : Init model faild, {}, {}".format(scp_path, e))
        sys.exit(0)
        
    return model


def asr_punc_vad(model, scp_path, save_dir, _type="wo_punc"):
    # 先检查数据是否正确, 不正确的过滤
    scp_info, scp_path = _check_scp_format(scp_path, save_dir)

    print("[INFO] : Eatracting asr text for [{}] by seaco-paraformer...".format(_type))
    res = None
    # 识别
    try:
        if _type == 'wo_punc':
            res = model.generate(input=scp_path,
                             batch_size=6,
                            )
        else:
            res = model.generate(input=scp_path,
                             batch_size=6, #batch_size_s=300,
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
                    print("[Error] : lost data : {}".format(resi['key']))
                    continue
                # key|wav_path|text_asr|timestamp
                f.write("{}|{}|{}|{}\n".format(resi['key'], scp_info[resi['key']], resi['text'], resi['timestamp']))
            except:
                print("[ERROR] : Can not write [{}] into result files".format(resi))

                
def check_and_excute(scp_dir, cur_index, infer_type):
    # 如果结果文件不存在，说明功能没有执行过，执行全量音频的asr推理
    scp_path = get_infer_scp_file(scp_dir, cur_index, infer_type, 'raw')
    
    if infer_type == "w_punc":
        save_dir = scp_dir
        if 'clean' in scp_path:
            save_dir = os.path.dirname(scp_dir.rstrip('/'))
        save_path = os.path.join(save_dir, "post_asr_clean_{}.res".format(cur_index))
    else:
        save_path = os.path.join(scp_dir, "{}_seaco.res".format(os.path.basename(scp_path)))

    if not os.path.isfile(save_path):
        return False, scp_path, {}, save_path
    
    
    # 如果结果文件存在，记录已经执行过的数据，返回待执行的列表
    bak_res_dict = get_res(save_path)
    tmp_scp_dict = prune_excud(bak_res_dict, scp_path)
    
    ##  没有待执行列表，结束当前程序
    if len(tmp_scp_dict.keys()) == 0:
        print('[INFO] : data of [{}] has been excuted, ending...'.format(scp_path))
        return True, scp_path, {}, save_path
    
    ## 否则返回待执行列表
    tmp_scp_path = get_infer_scp_file(scp_dir, cur_index, infer_type, 'excu')
    with open(tmp_scp_path, 'w') as f:
        for key, wav_path in tmp_scp_dict.items():
            f.write("{}\t{}\n".format(key, wav_path))
    print("[INFO] : {} data excuted and {} ready for asr, excuting from tmp scp [{}]\
             ".format(len(bak_res_dict), len(tmp_scp_dict), tmp_scp_path))
    
    return False, tmp_scp_path, bak_res_dict, save_path


if __name__ == "__main__":
    scp_dir = sys.argv[1]
    cur_index = sys.argv[2]
    infer_type = sys.argv[3]
    
    bool_is_excud, scp_path, d_processed_data, res_path = check_and_excute(scp_dir, cur_index, infer_type)
    assert os.path.isfile(scp_path), "[ERORR] : scp file [{}] not exist".format(scp_path)
    
    if bool_is_excud:
        sys.exit() 

    # 获取资源：
    model = get_asr_model(infer_type, scp_path)
    
    # 识别
    res, scp_info = asr_punc_vad(model, scp_path, scp_dir, infer_type)
    
    save_res(res, scp_info, res_path, d_processed_data)

    if os.path.basename(scp_path).startswith('tmp_'):
        delete_file(scp_path)
