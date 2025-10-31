"""
两份结果，seaco-paraformer带有标点和时间戳，因此以seaco-paraformer为主。
sencevoice 识别准确率预seaco-paraformer基本一致，但是略低，sensevoice 为辅

两份结果做编辑距离检查。
先做分析，哪20000条结果做对比，检查：
有多少0/1/2/3/4字以上的mismatch
mismatch 主要是哪一种
mismatch = 0 有多少是错误的。 
""" 
import re
import os, sys
import glob
import Levenshtein
from tqdm import tqdm 
import numpy as np
# from pydub import AudioSegment


def process_text(text):
    extended_punctuation = """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~，。、；：‘’“”（）《》【】！？…—"""
    pattern = re.compile("[!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~，。、；：‘’“”（）《》【】！？…—]")
    text = re.sub(pattern, "", text)
    text = text.replace(' ', '').strip()
    return text

def _split_dict_into_chunks(data, num_chunks):
    # 将字典的键转换为列表
    keys = list(data.keys())
    # 计算每个分块的大小
    chunk_size = len(keys) // num_chunks
    # 使用 np.array_split 将列表平均分成 num_chunks 份
    chunks = np.array_split(keys, num_chunks)
    
    # 生成分好的字典
    result = [{key: data[key] for key in chunk} for chunk in chunks]
    
    return result

def _convert_format_for_sence(input_file, output_file):
    with open(input_file, 'r') as f, open(output_file, 'w') as f1:
        for line in f.readlines():
            line = line.strip().split('|')
            f1.write("{}|{}\n".format(line[0], process_text(line[-1])))
    print("[INFO] : saved WER format info into {}".format(output_file))

def _convert_format_for_seaco(input_file, output_file):
    with open(input_file, 'r') as f, open(output_file, 'w') as f1:
        for line in f.readlines():
            line = line.strip().split('|')
            f1.write("{}|{}\n".format(line[0], process_text(line[-2])))
    print("[INFO] : saved WER format info into {}".format(output_file))
       
        
def convert_output_to_werformat(input_file, output_file, _type=""):
    if "sense" in _type:
        _convert_format_for_sence(input_file, output_file)
    else:
        _convert_format_for_seaco(input_file, output_file)
   
def get_saeco_info(seaco_asr_res_path):
    d = {}
    with open(seaco_asr_res_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split('|')
            d[line[0]] = line
    return d

def get_res_names_zip(base_dir):
    filenames = glob.glob(os.path.join(base_dir, '*.res'))
    
    sense_res = [filename for filename in filenames if filename.endswith("_sense.res")]
    seaco_res = [filename for filename in filenames if filename.endswith('_seaco.res')]
    
    seaco_res.sort()
    sense_res.sort()
    
    return sense_res, seaco_res


def cal_wer(f_pt, f_gt):
    d_pt, d_gt = {}, {}
    with open(f_gt) as f:
        ls = f.readlines()
        for l in tqdm(ls):
            if '|' not in l: continue
            l = l.strip()
            utt_id, sent = l.split('|')
            d_gt[utt_id] = sent.replace('%', '').replace('$', '')
    pred, true = [], []
    n_errs, _sum = 0, 0
    with open(f_pt) as f:
        ls = f.readlines()
        for l in tqdm(ls):
            if '|' not in l: continue
            if l.strip().split('|')[-1].strip() == "": continue
            l = l.strip()
            utt_id, sent = l.split('|')
            if utt_id not in d_gt: 
                # print("[{}]".format(utt_id))
                continue
            d_pt[utt_id] = sent
            n_err = Levenshtein.distance(sent, d_gt[utt_id])
            n_errs += n_err
            _sum += max(len(sent), len(d_gt[utt_id]))
            # print("id, asr, gt, n_err : ", utt_id, sent.strip(), d[utt_id].strip(), n_err)
    # print(n_errs/ _sum)
    print("[INFO] : cal WER = {}, between {} and {}".format(n_errs/ _sum, f_pt, f_gt))
    return d_pt, d_gt


def double_check_wer_based(f_pt, f_gt):
    d_pt, d_gt = {}, {}
    with open(f_gt) as f:
        ls = f.readlines()
        for l in tqdm(ls):
            if '|' not in l: continue
            l = l.strip()
            utt_id, sent = l.split('|')
            d_gt[utt_id] = sent.replace('%', '').replace('$', '')
    pred, true = [], []
    n_errs, _sum = 0, 0
    with open(f_pt) as f:
        ls = f.readlines()
        for l in tqdm(ls):
            if '|' not in l: continue
            if l.strip().split('|')[-1].strip() == "": continue
            l = l.strip()
            utt_id, sent = l.split('|')
            if utt_id not in d_gt: 
                # print("[{}]".format(utt_id))
                continue
            
            n_err = Levenshtein.distance(sent, d_gt[utt_id])
            n_errs += n_err
            _sum += max(len(sent), len(d_gt[utt_id]))
            d_pt[utt_id] = [n_err, d_gt[utt_id]]
            # print("id, asr, gt, n_err : ", utt_id, sent.strip(), d[utt_id].strip(), n_err)
    # print(n_errs/ _sum)
    try:
        print("[INFO] : cal WER = {}, between {} and {}".format(n_errs/ _sum, f_pt, f_gt))
    except:
        print("[ERROR] : cal WER faild")
    return d_pt

def insert_data(d, data, key):
    if data not in d.keys():
        d[data] = [key]
        return d
    tmp = d[data]
    tmp.append(key)
    d[data] = tmp
    return d

def display_info(occurrence_dict, info_str):
    total_occurrences = sum(len(lst) for lst in occurrence_dict.values())

    lower_4 = 0
    sorted_dict = {k: occurrence_dict[k] for k in sorted(occurrence_dict)}
    for key, lst in sorted_dict.items():
        # lst = occurrence_dict[key]
        occurrences = len(lst)
        percentage = (occurrences / total_occurrences) * 100
        print("{:15} : {:4s},  {:6d} ｜ {:.2f}%".format(info_str, str(key), occurrences, percentage))
        if key <= 4:
            lower_4 += occurrences
    print("lower than 4 : {}, {}, {}".format(lower_4, total_occurrences, lower_4 / total_occurrences))
    print('-----')


def analyse(d_res1, d_res2, d_res_gt):
    d_res_cmp = {}
    d_gt_cmp1 = {}
    d_gt_cmp2 = {}
    for key in d_res1.keys():
        if key not in d_res2.keys() or key not in d_res_gt.keys():
            continue
        
        text1, text2, text_gt = d_res1[key], d_res2[key], d_res_gt[key]
        wer_12 = Levenshtein.distance(text1, text2)
        wer_13 = Levenshtein.distance(text1, text_gt)
        wer_23 = Levenshtein.distance(text2, text_gt)
        d_res_cmp = insert_data(d_res_cmp, wer_12, key)
        d_gt_cmp1 = insert_data(d_gt_cmp1, wer_13, key)
        d_gt_cmp2 = insert_data(d_gt_cmp2, wer_23, key)
    display_info(d_res_cmp, "sense VS seaco")
    display_info(d_gt_cmp1, "sense VS gt")
    display_info(d_gt_cmp2, "seaco VS gt")
    
    return d_res_cmp 
      

def save_cmp_info(save_path, d_cmp, d_sense, d_seaco, d_gt):
    sorted_dict = {k: d_cmp[k] for k in sorted(d_cmp)}
    with open(save_path, 'w') as f:
        for key in sorted_dict.keys():
            values = sorted_dict[key]
            for i in values:
                f.write("{}|{}|{}|num,key,sense,seqco,gt\n\t{}\n\t{}\n\t{}\n".format(key, Levenshtein.distance(d_seaco[i], d_gt[i]), i, d_sense[i], d_seaco[i], d_gt[i]))
    

def analyse_sense_seaco_gt():
    sense_asr_res_path = "../data/results/test.scp_sense.res"
    seaco_asr_res_path = "../data/results/test.scp_seaco.res"
    gt_path = "/mnt/nas1/guoshaotong.gst/data/corpus-pipeline/common_voice/dev_forwer.txt"
    
    sense_wer_path = sense_asr_res_path.strip() + "_forwer.txt"
    seaco_wer_path = seaco_asr_res_path.strip() + "_forwer.txt"
    final_cmp_info = os.path.join(os.path.dirname(seaco_wer_path.rstrip('/')), "final_cmp.info")
    
    convert_output_to_werformat(sense_asr_res_path, sense_wer_path, 'sense')
    convert_output_to_werformat(seaco_asr_res_path, seaco_wer_path, 'seaco')
    
    d_sense, d_gt = cal_wer(sense_wer_path, gt_path)
    d_seaco, d_gt = cal_wer(seaco_wer_path, gt_path)
    print(len(d_sense), len(d_seaco), len(d_gt))
    
    d_cmp = analyse(d_sense, d_seaco, d_gt)
    
    save_cmp_info(final_cmp_info, d_cmp, d_sense, d_seaco, d_gt)    
        
        
def _double_check():
    sense_asr_res_path = "../data/results/test.scp_sense.res"
    seaco_asr_res_path = "../data/results/test.scp_seaco.res"
    sense_wer_path = sense_asr_res_path.strip() + "_forwer.txt"
    seaco_wer_path = seaco_asr_res_path.strip() + "_forwer.txt"
    final_cmp_info = os.path.join(os.path.dirname(seaco_wer_path.rstrip('/')), "final_cmp.info")
    
    convert_output_to_werformat(sense_asr_res_path, sense_wer_path, 'sense')
    convert_output_to_werformat(seaco_asr_res_path, seaco_wer_path, 'seaco')
    
    d_seaco = double_check_wer_based(sense_wer_path, seaco_wer_path)
    d_seaco_raw = get_saeco_info(seaco_asr_res_path)
    
    with open(final_cmp_info, 'w') as f:
        for key, value in d_seaco.items():
            if key in d_seaco_raw.keys():
                nerr = value[0]
                str_res = [d_seaco_raw[key][0]] + [value[1]] + d_seaco_raw[key][1:]
                f.write("{}|{}|{}\n".format(key, nerr, "|".join(str_res)))    # key, n_err, seaco_res_text
            else:
                print(key, d_seaco_raw.keys())
    
        
def double_check(sense_asr_res_path, seaco_asr_res_path):
    tmp_dir = os.path.join(os.path.dirname(sense_asr_res_path.rstrip('/')), "double_check")
    os.makedirs(tmp_dir, exist_ok=True)
    
    sense_wer_path = os.path.join(tmp_dir, os.path.basename(sense_asr_res_path).replace('.res', '_forwer.txt'))
    seaco_wer_path = os.path.join(tmp_dir, os.path.basename(seaco_asr_res_path).replace('.res', '_forwer.txt'))
    
    convert_output_to_werformat(sense_asr_res_path, sense_wer_path, 'sense')
    convert_output_to_werformat(seaco_asr_res_path, seaco_wer_path, 'seaco')
    
    # print("[INFO] : double checking : {}, {}".format(sense_wer_path, seaco_wer_path))
    d_seaco = double_check_wer_based(sense_wer_path, seaco_wer_path)
    d_seaco_raw = get_saeco_info(seaco_asr_res_path)
    
    res = {}

    for key, value in d_seaco.items():
        if key in d_seaco_raw.keys():
            nerr = value[0]
            if nerr < 5:
                res[key] = d_seaco_raw[key][1]
        else:
            print(key, d_seaco_raw.keys())
    return res


def collect_res_files(res_base_dir, n_groups):
    """ 防止上一步有的数据推理失败，需要按照n_groups获取文件，"""
    res = []
    for i in range(n_groups):
        seaco_path = os.path.join(res_base_dir, "post_asr_{}.scp_seaco.res".format(i))
        if not os.path.isfile(seaco_path):
            print("[INFO] : seaco path not exist, change to celan res")
            seaco_path = os.path.join(res_base_dir, "post_asr_clean_{}.res".format(i))
        if not os.path.isfile(seaco_path):
            print("[ERROR] : [{}] not exist".format(seaco_path))
        sense_path = os.path.join(res_base_dir, "post_asr_{}.scp_sense.res".format(i))
        
        if not os.path.isfile(seaco_path) or not os.path.isfile(sense_path):
            print("[ERROR] : [{}] OR [{}] not exist, skip these data".format(seaco_path, sense_path))
            res.append([None, None])
            continue
            
        res.append([seaco_path, sense_path])
    return res



def double_cehck_v1_based_zip():
    base_dir = sys.argv[1]

    # todo : 按照n_groups 获取文件，防止有一些文件不存在
    seaco_res_names, sense_res_names = get_res_names_zip(base_dir)
    res = {}
    for sense_name, seaco_name in zip(seaco_res_names, sense_res_names):
        cur_res = double_check(sense_name, seaco_name)
        res.update(cur_res)
    n_groups = len(seaco_res_names)
                                  
    l_dicts = _split_dict_into_chunks(res, n_groups)
                                  
    for idx, cur_dict in enumerate(l_dicts):
        save_path = os.path.join(base_dir, "post_asr_clean_{}.scp".format(idx))
        with open(save_path, 'w') as f:
            item_num = 0
            for key, value in cur_dict.items():
                f.write("{}\t{}\n".format(key, value))
                item_num += 1
        print("[INFO] : Saved [{}] clean asr scp into [{}]".format(item_num, save_path))
            
def get_punc_res(res_names):
    d = {}
    for seaco_name, sense_name in res_names:
        if not seaco_name or not sense_name:
            continue
        with open(seaco_name, 'r') as f:
            for line in f.readlines():
                l = line.strip().split('|')
                key = l[0]
                infos = "|".join(l[1:])
                d[key] = infos
    return d

if __name__ == "__main__":
    base_dir = sys.argv[1]
    n_groups = int(sys.argv[2])
    save_dir = base_dir
    _type = sys.argv[3]
    if _type == 'simple':
        save_dir = os.path.dirname(base_dir)
        

    # todo : 按照n_groups 获取文件，防止有一些文件不存在
    res_names = collect_res_files(base_dir, n_groups)
    res = {}
    for seaco_name, sense_name in res_names:
        if not sense_name:
            continue
            
        cur_res = double_check(sense_name, seaco_name)
        res.update(cur_res)
           
    # 调整每个进程都识别相同数据量
    l_dicts = _split_dict_into_chunks(res, n_groups)
        

    print(res_names)
    if _type == 'simple':
        d_seaco_punc_res = get_punc_res(res_names)
        for idx, cur_dict in enumerate(l_dicts):
            save_path = os.path.join(save_dir,  "post_asr_clean_{}.res".format(idx))
            with open(save_path, 'w') as f:
                item_num = 0
                for key, value in cur_dict.items():
                    f.write("{}\t{}\n".format(key, d_seaco_punc_res[key]))
                    item_num += 1
            print("[INFO] : Saved [{}] clean asr res into [{}]".format(item_num, save_path))
    else:
        for idx, cur_dict in enumerate(l_dicts):
            save_path = os.path.join(save_dir, "post_asr_clean_{}.scp".format(idx))
            with open(save_path, 'w') as f:
                item_num = 0
                for key, value in cur_dict.items():
                    f.write("{}\t{}\n".format(key, value))
                    item_num += 1
            print("[INFO] : Saved [{}] clean asr scp into [{}]".format(item_num, save_path))


            
    
        

# if __name__ == "__main__":
#     # analyse_sense_seaco_gt()
#     double_check()

        
    
        
