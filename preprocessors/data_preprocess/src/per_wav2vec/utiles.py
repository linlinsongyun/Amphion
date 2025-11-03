import torch
import torch.nn.functional as F
import numpy as np
import io
import scipy
from pydub import AudioSegment

MAX_VALUE = 32768.0
SAMPLE_RATE_PYANNOTE = 16000

def convert_arr_to_audiosegment(arr):
    wav_io = io.BytesIO()
    scipy.io.wavfile.write(wav_io, SAMPLE_RATE_PYANNOTE, (arr * MAX_VALUE).astype(np.int16))
    wav_io.seek(0)
    sound = AudioSegment.from_wav(wav_io)
    #assert len(sound) / 1000 * self.SAMPLE_RATE_PYANNOTE == len(arr)
    return sound

def convert_audiosegment_to_arr(seg):
    audio = np.array(seg.get_array_of_samples()).astype(np.float32) / MAX_VALUE
    return audio

def remove_sil(l):
    # print('pre:', l)
    sil_id = 577
    if l[0] == sil_id:
        l = l[1:]
    if l[-1] == sil_id:
        l = l[:-1]
    # print('after:', l)
    return l
    
    
def phr(lista, listb):
    edit_distance = min_edit_distance(lista, listb)
    return edit_distance/len(listb)

def min_edit_distance(lista, listb):
    lista = remove_sil(lista)
    listb = remove_sil(listb)

    m, n = len(lista), len(listb)
    
    # 创建二维DP数组
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # 初始化DP数组边界情况
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # 填充DP数组
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if lista[i - 1] == listb[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],    # 删除
                                   dp[i][j - 1],    # 插入
                                   dp[i - 1][j - 1])# 替换
    
    return dp[m][n]



def beam_search_decode(logits, beam_width=5):
    """
    使用Beam Search解码模型输出。

    参数:
    - logits: `torch.Tensor`形状为[T, D]，其中T是时间步数，D是词汇表的大小
    - beam_width: Beam宽度

    返回:
    - best_sequence: `torch.Tensor`形状为[T, 1]，表示解码出的最佳序列
    """
    print(logits.size())
    T, D = logits.size()
    
    # 初始化取前 beam_width 个的probabilities和对应的indices
    probs, indices = F.log_softmax(logits[0], dim=-1).topk(beam_width)
    sequences = indices.unsqueeze(1)  # [beam_width, 1]
    
    for t in range(1, T):
        # 初始化新的候选序列和其概率
        all_candidates = []
        
        for i in range(beam_width):
            current_prob = probs[i]
            current_sequence = sequences[i]
            # print('current_sequence', current_sequence)

            # Expand the current probability and sequence with the next step's logits
            logits_t = F.log_softmax(logits[t], dim=-1)
            prob_t, index_t = logits_t.topk(beam_width)
            
            # print('index_t', index_t)
            
            for j in range(beam_width):
                new_prob = current_prob + prob_t[j]
                new_sequence = torch.cat((current_sequence, index_t[j].unsqueeze(0)))
                all_candidates.append((new_prob, new_sequence))
        
        # 从候选池里选最有可能的beam_width个序列
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        all_candidates = all_candidates[:beam_width]
        
        # 更新probabilities和sequences
        probs, sequences = zip(*all_candidates)
        probs = torch.stack(probs)
        sequences = torch.stack(sequences)
    
    # 返回具有最高概率的序列
    best_sequence = sequences[0]
    return best_sequence

