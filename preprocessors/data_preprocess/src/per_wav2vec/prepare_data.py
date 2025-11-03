from datasets import load_from_disk, load_dataset, Audio
from transformers import Wav2Vec2Processor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments
from transformers import Trainer
import os
import numpy as np
import math
import faulthandler
faulthandler.enable()
from utiles import convert_arr_to_audiosegment, convert_audiosegment_to_arr
from pydub.effects import normalize

tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
print('tokenizer', tokenizer)

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=False, return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


from_train   = 'metadata.csv'
from_test    = 'metadata.csv'
from_valid   = 'metadata.csv'




save_train = './token_data_binary_data/train_data_pack'
save_valid = './token_data_binary_data/valid_data_pack'
save_test =  './token_data_binary_data/test_data_pack'


common_voice_train = load_dataset('csv', data_files=from_train)
common_voice_valid = load_dataset('csv', data_files=from_valid)
common_voice_test  = load_dataset('csv', data_files=from_test)
common_voice_train = common_voice_train['train']
common_voice_valid = common_voice_valid['train']
common_voice_test  = common_voice_test['train']
common_voice_train = common_voice_train.cast_column("path", Audio(16000))
common_voice_valid = common_voice_valid.cast_column("path", Audio(16000))
common_voice_test  = common_voice_test.cast_column("path", Audio(16000))

print(common_voice_train) # ['sentence', 'audio', 'speech', 'sampling_rate']
print(common_voice_valid) # ['sentence', 'audio', 'speech', 'sampling_rate']
# print(len(common_voice_test[0]["speech"]))
# print(len(common_voice_test[1]["speech"]))
print(common_voice_test)

def prepare_dataset(batch):
    speech_array = batch["path"]['array']
    # sampling_rate = batch["path"]['sampling_rate']
    sampling_rate = 16000
    
    audio_segment = convert_arr_to_audiosegment(speech_array)
    audio_segment  = audio_segment.set_frame_rate(16000)
    audio_segment_norm = normalize(audio_segment)
    speech_array = convert_audiosegment_to_arr(audio_segment_norm)
    
    batch["input_values"] = processor(speech_array, sampling_rate=sampling_rate).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch

common_voice_train = common_voice_train.map(prepare_dataset, remove_columns=common_voice_train.column_names, num_proc=4)
common_voice_valid = common_voice_valid.map(prepare_dataset, remove_columns=common_voice_valid.column_names, num_proc=4)
common_voice_test = common_voice_test.map(prepare_dataset, remove_columns=common_voice_test.column_names, num_proc=4)

max_input_length_in_sec = 15.0
common_voice_train = common_voice_train.filter(lambda x: x < max_input_length_in_sec * processor.feature_extractor.sampling_rate, input_columns=["input_length"])
common_voice_valid = common_voice_valid.filter(lambda x: x < max_input_length_in_sec * processor.feature_extractor.sampling_rate, input_columns=["input_length"])
common_voice_test = common_voice_test.filter(lambda x: x < max_input_length_in_sec * processor.feature_extractor.sampling_rate, input_columns=["input_length"])


common_voice_train.save_to_disk(save_train)
common_voice_valid.save_to_disk(save_valid)
common_voice_test.save_to_disk(save_test)
