from datasets import load_from_disk, load_dataset, Audio
from transformers import Wav2Vec2Processor
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments
from transformers import Trainer
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import IterableDataset
import math
from torch.utils.data import DataLoader
import faulthandler
faulthandler.enable()
from utiles import convert_arr_to_audiosegment, convert_audiosegment_to_arr
from pydub.effects import normalize

import time
import pickle as pkl
def save_pickle(data, path):
    with open(path, 'wb') as f:
        pkl.dump(data, f)


tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
print('tokenizer', tokenizer)

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=False, return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# common_voice_train = load_from_disk('cleand_train_data_gst_haitian/v2_sp3/train_data_pack')
# common_voice_test = load_from_disk('cleand_train_data_gst_haitian/v2_sp3/valid_data_pack')
# model_save_dir = './saved_models_spk_no_norm_pydub_norm_haitianclean_v2-sp3-lr3e-4/'
# log_save_dir = './log_spk_no_norm_pydub_norm_haitianclean_v2-sp3-lr3e-4'

common_voice_train = load_from_disk('/mnt/nas1/guoshaotong.gst/wer-per/xlsr-wav2vec2/cleand_train_data_gst_haitian-ps2/v1//train_data_pack')
common_voice_test = load_from_disk('/mnt/nas1/guoshaotong.gst/wer-per/xlsr-wav2vec2/cleand_train_data_gst_haitian-ps2/v1//valid_data_pack')

#common_voice_train = load_from_disk('cleand_train_data_gst_haitian-ps2/v1//train_data_pack')
#common_voice_test = load_from_disk('cleand_train_data_gst_haitian-ps2/v1//valid_data_pack')
model_save_dir = './saved_models_spk_no_norm_pydub_norm_haitianps2clean_v2-sp3-lr3e-4-tmp/'
log_save_dir = './log_spk_no_norm_pydub_norm_haitianps2clean_v2-sp3-lr3e-4-tmp'



@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        # print('DataCollatorCTCWithPadding __call__ ==========')
        # print('self.padding', self.padding, 'self.max_length', self.max_length, 'self.max_length_labels', self.max_length_labels)
        
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch
    
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)



def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    print('pred_str', pred_str[0])
    print('label_str', label_str[0])
    # cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": None}

print('len(processor.tokenizer)', len(processor.tokenizer))

print('processor.tokenizer.pad_token_id', processor.tokenizer.pad_token_id)

model = Wav2Vec2ForCTC.from_pretrained(
    "/mnt/nas1/guoshaotong.gst/wer-per/pretrain_moedl",
    attention_dropout=0.1, # transformer中的dropout概率
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05, # 论文中是mask0.065
    layerdrop=0.1,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)
)

model.freeze_feature_extractor()

# print('model===============')
# print(model)

training_args = TrainingArguments(
  output_dir=model_save_dir,
  group_by_length=False,
  per_device_train_batch_size=64,
  per_device_eval_batch_size=4,
  gradient_accumulation_steps=2,
  evaluation_strategy="steps",
  logging_strategy="steps",
  save_strategy="steps",
  num_train_epochs=10,
  gradient_checkpointing=True,
  fp16=True,
  save_steps=500,
  eval_steps=2000,
  logging_steps=10,
  learning_rate=3e-4,
  warmup_steps=1000,
  save_total_limit=20,
  push_to_hub=False,
  report_to="tensorboard",
  logging_dir=log_save_dir
)

# print('training_args==========')
# print(training_args)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=common_voice_train,
    eval_dataset=common_voice_test,
    tokenizer=processor.feature_extractor,
)

print('====start trining====')
trainer.train(resume_from_checkpoint=False)
