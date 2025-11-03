from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
from datasets import load_from_disk
import torch
import os
from utiles import phr, min_edit_distance, beam_search_decode, convert_arr_to_audiosegment, convert_audiosegment_to_arr
import time
from pydub.effects import normalize


start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f'Using device: {device}')






def func(saved_models, test_csv_path, save_f):
    from datasets import load_dataset, Audio # load_metric, 
    common_voice_test = load_dataset('csv', data_files=test_csv_path)
    common_voice_test = common_voice_test['train']

    common_voice_test = common_voice_test.cast_column("path", Audio(16000))

    total_phr = 0
    total_edit_distance = 0
    total_item_num = 0
    total_pho_num = 0
    total_yunxing_time = 0
    total_rft = 0

    ids_to_remove = [570, 571, 572, 573]

    for i in range(len(common_voice_test)):
        try:
            duration = common_voice_test[i]["duration"]

            print(common_voice_test[i]["path"])
            key_name = os.path.basename(common_voice_test[i]["path"]["path"])
            wav_path = common_voice_test[i]["path"]["path"]

            start_time = time.time()

            speech_array = common_voice_test[i]["path"]["array"]
            audio_segment = convert_arr_to_audiosegment(speech_array)
            audio_segment  = audio_segment.set_frame_rate(16000)
            audio_segment_norm = normalize(audio_segment)
            speech_array = convert_audiosegment_to_arr(audio_segment_norm)

            input_dict = processor(speech_array, return_tensors="pt", padding=True)


            print('input_dict["input_values"] : {}\n'.format(input_dict["input_values"].shape))
            save_f.write('input_dict["input_values"] : {}\n'.format(input_dict["input_values"].shape))

            logits = model(input_dict.input_values.to(device)).logits
            # print("logtis : ", logits.shape)
            # print("pred_ids : ", torch.argmax(logits, dim=-1).shape)
            # print("pred_ids : ", torch.argmax(logits, dim=-1)[0].shape)
            # print("pred_ids : ", torch.argmax(logits, dim=-1)[0])
            # exit(0)

            pred_ids = torch.argmax(logits, dim=-1)[0]

            # print('pred_ids', pred_ids.shape)

            true_text = common_voice_test[i]['sentence']
            print('true_text:', true_text)
            save_f.write('true_text: {}\n'.format(true_text))

            pred_text = processor.decode(pred_ids)

            true_ids = None
            with processor.as_target_processor():
                true_ids = processor(true_text).input_ids
                true_ids = list(filter(lambda id_: id_ not in ids_to_remove, true_ids))


            pred_ids = None
            with processor.as_target_processor():
                pred_ids = processor(pred_text).input_ids
                pred_ids = list(filter(lambda id_: id_ not in ids_to_remove, pred_ids))

                pred_phns = ' '.join([tokenizer._convert_id_to_token(_t) for _t in pred_ids])

            print('pred_phns:', pred_phns)
            print('true_ids', true_ids)
            print('pred_ids', pred_ids)
            print('per', phr(pred_ids, true_ids), key_name, wav_path)

            # save_f.write('pred_phns: {}\n'.format(pred_phns))
            save_f.write('true_ids: {}\n'.format(true_ids))
            save_f.write('pred_ids: {}\n'.format(pred_ids))
            save_f.write('per: {} | {} | {} | {} | {}\n'.format(phr(pred_ids, true_ids), key_name, wav_path, true_text, pred_phns))

            total_phr += phr(pred_ids, true_ids)
            total_edit_distance += min_edit_distance(pred_ids, true_ids)
            total_pho_num += len(true_ids)
            total_item_num += 1
            end_time = time.time()

            rtf = (end_time - start_time)/duration

            print('该样本运行时长', end_time - start_time)
            print('rtf', rtf)

            save_f.write('该样本运行时长 : {}\n'.format(end_time - start_time))
            save_f.write('rtf : {}\n'.format(rtf))

            total_rft += rtf

            total_yunxing_time += end_time - start_time
        except:
            continue


    print('total_phr/total_item_num: ', total_phr/total_item_num)
    print('total_edit_distance/total_pho_num: ', total_edit_distance/total_pho_num)
    print('平均运行时间: ', total_yunxing_time/total_item_num)
    print('平均rtf: ', total_rft/total_item_num)


    save_f.write('total_phr/total_item_num : {:4f}\n'.format(total_phr/total_item_num))
    save_f.write('total_edit_distance/total_pho_num : {:4f}\n'.format(total_edit_distance/total_pho_num))
    save_f.write('平均运行时间 : {:4f}\n'.format(total_yunxing_time/total_item_num))
    save_f.write('平均rtf : {:4f}\n\n'.format(total_rft/total_item_num))

    


# meta.csv format
# path,sentence,duration
# pinyin/掉头_随后右转.wav,sil diao4 tou2 sui2 hou4 you4 zhuan3 sil,1.88
# pinyin/请起步_紧跟前车_左转_随后走最右侧车道.wav,sil qing2 qi3 bu4 jin3 gen1 qian2 che1 zuo2 zhuan3 sui2 hou4 zou3 zui4 you4 ce4 che1 dao4 sil,4.67

for test_csv_path in test_csv_paths:
    #test_csv_path = os.path.join(base_csv_dir, test_csv_path)
    test_csv_path = 'metadata.csv'
  
    saved_models = 'saved_models_spk_no_norm_pydub_norm_haitianps2clean_v2-sp3-lr3e-4-tmp/checkpoint-1000/'
   #'saved_models_spk_no_norm_pydub_norm_haitianps2clean_v2-sp3-lr3e-4/checkpoint-5500/'
    save_res_path = os.path.dirname(test_csv_path)  + "/" + os.path.basename(test_csv_path).replace('.csv', "_htps2_ckpt5500.txt")
    os.makedirs(os.path.dirname(save_res_path), exist_ok=True)
    # saved_models = "/mnt/nas1/libowen/project/fine-tune-xlsr-wav2vec2/saved_models_spk_no_norm/checkpoint-10000"
    # save_res_path = test_csv_path.replace('.csv', "_bowen__checkpoint-10000_{}.txt".format(time.time()))
    save_f = open(save_res_path, 'w')

    model = Wav2Vec2ForCTC.from_pretrained(saved_models).to(device)
    tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=False, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    end_time = time.time()
    print(f"模型加载时间: {end_time - start_time} 秒")
    save_f.write(f"模型加载时间: {end_time - start_time} 秒\n")


    print('saved_models==========', saved_models)
    print('test_csv_path=======', test_csv_path)

    save_f.write('saved_models==========  {}\n'.format(saved_models))
    save_f.write('test_csv_path=======   {}\n'.format(test_csv_path))

    func(saved_models, test_csv_path, save_f)
    save_f.close()

