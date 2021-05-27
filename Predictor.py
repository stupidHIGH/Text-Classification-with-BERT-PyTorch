import numpy as np
import torch.nn as nn
from pytorch_transformers import BertTokenizer
from tqdm import tqdm, trange
import torch,argparse

from BertModules import BertClassifier
from Constants import *
from DataModules import SequenceDataset
from Utils import seed_everything

SEP_TOKEN = '[SEP]'
CLS_TOKEN = '[CLS]'
MAX_SEQ_LENGTH = 512
DEVICE="cuda"

parser = argparse.ArgumentParser(description='extract CNN pooling features from images')
parser.add_argument('--input_path',default='')
parser.add_argument('--file_name',default='')
parser.add_argument('--output_path',default='')
parser.add_argument('--sentiment-predictor-ckpt-dir',
                    default='/apdcephfs/share_916081/zltian/send_to_cluster_standard/rag/wuzhenghao/openvidial/repo/OpenViDial/pretrain/sentiment_predictor/models/AllDataCkpts/best_ckpt.pt')
parser.add_argument('--sentiment-predictor-dir',
                    default='/apdcephfs/share_916081/zltian/send_to_cluster_standard/rag/wenzhihua/LM/bert-base-cased/pytorch_version',
                    help='data directory')
args = parser.parse_args()

seed_everything()
bert_tokenizer = BertTokenizer.from_pretrained(args.sentiment_predictor_dir, do_lower_case=False)
bert = torch.load(args.sentiment_predictor_ckpt_dir)
bert.eval()
lines = open(args.input_path+"/"+args.file_name,encoding="utf8").readlines()

input_ids = []
segment_ids = []
input_mask = []
for words in lines:
    words = words.strip().replace(" &apos;", "'")
    tokens = bert_tokenizer.tokenize(words)
    tokens = [CLS_TOKEN] + tokens + [SEP_TOKEN]
    temp_input_ids = bert_tokenizer.convert_tokens_to_ids(tokens)

    # Segment ID for a single sequence in case of classification is 0.
    temp_segment_ids = [0] * len(temp_input_ids)

    # Input mask where each valid token has mask = 1 and padding has mask = 0
    temp_input_mask = [1] * len(temp_input_ids)

    # padding_length is calculated to reach max_seq_length
    padding_length = MAX_SEQ_LENGTH - len(temp_input_ids)
    temp_input_ids = temp_input_ids + [0] * padding_length
    temp_input_mask = temp_input_mask + [0] * padding_length
    temp_segment_ids = temp_segment_ids + [0] * padding_length

    input_ids += torch.tensor([temp_input_ids], dtype=torch.long, device=DEVICE)
    segment_ids += torch.tensor([temp_segment_ids], dtype=torch.long, device=DEVICE)
    input_mask += torch.tensor([temp_input_mask], device=DEVICE, dtype=torch.long)
    print("input_ids",input_ids.shape)
    
input_ids = torch.stack(input_ids)
segment_ids = torch.stack(segment_ids)
input_masks = torch.stack(input_mask)

tso_label_score_list = []
for input_id, segment_id, input_mask in zip(input_ids, segment_ids, input_masks):
    with torch.no_grad():
        tso_label_score_list.append(
            bert(input_ids=input_id.unsqueeze(0), token_type_ids=segment_id.unsqueeze(0),
                      attention_mask=input_mask.unsqueeze(0)))

with open(args.output_path+"/"+args.file_name,"w",encoding="utf8") as f:
    for score in tso_label_score_list:
        f.write(str(score.argmax(-1))+"\n")

tso_label_score_list = [each.numpy() for each in tso_label_score_list]
tso_label_score_list = np.stack(tso_label_score_list)
np.save(tso_label_score_list,args.output+"/"+args.file_name,allow_pickle=True)
