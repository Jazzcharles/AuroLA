
import os
import json
import random
import torch
import numpy as np
import torch.nn.functional as F
from toolz.sandbox import unzip
from torch.utils.data import Dataset
from utils.logger import LOGGER
# from .vision_mapper import VisionMapper
# from .audio_mapper import AudioMapper

from torch.utils.data import ConcatDataset
from .base.base_collactor import BaseDataCollator

from qwen_omni_utils import process_mm_info
from typing import Dict, Sequence
from collections import defaultdict
from torch.utils.data import get_worker_info

def construct_rerank_messages_single_candidate(query_dict, cand_dict, type='pos'):
    """Construct rerank message for single candidate evaluation."""
    message = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "I will provide you with a query and a candidate. Please evaluate whether the candidate\
                    matches the query. If it does, respond with 'Yes'; if it doesn't, respond with 'No'."}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Yes" if type == 'pos' else "No"}
            ]
        }
    ]
    query = [{'type': 'text', 'text': 'Query:'}]
    cand = [{'type': 'text', 'text': 'Candidate:'}]

    if 'audio' in query_dict:
        query.append({'type': 'audio', 'audio': query_dict['audio']})
    if 'txt' in query_dict:
        query.append({'type': 'text', 'text': query_dict['txt']})
    if 'audio' in cand_dict:
        cand.append({'type': 'audio', 'audio': cand_dict['audio']})
    if 'txt' in cand_dict:
        cand.append({'type': 'text', 'text': cand_dict['txt']})

    for item in query:
        message[0]['content'].append(item)

    for item in cand:
        message[0]['content'].append(item)

    return message

class AudioVerse_rerank(Dataset):
    def __init__(
        self,
        audio_folder,
        metadata_path,
        tokenizer,
    ):
        # self.vision_mapper = VisionMapper(d_cfg, args) if 'vision' in d_cfg else None
        self.audio_folder = audio_folder
        self.metadata_path = metadata_path
        self.tokenizer = tokenizer
        self.annos = json.load(open(metadata_path))
            
    def __len__(self):
        return len(self.annos)


    def get_caption(self, anno, rand_indices=0):
        ### load text ###
        raw_captions = None
        
        if 'caption_long' in anno and 'caption_short' in anno and 'caption_tag' in anno:
            raw_captions = [anno['caption_long'], anno['caption_short']]
            if len(anno['caption_tag']) > 0: raw_captions.append(anno['caption_tag'])

            if rand_indices == -1:
                rand_indices = np.random.randint(0, len(raw_captions))
                
            raw_captions = raw_captions[rand_indices]
        else:
            raw_captions = anno['caption']
        
        ### debug, we just take the first 1 of 5 caption for now !!!! ###
        ### construct text ###
        if isinstance(raw_captions, list):
            raw_captions = random.choice(raw_captions)
            
        return raw_captions

    def get_instance(self, anno, rand_indices=0):
        raw_captions = self.get_caption(anno, rand_indices)
        
        ### load audio path ###
        for key in ['video_id','image_id','image','id']:
            if key in anno: 
                id_ = anno[key]
                break
        
        audio_path = os.path.join(self.audio_folder, id_)
        if '.' not in audio_path:
            audio_path += '.wav'
        # 尝试几种后缀
        for ext in ['wav', 'flac', 'mp3', 'mkv']:
            fpath = audio_path.rsplit('.', 1)[0] + '.' + ext
            if os.path.exists(fpath):
                audio_path = fpath
                break
        
        assert os.path.exists(audio_path)
        
        # Process text caption
        raw_captions = self.tokenizer(raw_captions, truncation=True, max_length=480, padding=False, return_tensors=None, add_special_tokens=False)
        raw_captions = self.tokenizer.decode(raw_captions['input_ids'])

        # Prepare data dict for rerank format
        data_dict = {}
        if audio_path:
            data_dict['audio'] = audio_path
        if raw_captions:
            data_dict['txt'] = raw_captions

        return data_dict

    
    def __getitem__(self, i):
        try:
            ### Initialize worker-specific random seed to prevent deadlocks in multi-worker training
            worker_info = get_worker_info()
            if worker_info is not None:
                # We're in a worker process, set unique random seed
                worker_id = worker_info.id
                worker_seed = (torch.initial_seed() + worker_id) % (2**32)
                random.seed(worker_seed)
                np.random.seed(worker_seed)

            anno = self.annos[i]

            # Get query (current sample's audio and text)
            sample_dict = self.get_instance(anno, rand_indices=-1)
            # Determine query type first (audio as query or text as query)
            rand_type = random.random() < 0.5
            # Get negative candidates using stage1 hard negatives based on rand_type
            num_candidates = 1
            
            # get pre-computed hard-negatives from metafile #
            if 'stage1_audio_hard_negative' in anno and 'stage1_text_hard_negative' in anno:
                audio_hard_negative_ids = anno['stage1_audio_hard_negative']
                text_hard_negative_ids = anno['stage1_text_hard_negative']
                
                # Select hard negative, use either audio or text hard negative is ok 
                if rand_type == 0:
                    hard_negative_ids = text_hard_negative_ids.copy()
                else:
                    hard_negative_ids = audio_hard_negative_ids.copy()
                
                # Filter out current sample index if it exists
                hard_negative_ids = [idx for idx in hard_negative_ids if idx != i]
                
                if len(hard_negative_ids) == 0:
                    # Fallback: use all available indices except current
                    available_indices = [idx for idx in range(len(self.annos)) if idx != i]
                    hard_negative_ids = random.sample(available_indices, min(len(available_indices), num_candidates))
                elif len(hard_negative_ids) < num_candidates:
                    # Sample from hard negatives and supplement if needed
                    selected_neg_ids = random.sample(hard_negative_ids, len(hard_negative_ids))
                    available_indices = [idx for idx in range(len(self.annos)) 
                                        if idx != i and idx not in selected_neg_ids]
                    if len(available_indices) > 0:
                        additional_samples = random.sample(available_indices, 
                                                            min(len(available_indices), num_candidates - len(selected_neg_ids)))
                        selected_neg_ids.extend(additional_samples)
                    hard_negative_ids = selected_neg_ids
                else:
                    hard_negative_ids = random.sample(hard_negative_ids, num_candidates)
            else:
                # Fallback: random sampling
                available_indices = [idx for idx in range(len(self.annos)) if idx != i]
                hard_negative_ids = random.sample(available_indices, min(len(available_indices), num_candidates))

            # Get negative candidate dicts
            neg_cand_dicts = []
            for neg_idx in hard_negative_ids:
                neg_anno = self.annos[neg_idx]
                neg_dict = self.get_instance(neg_anno, random.choice([k for k in range(2)]))
                neg_cand_dicts.append(neg_dict)

            neg_cand_dict = neg_cand_dicts[0] if len(neg_cand_dicts) > 0 else sample_dict.copy()

            if rand_type == 0:
                ### use audio as query ###
                query_dict = sample_dict.copy()
                query_dict['txt'] = 'Find a caption describing the sound events in the given audio.'
                pos_cand_dict = {'txt': sample_dict['txt']}
                neg_cand_dict = {'txt': neg_cand_dict['txt']}
            else:
                ### use text as query ###
                query_dict = {
                    'txt': f'Find an audio containing the sound events in the following caption: {sample_dict["txt"]}'
                }
                pos_cand_dict = {'audio': sample_dict['audio']}
                neg_cand_dict = {'audio': neg_cand_dict['audio']}
                
            # Construct rerank messages
            rerank_pos_message = construct_rerank_messages_single_candidate(query_dict, pos_cand_dict, type='pos')
            rerank_neg_message = construct_rerank_messages_single_candidate(query_dict, neg_cand_dict, type='neg')
            return rerank_pos_message, rerank_neg_message

        except:
            return self.__getitem__(random.randint(0, len(self.annos) - 1))