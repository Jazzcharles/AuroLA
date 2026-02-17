
import os
import json
import random
import torch
import numpy as np
from torch.utils.data import Dataset

def construct_messages(text=None, audio=None):
    # sys_prompt = {
    #     "role": "system",
    #     "content": [
    #         {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
    #     ],
    # }
    if audio is not None and text is not None:
        message = [
            #sys_prompt,
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": text},
                    {"type": "text", "text": f"\nSummarize above audio and sentence in one word: "}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"<emb>."}
                ]
            },
        ]
    elif audio is None:
        message = [
            #sys_prompt,
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{text}\nSummarize above sentence in one word: "}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"<emb>."}
                ]
            },
        ]
    else:
        message = [
            #sys_prompt,
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": f"\nSummarize above audio in one word: "}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"<emb>."}
                ]
            },
        ]
    return message

class AudioVerse(Dataset):
    def __init__(self, audio_folder, metadata_path, tokenizer,
                 use_tag_clustering=False, tag_cluster_path=None, use_positive_mask=False,
                 ):

        self.audio_folder = audio_folder
        self.metadata_path = metadata_path
        self.tokenizer = tokenizer
        self.annos = json.load(open(metadata_path))
        self.idx = list(range(len(self.annos)))
        self.use_tag_clustering = use_tag_clustering
        self.use_positive_mask = use_positive_mask

        if self.use_tag_clustering:
            assert tag_cluster_path is not None, 'tag_cluster_path is required when use_tag_clustering is True'
            self.tag_cluster_mapping = json.load(open(tag_cluster_path))
            self.num_clusters = max([x['cluster_index'] if x['cluster_index'] is not None else 0 for x in self.tag_cluster_mapping.values()]) + 1 ### cluster_index starts from 1
            print('Total {} clusters'.format(self.num_clusters))
            
    def __len__(self):
        return len(self.annos)


    def get_caption(self, anno, rand_indices=0):
        ### load text ###
        raw_captions = None
        
        ### optionally, add original caption here ###
        if 'caption_long' in anno and 'caption_short' in anno and 'caption_tag' in anno:
            raw_captions = [anno['caption_long'], anno['caption_short']]
            if len(anno['caption_tag']) > 0: raw_captions.append(anno['caption_tag'])

            if rand_indices == -1:
                rand_indices = np.random.randint(0, len(raw_captions))
                
            raw_captions = raw_captions[rand_indices]
        else:
            raw_captions = anno['caption']
        
        ### construct text ###
        if isinstance(raw_captions, list):
            raw_captions = random.choice(raw_captions)
            
        ### process mask if needed ###
        if self.use_tag_clustering and self.use_positive_mask:
            if rand_indices > 1 and 'caption_tag' in anno:
                tag_mask = self.build_tag_mask(anno['caption_tag'])
            else:
                ### selected long and short caption, we do not need to set tag_mask ###
                tag_mask = torch.zeros(self.num_clusters)
            return raw_captions, tag_mask

        return raw_captions

    def build_tag_mask(self, caption):
        tag_mask = torch.zeros(self.num_clusters)
    
        all_tags = caption.split(',')
        all_tags = [tag.lstrip(' ').rstrip(' ') for tag in all_tags]
        for tag in all_tags:
            if tag not in self.tag_cluster_mapping:
                continue
            elif self.tag_cluster_mapping[tag]['cluster_index'] is None:
                continue
            tag_cluster = self.tag_cluster_mapping[tag]['cluster_index']
            tag_mask[tag_cluster] = 1

        return tag_mask

    def get_instance(self, anno, rand_indices=0):
        # raw_captions = self.get_caption(anno, rand_indices)
        if self.use_tag_clustering and self.use_positive_mask:
            raw_captions, tag_mask = self.get_caption(anno, rand_indices)
        else:
            raw_captions = self.get_caption(anno, rand_indices)
        
        ### load audio path ###
        for key in ['video_id','image_id','image','id']:
            if key in anno: 
                id_ = anno[key]
                break
        
        audio_path = os.path.join(self.audio_folder, id_)
        if '.' not in audio_path:
            audio_path += '.wav'
        
        # try different audio extensions
        for ext in ['wav', 'flac', 'mp3', 'mkv']:
            fpath = audio_path.rsplit('.', 1)[0] + '.' + ext
            if os.path.exists(fpath):
                audio_path = fpath
                break
        
        assert os.path.exists(audio_path)
        audio_message = construct_messages(audio=audio_path)
        text_message = self.get_text_message(raw_captions)

        if self.use_tag_clustering and self.use_positive_mask:
            return audio_message, text_message, tag_mask

        return audio_message, text_message

    def get_text_message(self, raw_captions):   
        raw_captions = self.tokenizer(raw_captions, truncation=True, max_length=480, padding=False, return_tensors=None, add_special_tokens=False)
        raw_captions = self.tokenizer.decode(raw_captions['input_ids'])
        text_message = construct_messages(text=raw_captions)
        return text_message

    def __getitem__(self, i):
        try:
            anno = self.annos[i]

            if self.use_tag_clustering and self.use_positive_mask:
                audio_message, text_message, tag_mask = self.get_instance(anno, rand_indices=-1)   
                return audio_message, text_message, tag_mask         
            else:
                audio_message, text_message = self.get_instance(anno, rand_indices=-1)
            
            return audio_message, text_message

        except:
            return self.__getitem__(random.randint(0, len(self.annos) - 1))
