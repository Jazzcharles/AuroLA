from typing import Tuple, Optional, List, Union 
import torch 
from transformers.utils import logging

logger = logging.get_logger(__name__)

# from transformers import AutoProcessor, AutoModel, AutoModelForCausalLM, Qwen2VLForConditionalGeneration, PreTrainedTokenizer
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

from torch import nn 
import torch.distributed as dist
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast

from transformers.processing_utils import Unpack
from transformers.cache_utils import Cache, DynamicCache

# from transformers.utils.generic import TransformersKwargs
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniThinkerCausalLMOutputWithPast

import torch.nn.functional as F
import numpy as np

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp=0.07):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class Qwen2_50OmniRetFinetuneForConditionalGeneration(Qwen2_5OmniThinkerForConditionalGeneration):
    @property
    def temperature(self):
        return 0.05
    
    @property
    def contrastive_alpha(self):
        """
        Alpha parameter for HN-NCE loss.
        Lower values (0.3-0.5) help when false negatives exist in the data.
        Can be overridden by setting the config attribute.
        """
        return getattr(self.config, 'contrastive_alpha', 1.0)
    
    @property
    def contrastive_lambda(self):
        """
        Lambda parameter for HN-NCE loss.
        Weight for extra positive pairs from cluster matching.
        Lower values (0.3-0.7) help when false positives exist.
        Lambda=0 gives standard InfoNCE (diagonal only).
        """
        return getattr(self.config, 'contrastive_lambda', 1.0)
    
    @property
    def contrastive_beta(self):
        """
        Beta parameter for HN-NCE loss.
        Controls the hardness of negative samples.
        Higher values (0.2-0.5) emphasize harder negatives.
        """
        return getattr(self.config, 'contrastive_beta', 0.1)
    
    
    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None,
        video_features: Optional[torch.FloatTensor] = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
            special_audio_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(self.config.audio_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            ).all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id
            special_audio_mask = input_ids == self.config.audio_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Videos features and image tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )

        special_audio_mask = special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        return special_image_mask, special_video_mask, special_audio_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_audio_in_video: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        video_second_per_grid: Optional[torch.LongTensor] = None,

        inference=False,
        positive_mask=None,

        qids=None,
        dids=None,
        ids=None,
        **kwargs,
    ) -> Union[tuple, Qwen2_5OmniThinkerCausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # ============ 分块策略开始 ============
        # Set mini_batch to 8
        mini_batch_size = 4
        input_ids_list = torch.split(input_ids, mini_batch_size)
        attention_mask_list = torch.split(attention_mask, mini_batch_size)
        
        # 准备累积索引用于分块处理多模态特征
        image_nums = 0
        video_nums = 0
        audio_nums = 0
        
        if image_grid_thw is not None:
            cumsum_pixel_values = torch.cumsum(image_grid_thw[:, 1] * image_grid_thw[:, 2], dim=-1)
            zero_tensor = torch.tensor([0], device=cumsum_pixel_values.device)
            cumsum_pixel_values = torch.cat((zero_tensor, cumsum_pixel_values))
        
        if video_grid_thw is not None:
            cumsum_video_values = torch.cumsum(video_grid_thw[:, 0] * video_grid_thw[:, 1] * video_grid_thw[:, 2], dim=-1)
            zero_tensor = torch.tensor([0], device=cumsum_video_values.device)
            cumsum_video_values = torch.cat((zero_tensor, cumsum_video_values))
        
        if audio_feature_lengths is not None:
            cumsum_audio_lengths = torch.cumsum(audio_feature_lengths, dim=0)
            zero_tensor = torch.tensor([0], device=cumsum_audio_lengths.device)
            cumsum_audio_lengths = torch.cat((zero_tensor, cumsum_audio_lengths))
        
        all_hidden_states = []
        
        # 循环处理每个 mini-batch
        for i in range(len(input_ids_list)):
            if inputs_embeds is None:
                # 1. Extract the input embeddings for current batch
                batch_inputs_embeds = self.get_input_embeddings()(input_ids_list[i])
                batch_attention_mask = attention_mask_list[i]
                
            # 2. Merge text, audios, image and video for current batch
            # Process audio features
            if input_features is not None:
                audio_mask = input_ids_list[i] == self.config.audio_token_id
                current_audio_num = torch.sum(torch.any(audio_mask, dim=-1)).cpu().item()
                
                if current_audio_num != 0:
                    # 提取当前 batch 对应的 audio features
                    batch_audio_features = input_features[audio_nums:audio_nums + current_audio_num]
                    
                    if feature_attention_mask is not None:
                        batch_feature_attention_mask = feature_attention_mask[audio_nums:audio_nums + current_audio_num]
                        batch_audio_feature_lengths = torch.sum(batch_feature_attention_mask, dim=1)
                    else:
                        batch_feature_attention_mask = None
                        batch_audio_feature_lengths = None
                    
                    audio_features = self.get_audio_features(
                        batch_audio_features,
                        feature_attention_mask=batch_feature_attention_mask,
                        audio_feature_lengths=batch_audio_feature_lengths,
                    )
                    audio_features = audio_features.to(batch_inputs_embeds.device, batch_inputs_embeds.dtype)
                    _, _, audio_mask_full = self.get_placeholder_mask(input_ids_list[i], inputs_embeds=batch_inputs_embeds)
                    
                    if self.training:
                        batch_inputs_embeds = batch_inputs_embeds.clone()
                    batch_inputs_embeds = batch_inputs_embeds.masked_scatter(audio_mask_full, audio_features)
                    audio_nums += current_audio_num

                    # print('audio_nums', audio_nums)
                    # print('current_audio_num', current_audio_num)
                    # print('audio_features shape', audio_features.shape)
                    # print('audio_mask_full shape', audio_mask_full.shape)
                    # print('batch_inputs_embeds shape', batch_inputs_embeds.shape)
                    # breakpoint()
            else:
                batch_feature_attention_mask = None
                batch_audio_feature_lengths = None
                

            # Process image features
            if pixel_values is not None:
                image_mask = input_ids_list[i] == self.config.image_token_id
                current_image_num = torch.sum(torch.any(image_mask, dim=-1)).cpu().item()
                
                if current_image_num != 0:
                    # 提取当前 batch 对应的 pixel_values
                    batch_pixel_values = pixel_values[cumsum_pixel_values[image_nums]:cumsum_pixel_values[image_nums + current_image_num]]
                    batch_image_grid_thw = image_grid_thw[image_nums:image_nums + current_image_num]
                    
                    image_embeds = self.get_image_features(batch_pixel_values, batch_image_grid_thw)
                    image_embeds = image_embeds.to(batch_inputs_embeds.device, batch_inputs_embeds.dtype)
                    image_mask_full, _, _ = self.get_placeholder_mask(
                        input_ids_list[i], inputs_embeds=batch_inputs_embeds, image_features=image_embeds
                    )
                    
                    if self.training:
                        batch_inputs_embeds = batch_inputs_embeds.clone()
                    batch_inputs_embeds = batch_inputs_embeds.masked_scatter(image_mask_full, image_embeds)
                    image_nums += current_image_num
            
            # Process video features
            if pixel_values_videos is not None:
                video_mask = input_ids_list[i] == self.config.video_token_id
                current_video_num = torch.sum(torch.any(video_mask, dim=-1)).cpu().item()
                
                if current_video_num != 0:
                    # 提取当前 batch 对应的 video pixel_values
                    batch_pixel_values_videos = pixel_values_videos[cumsum_video_values[video_nums]:cumsum_video_values[video_nums + current_video_num]]
                    batch_video_grid_thw = video_grid_thw[video_nums:video_nums + current_video_num]
                    
                    video_embeds = self.get_video_features(batch_pixel_values_videos, batch_video_grid_thw)
                    video_embeds = video_embeds.to(batch_inputs_embeds.device, batch_inputs_embeds.dtype)
                    _, video_mask_full, _ = self.get_placeholder_mask(
                        input_ids_list[i], inputs_embeds=batch_inputs_embeds, video_features=video_embeds
                    )
                    
                    if self.training:
                        batch_inputs_embeds = batch_inputs_embeds.clone()
                    batch_inputs_embeds = batch_inputs_embeds.masked_scatter(video_mask_full, video_embeds)
                    video_nums += current_video_num
            
        
            # Compute position_ids for current batch
            batch_position_ids = None
            if batch_attention_mask is not None and position_ids is None:
                if (
                    cache_position is None
                    or (cache_position is not None and cache_position[0] == 0)
                    or self.rope_deltas is None
                ):
                    delta0 = (1 - batch_attention_mask).sum(dim=-1).unsqueeze(1)
                    
                    # 为当前 batch 计算对应的参数
                    batch_image_grid_thw = None if image_grid_thw is None else (
                        image_grid_thw[image_nums - current_image_num if pixel_values is not None and current_image_num > 0 else 0:image_nums]
                    )
                    batch_video_grid_thw = None if video_grid_thw is None else (
                        video_grid_thw[video_nums - current_video_num if pixel_values_videos is not None and current_video_num > 0 else 0:video_nums]
                    )
                    
                    batch_position_ids, rope_deltas_batch = self.get_rope_index(
                        input_ids_list[i],
                        batch_image_grid_thw,
                        batch_video_grid_thw,
                        batch_attention_mask,
                        use_audio_in_video,
                        batch_audio_feature_lengths,
                        video_second_per_grid,
                    )
                    rope_deltas_batch = rope_deltas_batch - delta0
                else:
                    batch_size_curr, seq_length = input_ids_list[i].shape
                    delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                    batch_position_ids = torch.arange(seq_length, device=input_ids_list[i].device)
                    batch_position_ids = batch_position_ids.view(1, -1).expand(batch_size_curr, -1)
                    batch_position_ids = batch_position_ids.add(delta)
                    batch_position_ids = batch_position_ids.unsqueeze(0).expand(3, -1, -1)
            
            # Forward through model for current batch
            outputs = self.model(
                attention_mask=batch_attention_mask,
                position_ids=batch_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=batch_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )
            
            hidden_states = outputs[0]
            all_hidden_states.append(hidden_states)
        
        # 拼接所有 mini-batch 的结果
        hidden_states = torch.cat(all_hidden_states, dim=0)
        
        # ============ 分块策略结束 ============
        if not inference:
            batch_size = len(hidden_states) // 2
        else:
            batch_size = len(hidden_states)

        embed_index = self.config.emb_token_ids[0]
        embed_indices = torch.argmax((labels == embed_index).int(), dim=1)
        embed_features = hidden_states[torch.arange(len(embed_indices)), embed_indices - 1]

        if inference:
            if ids is not None:
                return embed_features, ids
            elif qids is not None or dids is not None:
                return embed_features, qids, dids
            return embed_features

        embed1, embed2 = embed_features[:batch_size], embed_features[batch_size:]

        if dist.is_initialized():
            embed1_list = [torch.zeros_like(embed1) for _ in range(dist.get_world_size())]
            embed2_list = [torch.zeros_like(embed2) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=embed1_list, tensor=embed1.contiguous())
            dist.all_gather(tensor_list=embed2_list, tensor=embed2.contiguous())

            embed1_list[dist.get_rank()] = embed1
            embed2_list[dist.get_rank()] = embed2
            embed1 = torch.cat(embed1_list, 0)
            embed2 = torch.cat(embed2_list, 0)

            if positive_mask is not None:
                positive_mask_list = [torch.zeros_like(positive_mask) for _ in range(dist.get_world_size())]
                dist.all_gather(tensor_list=positive_mask_list, tensor=positive_mask.contiguous())
                positive_mask_list[dist.get_rank()] = positive_mask
                positive_mask = torch.cat(positive_mask_list, 0)

        embed1 = F.normalize(embed1, dim=-1)
        embed2 = F.normalize(embed2, dim=-1)

        loss = hybrid_nce_loss(
            embed1,
            embed2,
            temperature=self.temperature,
            alpha=self.contrastive_alpha,
            lambd=self.contrastive_lambda,
            beta=self.contrastive_beta,
            bidirectional=False,
            positive_mask=positive_mask,
        )
        return SequenceClassifierOutput(loss=loss)

def hybrid_nce_loss(audio, text, temperature=0.05, alpha=1.0, beta=0.1, lambd=1.0, bidirectional=False, estimator='hard', positive_mask=None):
    """
    audio: [B, D]
    text : [B, D]
    实现 Hybrid NCE loss，支持动态正样本权重（基于 cluster IoU）
    
    参数:
        temperature: 温度参数
        alpha: 处理 false negatives，降低正样本在分母中的权重
        lambd: 全局缩放因子，控制额外正样本对的权重
               对角线权重=1，额外正样本权重=lambda * IoU(cluster_i, cluster_j)
               IoU 是两个样本 cluster 标签的交并比，反映相似度
               lambd=0 时退化为标准 InfoNCE（只有对角线）
        beta: 控制负样本重加权强度
        positive_mask: [B, num_clusters], 每行表示样本属于哪些cluster (0/1)
    """
    device = audio.device
    B = audio.size(0)

    # 构建正样本和负样本掩码
    if positive_mask is not None:    
        # positive_mask: [B, num_clusters], 每行表示该sample属于哪些cluster (0或1)
        # 要求：两个sample必须有至少一个共同的cluster(都为1)，且cluster集合完全一致
        # 策略：计算交集和并集，交集==并集 且 交集>0
        
        mask_i = positive_mask.unsqueeze(1)  # [B, 1, num_clusters]
        mask_j = positive_mask.unsqueeze(0)  # [1, B, num_clusters]
        
        # 交集：两个sample都是1的cluster
        intersection = (mask_i * mask_j)  # [B, B, num_clusters]
        # 并集：至少一个sample是1的cluster
        union = ((mask_i + mask_j) > 0).float()  # [B, B, num_clusters]
        
        # 交集 == 并集 说明在所有非零位置都相同
        same_clusters = (intersection == union).all(dim=2)  # [B, B]
        # 交集非空，说明至少有一个共同cluster
        has_common = (intersection.sum(dim=2) > 0)  # [B, B]
        
        # 两个条件都满足才是正样本对
        pos_mask = (same_clusters & has_common).float()  # [B, B]
        
        # 确保对角线是正样本（但避免重复加导致>1）
        diag_mask = torch.eye(B, dtype=torch.float32, device=device)  # [B, B] 
        pos_mask = torch.clamp(pos_mask + diag_mask, max=1.0)  # 确保值在[0,1]
        
        # 提前计算 IoU 权重（用于调试输出）
        intersection_count = intersection.sum(dim=2)  # [B, B]
        union_count = union.sum(dim=2)  # [B, B]
        iou_weights = intersection_count / (union_count + 1e-8)  # [B, B]
        
        num_total_pos = len(torch.where(pos_mask > 0)[0])
        num_extra_pos = num_total_pos - B  # 除对角线外的额外正样本对数量
        
        # 计算额外正样本对的平均 IoU（用于监控）
        if num_extra_pos > 0:
            extra_pos_indices = torch.where((pos_mask > 0) & (torch.eye(B, device=device) == 0))
            if len(extra_pos_indices[0]) > 0:
                avg_iou = iou_weights[extra_pos_indices].mean().item()
            else:
                avg_iou = 0.0
        else:
            avg_iou = 0.0
        
        # print(f'[HN-NCE] alpha={alpha:.2f}, lambda={lambd:.2f}, beta={beta:.2f}, '
        #       f'total_pos_pairs={num_total_pos}, extra_pos_pairs={num_extra_pos}, avg_IoU={avg_iou:.3f}')
    else:
        # 默认只有对角线是正样本
        pos_mask = torch.eye(B, dtype=torch.float32, device=device)  # [B, B]
    
    # 负样本掩码：非正样本的位置
    neg_mask = 1.0 - pos_mask  # [B, B], 1表示负样本

    # 相似度矩阵 (audio_i, text_j)
    sim = torch.mm(audio, text.t())     # [B, B]
    sim_div = sim / temperature

    # 负样本矩阵 exp(sim(i,j))
    exp_sim = torch.exp(sim_div)        # [B, B]

    # 正样本项：对角线权重=1，额外正样本对权重=IoU(cluster重合度)
    if positive_mask is not None:
        # iou_weights 已在上面计算过
        # 对角线设置为 1（自己和自己的权重始终为1）
        diag_mask_for_pos = torch.eye(B, dtype=torch.float32, device=device)  # [B, B]
        iou_weights = iou_weights * (1 - diag_mask_for_pos) + diag_mask_for_pos
        
        # 使用 lambda 作为全局缩放因子
        # 对角线权重=1，其他正样本权重=lambda * IoU
        pos_weights = diag_mask_for_pos + lambd * (iou_weights - diag_mask_for_pos) * pos_mask
    else:
        # 没有 positive_mask 时，只有对角线
        diag_mask_for_pos = torch.eye(B, dtype=torch.float32, device=device)
        pos_weights = diag_mask_for_pos
    
    # pos_sum = Σ exp(sim) * weight，对角线权重=1，其他正样本权重=lambda*IoU
    pos_sum = (exp_sim * pos_weights).sum(dim=1)  # [B]

    # -------------------------------
    #    方向 1: audio_i -> text_j
    # -------------------------------

    # e^{β x_i^T t_j / τ}, 只计算负样本
    weight_logits_a2t = torch.exp( (beta / temperature) * sim )  # [B, B]
    weight_logits_a2t = weight_logits_a2t * neg_mask             # 只保留负样本

    if estimator == 'hard':
        # 计算负样本的数量（每行）
        num_negs = neg_mask.sum(dim=1, keepdim=True)  # [B, 1]
        
        # w_{i->t} 按公式：num_negs*exp(...) / sum_k exp(...)
        denom_a2t = weight_logits_a2t.sum(dim=1, keepdim=True) + 1e-8  # [B,1], 防止除0
        w_a2t = num_negs * weight_logits_a2t / denom_a2t               # [B,B]

        # 求分母中的 ∑_{j:neg} e^{sim/τ} * w
        weighted_neg_a2t = (exp_sim * w_a2t).sum(dim=1)           # [B]

    else:
        weighted_neg_a2t = (exp_sim * neg_mask).sum(dim=1)        # [B]

    # 分母： α*正样本和 + weighted negatives
    denom_a2t_total = alpha * pos_sum + weighted_neg_a2t          # [B]

    loss_a2t = - torch.log(pos_sum / (denom_a2t_total + 1e-8)).mean()

    # -------------------------------
    #    方向 2: text_j -> audio_i
    #    完全对称
    # -------------------------------

    # e^{β x_j^T t_i / τ} = transpose
    weight_logits_t2a = torch.exp( (beta/temperature) * sim.t() )   # [B,B]
    weight_logits_t2a = weight_logits_t2a * neg_mask.t()            # 负样本掩码转置

    if estimator == 'hard':
        num_negs_t = neg_mask.t().sum(dim=1, keepdim=True)  # [B, 1]
        denom_t2a = weight_logits_t2a.sum(dim=1, keepdim=True) + 1e-8  # [B,1]
        w_t2a = num_negs_t * weight_logits_t2a / denom_t2a             # [B,B]

        # weighted negatives
        weighted_neg_t2a = (exp_sim.t() * w_t2a).sum(dim=1)            # [B]
    else:
        weighted_neg_t2a = (exp_sim.t() * neg_mask.t()).sum(dim=1)    # [B]

    denom_t2a_total = alpha * pos_sum + weighted_neg_t2a

    loss_t2a = - torch.log(pos_sum / (denom_t2a_total + 1e-8)).mean()


    # -------------------------------
    #      最终对称 loss
    # -------------------------------
    if bidirectional:
        loss = (loss_a2t + loss_t2a) / 2
    else:
        loss = loss_a2t

    return loss
