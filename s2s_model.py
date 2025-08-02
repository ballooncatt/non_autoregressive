import torch
import os
import logging
import torch.nn.functional as F
from slam_llm.models.slam_model import (
    slam_model,
    setup_tokenizer,
    setup_encoder,
    setup_encoder_projector,
    setup_llm,
)
from slam_llm.utils.train_utils import print_model_size
from typing import List, Optional, Generator
from slam_llm.utils.metric import compute_accuracy
from transformers import T5ForConditionalGeneration
from tqdm import tqdm
from utils.tts_adapter_utils import setup_tts_adapter
from utils.codec_utils import setup_codec
from utils.trick_utils import partial_freeze_weights, train_embedding_layer_only, train_embedding_layer
from utils.snac_utils import get_snac, generate_audio_data, simple_shift
from utils.snac_utils import layershift as layer_shift
from utils.projector_utils import setup_group_decode_adapter
from slam_llm.utils.config_utils import generate_peft_config
from peft import get_peft_model

logger = logging.getLogger(__name__)


def model_factory(train_config, model_config, **kwargs):
    # return necessary components for training
    tokenizer = setup_tokenizer(train_config, model_config, **kwargs)

    whisper_model = None
    if train_config.task_type == "s2s" or train_config.task_type == "asr":
        if not model_config.whisper_decode:
            encoder = setup_encoder(train_config, model_config, **kwargs)
        else:
            whisper_model = setup_encoder(train_config, model_config, **kwargs)
            encoder = whisper_model.encoder
    elif train_config.task_type == "tts":
        encoder = None
    else:
        raise NotImplementedError

    # llm
    llm = setup_llm(train_config, model_config, **kwargs)

    # projector
    if encoder is not None:
        encoder_projector = setup_encoder_projector(
            train_config, model_config, **kwargs
        )
        if train_config.freeze_encoder_projector:
            for name, param in encoder_projector.named_parameters():
                param.requires_grad = False
            encoder_projector.eval()
    else:
        encoder_projector = None

    codec_decoder = None
    if model_config.codec_decode:
        codec_decoder = setup_codec(train_config, model_config, **kwargs)

    tts_adapter = None
    if model_config.tts_adapter:
        adapter_config = model_config.tts_adapter_config
        tts_adapter = setup_tts_adapter(adapter_config, model_config, **kwargs)

    group_decode_adapter = None
    if model_config.group_decode:
        group_decode_adapter = setup_group_decode_adapter(model_config, train_config, **kwargs)
        if train_config.freeze_group_decode_adapter:
            for name, param in group_decode_adapter.named_parameters():
                param.requires_grad = False
            group_decode_adapter.eval()

    model = slam_model_s2s(
        encoder,
        llm,
        encoder_projector,
        tokenizer,
        tts_adapter,
        codec_decoder,
        group_decode_adapter,
        whisper_model,
        train_config,
        model_config,
        **kwargs,
    )

    ckpt_path = kwargs.get(
        "ckpt_path", None
    )  # FIX(MZY): load model ckpt(mainly projector, related to model_checkpointing/checkpoint_handler.py: save_model_checkpoint_peft)
    if ckpt_path is not None:
        logger.info("loading other parts from: {}\n".format(ckpt_path))
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt_dict, strict=False)

    if train_config.train_audio_embed_only:
        partial_freeze_weights(model, model_config.vocab_config.padded_text_vocabsize, model_config.vocab_config.total_vocabsize)

    if train_config.train_embed_only:
        train_embedding_layer_only(model)

    if train_config.train_embed:
        train_embedding_layer(model)

    print_model_size(
        model,
        train_config,
        (
            int(os.environ["RANK"])
            if train_config.enable_fsdp or train_config.enable_ddp
            else 0
        ),
    )
    return model, tokenizer


class slam_model_s2s(slam_model):
    def __init__(
        self,
        encoder,
        llm,
        encoder_projector,
        tokenizer,
        tts_adapter,
        codec_decoder,
        group_decode_adapter,
        whisper_model,
        train_config,
        model_config,
        **kwargs,
    ):
        super().__init__(
            encoder,
            llm,
            encoder_projector,
            tokenizer,
            train_config,
            model_config,
            **kwargs,
        )

        # resize llm embedding layer
        self.original_vocabsize = self.llm.lm_head.weight.size(0)
        if self.model_config.vocab_config.total_vocabsize != self.original_vocabsize:
            self.llm.resize_token_embeddings(self.model_config.vocab_config.total_vocabsize)

            if int(os.environ.get("RANK", "0")) == 0:
                logger.info("Resize llm embedding layer's vocab size to {}\n".format(self.model_config.vocab_config.total_vocabsize))

        self.codec_decoder = codec_decoder
        self.whisper_model = whisper_model
        self.tts_adapter = tts_adapter
        self.code_layer = self.model_config.vocab_config.code_layer
        self.group_decode_adapter = group_decode_adapter


    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs,
                ):
        audio_mel = kwargs.get("audio_mel", None)
        audio_embedding = kwargs.get("audio_embedding", None)
        audio_mel_post_mask = kwargs.get("audio_mel_post_mask", None) # 2x downsample for whisper

        audio = kwargs.get("audio", None)
        audio_mask = kwargs.get("audio_mask", None)

        modality_mask = kwargs.get("modality_mask", None)

        encoder_outs = None
        if audio_mel is not None or audio is not None:
            if audio_embedding is None:
                if self.train_config.freeze_encoder: # freeze encoder
                    self.encoder.eval()

                if self.model_config.encoder_name == "whisper":
                    encoder_outs = self.encoder.extract_variable_length_features(audio_mel.permute(0, 2, 1)) # bs*seq*dim
                if self.model_config.encoder_name == "wavlm":
                    encoder_outs = self.encoder.extract_features(audio, 1 - audio_mask) #(FIX:MZY): 1-audio_mask is needed for wavlm as the padding mask
                if self.model_config.encoder_name == "hubert":
                    results = self.encoder(source = audio, padding_mask = 1-audio_mask)
                    if self.model_config.encoder_type == "pretrain":
                        encoder_outs, audio_mel_post_mask = results["x"], results["padding_mask"]
                    if self.model_config.encoder_type == "finetune":
                        encoder_outs, audio_mel_post_mask = results["encoder_out"], results["padding_mask"]
                        encoder_outs = encoder_outs.transpose(0, 1)
                if self.encoder is None:
                    encoder_outs = audio_mel if audio_mel is not None else audio
            else:
                encoder_outs = audio_embedding

            if self.model_config.encoder_projector == "q-former":
                encoder_outs = self.encoder_projector(encoder_outs, audio_mel_post_mask)
            if self.model_config.encoder_projector == "linear":
                encoder_outs = self.encoder_projector(encoder_outs)
            if self.model_config.encoder_projector == "cov1d-linear": 
                encoder_outs = self.encoder_projector(encoder_outs)

        if input_ids is not None:
            input_ids[input_ids == -1] = 0  # [btz, code_layer + 1, seq_length]

            if isinstance(self.llm, T5ForConditionalGeneration):
                inputs_embeds = self.llm.shared(input_ids)
            else:
                if hasattr(self.llm.model, "embed_tokens"):
                    inputs_embeds = self.llm.model.embed_tokens(input_ids)  # [btz, code_layer + 1, seq_length, emb_dim]
                elif hasattr(self.llm.model.model, "embed_tokens"):
                    inputs_embeds = self.llm.model.model.embed_tokens(input_ids)
                else:
                    inputs_embeds = self.llm.model.model.model.embed_tokens(input_ids)

        if modality_mask is not None and encoder_outs is not None:
            modality_mask = modality_mask.unsqueeze(1).repeat(1, self.code_layer, 1)  # [btz, code_layer, seq_length]
            modality_mask_start_indices = (modality_mask == True).float().argmax(dim=2)
            modality_lengths = torch.clamp(modality_mask.sum(dim=2), max=encoder_outs.shape[1]).tolist()

            encoder_outs_pad = torch.zeros_like(inputs_embeds)
            for i in range(encoder_outs.shape[0]):
                for j in range(self.code_layer):
                    start_idx = modality_mask_start_indices[i, j].item()
                    length = modality_lengths[i][j]
                    encoder_outs_pad[i, j, start_idx:start_idx+length] = encoder_outs[i, :length]
            
            inputs_embeds[:, :self.code_layer, :, :] = encoder_outs_pad[:, :self.code_layer, :, :] + inputs_embeds[:, :self.code_layer, :, :] * (~modality_mask[:, :, :, None])
        
        inputs_embeds = torch.mean(inputs_embeds, dim=1)  # [btz, seq_length, emb_dim], average over the code layers

        if kwargs.get("inference_mode", False):
            return inputs_embeds, attention_mask

        text_labels = labels[:,self.code_layer] if labels is not None else None
        audio_labels = labels[:, :self.code_layer] if labels is not None else None
        model_outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=text_labels)    # here we use the text token layer as the target label

        # parrallel generation
        # TODO: add tts adapter forward
        x_ori = model_outputs.logits
        text_vocab_size = self.model_config.vocab_config.padded_text_vocabsize
        audio_vocab_size = self.model_config.vocab_config.padded_audio_vocabsize
        xt = x_ori[..., :text_vocab_size]
        xa = []

        if self.group_decode_adapter is not None:
            x_audio_ori = x_ori[..., text_vocab_size:]
            x_audio = self.group_decode_adapter(x_audio_ori)
            for i in range(self.code_layer):
                xa.append(x_audio[..., i * audio_vocab_size : (i + 1) * audio_vocab_size])
        else:
            for i in range(self.code_layer):
                xa.append(x_ori[..., text_vocab_size + audio_vocab_size * i : text_vocab_size + audio_vocab_size * (i + 1)])

        loss_recorder = []
        total_loss, loss_recorder = self.compute_parallel_loss(xt, text_labels, xa, audio_labels)
        model_outputs.loss = total_loss

        text_acc = -1
        audio_acc = [-1 for _ in range(self.code_layer)]
        if self.metric:
            with torch.no_grad():
                preds = torch.argmax(xt, -1)
                text_acc = compute_accuracy(preds.detach()[:, :-1], text_labels.detach()[:, 1:], ignore_label=-100)

                if self.train_config.task_type != "asr":
                    preds_audio = [torch.argmax(xa[i], -1) for i in range(self.code_layer)]
                    audio_acc = [compute_accuracy(preds_audio[i].detach()[:, :-1], audio_labels[:, i, 1:], ignore_label=-100) for i in range(self.code_layer)]
                else:
                    audio_acc = [-1 for _ in range(self.code_layer)]

        # metrics = {"text_acc": text_acc, "audio_acc": audio_acc, "layer_loss": loss_recorder}
        return model_outputs, text_acc, audio_acc, loss_recorder



    def compute_parallel_loss(self, xt, text_labels, xa, audio_labels):
        """
        Compute the parallel loss for text and audio layers.
        """
        text_vocab_size = self.model_config.vocab_config.padded_text_vocabsize
        audio_vocab_size = self.model_config.vocab_config.padded_audio_vocabsize
        layer_loss = [0 for _ in range(self.code_layer+1) ]
        
        if text_labels is not None:
            text_loss = F.cross_entropy(xt[:, :-1, :].reshape(-1, text_vocab_size), text_labels[:, 1:].reshape(-1), ignore_index=-100)
            layer_loss[self.code_layer] = text_loss
        else:
            text_loss = 0

        total_audio_loss = 0
        single_audio_loss = 0
        for i in range(self.code_layer):
            if audio_labels[:,i] is not None and self.train_config.task_type != "asr":
                single_audio_loss = F.cross_entropy(xa[i][:, :-1, :].reshape(-1, audio_vocab_size), audio_labels[:, i, 1:].reshape(-1), ignore_index=-100)
                layer_loss[i] = single_audio_loss
                total_audio_loss += single_audio_loss

        total_loss = (text_loss + total_audio_loss) / (self.code_layer+1)
        return total_loss, layer_loss


    @torch.no_grad()
    def generate_non(self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs,
            ):
        kwargs["inference_mode"] = True
        inputs_embeds, attention_mask = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        gen_length = kwargs.get("max_new_tokens", 360)
        # 直接调用
        generated_ids = self.llm.generate_with_embeds(
            inputs_embeds=inputs_embeds,
            gen_length=gen_length,
            **kwargs
        )
        return generated_ids

    @torch.no_grad()
    def generate_withx(self, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
                cfg_scale=0., remasking='low_confidence', mask_id=126336):
        '''
        Args:
            prompt: A tensor of shape (1, l).
            steps: Sampling steps, less than or equal to gen_length.
            gen_length: Generated answer length.
            block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
            temperature: Categorical distribution sampling temperature.
            cfg_scale: Unsupervised classifier-free guidance scale.
            remasking: Remasking strategy. 'low_confidence' or 'random'.
            mask_id: The toke id of [MASK] is 126336.
        '''
        x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(prompt.device)
        x[:, :prompt.shape[1]] = prompt.clone()

        prompt_index = (x != mask_id)

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks

        for num_block in range(num_blocks):
            block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
            num_transfer_tokens = self.get_num_transfer_tokens(block_mask_index, steps)
            for i in range(steps):
                mask_index = (x == mask_id)
                if cfg_scale > 0.:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(x).logits

                logits_with_noise = self.add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

                if remasking == 'low_confidence':
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]

        return x

    @staticmethod
    def add_gumbel_noise(logits, temperature):
        '''
        The Gumbel max is a method for sampling categorical distributions.
        According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
        Thus, we use float64.
        '''
        if temperature == 0:
            # When temperature=0, we can directly return the original logits. 
            # without any noise or transformation
            return logits
        
        # use float64 for more stable computation
        logits = logits.to(torch.float64)
        noise = torch.rand_like(logits, dtype=torch.float64)
        gumbel_noise = (- torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise


    @torch.no_grad()
    def generate_ar(self,
                    input_ids: torch.LongTensor = None,
                    attention_mask: Optional[torch.Tensor] = None,
                    position_ids: Optional[torch.LongTensor] = None,
                    past_key_values: Optional[List[torch.FloatTensor]] = None,
                    inputs_embeds: Optional[torch.FloatTensor] = None,
                    labels: Optional[torch.LongTensor] = None,
                    use_cache: Optional[bool] = None,
                    output_attentions: Optional[bool] = None,
                    output_hidden_states: Optional[bool] = None,
                    return_dict: Optional[bool] = None,
                    **kwargs,
                    ):
        kwargs["inference_mode"] = True

        inputs_embeds, attention_mask = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        
        # print(inputs_embeds.shape)
        pad_t = self.model_config.vocab_config.pad_t
        pad_a = self.model_config.vocab_config.pad_a
        eot = self.model_config.vocab_config.eot
        eoa = self.model_config.vocab_config.eoa



        max_new_tokens = kwargs.get("max_new_tokens", 360)

        total_length = inputs_embeds.shape[1] + max_new_tokens 
        padid = torch.tensor([[[pad_a], [pad_a], [pad_a], [pad_t]]])
        masked_embed = self.model.embed_tokens(torch.tensor([padid]).to(inputs_embeds.device)) # shape (1, d)
        x_embeds = masked_embed.repeat(1, total_length, 1).to(inputs_embeds.device) # shape (1, l + gen_length + suffix_len, d)
        x_embeds[:, :inputs_embeds.shape[1]] = inputs_embeds.clone()


        
        num_iter = kwargs.get("nar_steps", 1)  # 非自回归迭代轮数
        device = input_ids.device

        text_vocab_size = self.model_config.vocab_config.padded_text_vocabsize
        audio_vocab_size = self.model_config.vocab_config.padded_audio_vocabsize


        # 非自回归 Mask-Predict 过程
        for step in range(num_iter):
            

            # 模型前向
            outputs = self.llm(
                input_embedding=x_embeds,  # [1, seq, code_layer+1]
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = outputs.logits[0]  # [seq, vocab_sum]

            # 拆分 text/audio logits
            xt_logits = logits[..., :text_vocab_size]  # [seq, text_vocab]
            xa_logits = []
            for i in range(self.code_layer):
                xa_logits.append(
                    logits[..., text_vocab_size + audio_vocab_size * i : text_vocab_size + audio_vocab_size * (i + 1)]
                )  # [seq, audio_vocab]

            # 采样/贪心，mask位置才更新
            # text
            text_mask = (generated_ids[self.code_layer] == pad_t)
            xt_probs = torch.softmax(xt_logits, dim=-1)
            xt_sampled = torch.argmax(xt_probs, dim=-1)
            generated_ids[self.code_layer][text_mask] = xt_sampled[text_mask]

            # audio
            for i in range(self.code_layer):
                audio_mask = (generated_ids[i] == pad_a)
                xa_probs = torch.softmax(xa_logits[i], dim=-1)
                xa_sampled = torch.argmax(xa_probs, dim=-1)
                generated_ids[i][audio_mask] = xa_sampled[audio_mask]

            # if step < num_iter - 1:
            #     # text置信度
            #     text_conf = torch.max(xt_probs, dim=-1).values
            #     gen_mask = (torch.arange(max_new_tokens, device=device) >= prompt_len)
            #     mask_candidate = text_mask | gen_mask
            #     ratio = 1.0 - (step + 1) / num_iter
            #     num_mask = int(mask_candidate.sum().item() * ratio)
            #     if num_mask > 0:
            #         _, idx = torch.topk(text_conf * mask_candidate, k=num_mask, largest=False)
            #         generated_ids[self.code_layer][idx] = pad_t
            #     # audio置信度
            #     for i in range(self.code_layer):
            #         audio_conf = torch.max(torch.softmax(xa_logits[i], dim=-1), dim=-1).values
            #         audio_mask_candidate = (generated_ids[i] == pad_a)
            #         num_mask_audio = int(audio_mask_candidate.sum().item() * ratio)
            #         if num_mask_audio > 0:
            #             _, idx = torch.topk(audio_conf * audio_mask_candidate, k=num_mask_audio, largest=False)
            #             generated_ids[i][idx] = pad_a

        # 截断到 eot/eoa
        text_tokens = generated_ids[self.code_layer]
        eot_pos = (text_tokens == eot).nonzero(as_tuple=True)
        if len(eot_pos[0]) > 0:
            generated_ids[self.code_layer] = text_tokens[:eot_pos[0][0]]
        for i in range(self.code_layer):
            audio_tokens = generated_ids[i]
            eoa_pos = (audio_tokens == eoa).nonzero(as_tuple=True)
            if len(eoa_pos[0]) > 0:
                generated_ids[i] = audio_tokens[:eoa_pos[0][0]]

        return generated_ids


    @torch.no_grad()
    def generate_non_autoregressive_parallel(self,
                                            input_ids: torch.LongTensor = None,
                                            attention_mask: Optional[torch.Tensor] = None,
                                            position_ids: Optional[torch.LongTensor] = None,
                                            past_key_values: Optional[List[torch.FloatTensor]] = None,
                                            inputs_embeds: Optional[torch.FloatTensor] = None,
                                            labels: Optional[torch.LongTensor] = None,
                                            use_cache: Optional[bool] = None,
                                            output_attentions: Optional[bool] = None,
                                            output_hidden_states: Optional[bool] = None,
                                            return_dict: Optional[bool] = None,
                                            **kwargs,
                                            ):
        block_length = kwargs.get("block_length", 64)
        steps = kwargs.get("steps", 1)
        max_new_tokens = kwargs.get("max_new_tokens", 360)
        temperature = kwargs.get("temperature", 1.0)
        stopping_criteria = kwargs.get("stopping_criteria", None)
        upsampling_factor = kwargs.get("upsampling_factor", 1)
        decode_text_only = kwargs.get("decode_text_only", False)
        num_latency_tokens = kwargs.get("num_latency_tokens", 0)

        text_vocab_size = self.model_config.vocab_config.padded_text_vocabsize
        audio_vocab_size = self.model_config.vocab_config.padded_audio_vocabsize
        code_layer = self.code_layer

        pad_t = self.model_config.vocab_config.pad_t
        pad_a = self.model_config.vocab_config.pad_a
        eot = self.model_config.vocab_config.eot
        eoa = self.model_config.vocab_config.eoa

        kwargs["inference_mode"] = True
        # 获取prompt embedding
        print('nnnn',input_ids.shape)
        inputs_embeds, attention_mask = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        print(inputs_embeds.shape)
        batch_size = input_ids.size(0)
        prompt_length = inputs_embeds.shape[1]
        total_length = prompt_length + max_new_tokens

        # 维护一个text_tokens（1*1*total_length）和audio_tokens(1*3*total_length)

        text_tokens = torch.full((batch_size, total_length), pad_t, dtype=torch.long, device=input_ids.device)
        text_tokens[:, :prompt_length] = input_ids[:,code_layer,:]
        audio_tokens = []
        for i in range(code_layer):
            layer_audio = torch.full((batch_size, total_length), pad_a, dtype=torch.long, device=input_ids.device)
            layer_audio[:, :prompt_length] = input_ids[:,i,:]
            audio_tokens.append(layer_audio)

        # embedding函数
        if hasattr(self.llm.model, "embed_tokens"):
            embed_fn = self.llm.model.embed_tokens
        elif hasattr(self.llm.model.model, "embed_tokens"):
            embed_fn = self.llm.model.model.embed_tokens
        else:
            embed_fn = self.llm.model.model.model.embed_tokens


        padid = torch.tensor([[[pad_a], [pad_a], [pad_a], [pad_t]]],device=input_ids.device)
        print('aaaa',padid.shape)


        masked_embed, _ = self.forward(
            input_ids=padid,
            inference_mode=True
        )

        print(masked_embed.shape)  # 1*1*dim


        # 构造一个 batchsize * total_length * dim 的tensor，后半段被mask

        x_embeds = masked_embed.repeat(1, total_length, 1)

        print('total_length',total_length)

        x_embeds[:, :prompt_length, :] = inputs_embeds.clone()

        print('x_embeds.shape',x_embeds)
        
        
        num_blocks = max_new_tokens // block_length
        if max_new_tokens % block_length != 0:
            num_blocks += 1

        # 每次推理一个block
        for block_idx in range(num_blocks):
            block_start = prompt_length + block_idx * block_length
            block_end = min(prompt_length + (block_idx + 1) * block_length, total_length)

            # 每个block执行step步

            for step in range(steps):

                mask_index = torch.all(torch.abs(x_embeds - masked_embed) < 1e-5, dim=2)

                current_block_masks = mask_index[:, block_start:block_end]

                outputs = self.llm(
                    inputs_embeds=x_embeds
                )
                logits = outputs.logits  # [B, SEQ, vocab_sum]

                print(logits.shape) # torch.Size([1, 3318, 156160])

                print(text_vocab_size)

                # 拆分logits
                xt_logits = logits[..., :text_vocab_size]
                xa_logits = [logits[..., text_vocab_size + audio_vocab_size * i: text_vocab_size + audio_vocab_size * (i + 1)] for i in range(code_layer)]

                # 文本层采样
                text_logits_block = xt_logits[:, block_start:block_end, :]
                if temperature > 0:
                    probs = torch.softmax(text_logits_block / temperature, dim=-1)
                    next_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(batch_size, -1)
                else:
                    next_tokens = torch.argmax(text_logits_block, dim=-1)
                for b in range(batch_size):
                    mask_pos = current_block_masks[b].nonzero(as_tuple=True)[0]
                    text_tokens[b, block_start:block_end][mask_pos] = next_tokens[b][mask_pos]

                # 音频层采样
                for i in range(code_layer):
                    audio_logits_block = xa_logits[i][:, block_start:block_end, :]
                    if temperature > 0:
                        print('audio_logits_block',audio_logits_block.shape)
                        probs = torch.softmax(audio_logits_block / temperature, dim=-1)
                        next_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(batch_size, -1)
                    else:
                        next_tokens = torch.argmax(audio_logits_block, dim=-1)
                    for b in range(batch_size):
                        mask_pos = current_block_masks[b].nonzero(as_tuple=True)[0]
                        audio_tokens[i][b, block_start:block_end][mask_pos] = next_tokens[b][mask_pos]

                audio_tokens1 = torch.cat([layershift(audio_tokens[i], i).unsqueeze(1) for i in range(self.code_layer)], dim=1)
                combined_input_ids = torch.cat([audio_tokens1, text_tokens.unsqueeze(1)], dim=1)

                combined_input_emb, _ = self.forward(
                                            input_ids=combined_input_ids,
                                            inference_mode=True
                                        )

                # 更新x_embeds，把mask的部分填充为生成好的

                for b in range(batch_size):
                    mask_pos = current_block_masks[b].nonzero(as_tuple=True)[0]
                    x_embeds[b, block_start:block_end][mask_pos] = combined_input_emb[b, block_start:block_end][mask_pos]


                
                # 终止符截断
                for b in range(batch_size):
                    gen_text = text_tokens[b, prompt_length:]
                    if eot in gen_text:
                        end_pos = (gen_text == eot).nonzero(as_tuple=True)[0][0]
                        text_tokens[b, prompt_length + end_pos + 1:] = pad_t
                    for i in range(code_layer):
                        gen_audio = audio_tokens[i][b, prompt_length:]
                        if eoa in gen_audio:
                            end_pos = (gen_audio == eoa).nonzero(as_tuple=True)[0][0]
                            audio_tokens[i][b, prompt_length + end_pos + 1:] = pad_a

        # 只返回有效部分
        ret = []
        for i in range(code_layer):
            audio_gen = audio_tokens[i][:, prompt_length:]
            ret.append(audio_gen)

        text_gen = text_tokens[:, prompt_length:]
        if upsampling_factor > 1:
            text_gen = text_gen[:, ::upsampling_factor]
        ret.append(text_gen)

        return ret 


    



    @torch.no_grad()
    def generate(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs,
                ):
        kwargs["inference_mode"] = True
        inputs_embeds, attention_mask = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        max_new_tokens = kwargs.get("max_new_tokens", 360)
        generated_ids = [torch.zeros((max_new_tokens,), dtype=torch.long, device=input_ids.device) for _ in range(self.code_layer + 1)]
        current_input_text = None
        current_audio_tokens = [None for _ in range(self.code_layer)]
        past_key_values = None

        text_vocab_size = self.model_config.vocab_config.padded_text_vocabsize #152000
        audio_vocab_size = self.model_config.vocab_config.padded_audio_vocabsize #4160
        # 156160

        num_latency_tokens = kwargs.get("num_latency_tokens", 0)
        text_repetition_penalty = kwargs.get("text_repetition_penalty", 1.0)
        audio_repetition_penalty = kwargs.get("audio_repetition_penalty", 1.0)
        decode_text_only = kwargs.get("decode_text_only", False)
        upsampling_factor = kwargs.get("upsampling_factor", 1)
        do_layershift = kwargs.get("do_layershift", True)
        if do_layershift:
            layershift = layer_shift
        else:
            layershift = simple_shift

        pad_t = self.model_config.vocab_config.pad_t
        pad_a = self.model_config.vocab_config.pad_a
        eot = self.model_config.vocab_config.eot
        eoa = self.model_config.vocab_config.eoa

        text_end = False     # Track whether text generation has ended
        audio_end = False    # Track whether audio generation has ended

        # NOTE: currently, we only support greedy decoding and sampling for parallel generation, no beam search
        for step in tqdm(range(max_new_tokens), desc="Generating"):
            if current_input_text is not None:
                audio_tokens = torch.cat([layershift(current_audio_tokens[i], i).unsqueeze(1) for i in range(self.code_layer)], dim=1)
                combined_input_ids = torch.cat([audio_tokens, current_input_text.unsqueeze(1)], dim=1)
                if hasattr(self.llm.model, "embed_tokens"):
                    inputs_embeds = self.llm.model.embed_tokens(combined_input_ids)
                elif hasattr(self.llm.model.model, "embed_tokens"):
                    inputs_embeds = self.llm.model.model.embed_tokens(combined_input_ids)
                else:
                    inputs_embeds = self.llm.model.model.model.embed_tokens(combined_input_ids)
                inputs_embeds = torch.mean(inputs_embeds, dim=1).unsqueeze(1)
            
            outputs = self.llm(
                inputs_embeds=inputs_embeds,                  # [btz, seq_len / 1, emb_dim]
                attention_mask=attention_mask,                # single sample, no need for attention mask
                past_key_values=past_key_values,
                use_cache=True,
            )
            
            logits = outputs.logits[0]                      # batch size is 1

            print(logits.shape)


            past_key_values = outputs.past_key_values       # Update past_key_values for the next step

            # Split logits into text and audio layers based on vocab size
            xt_logits = logits[..., :text_vocab_size]
            if self.group_decode_adapter is not None:
                xa_logits = self.group_decode_adapter(logits[..., text_vocab_size:])
                xa_logits = [xa_logits[..., i * audio_vocab_size : (i + 1) * audio_vocab_size] for i in range(self.code_layer)]
            else:
                xa_logits = [logits[..., text_vocab_size + audio_vocab_size * i : text_vocab_size + audio_vocab_size * (i + 1)] for i in range(self.code_layer)]

            # Apply repetition penalty to the logits
            xt_logits = self.repetition_penalty(xt_logits, generated_ids[self.code_layer][:step], text_repetition_penalty)
            for i in range(self.code_layer):
                xa_logits[i] = self.repetition_penalty(xa_logits[i], generated_ids[i][:step], audio_repetition_penalty)

            if not text_end:
                next_token_text = self.sample_next_token(xt_logits[-1, :], **kwargs)
            else:
                next_token_text = torch.tensor([pad_t], device=input_ids.device)

            next_tokens_audio = []
            for i in range(self.code_layer):
                if not audio_end and not decode_text_only and num_latency_tokens <= step:
                    next_token_audio = self.sample_next_token(xa_logits[i][-1, :], **kwargs)
                else:
                    next_token_audio = torch.full((input_ids.size(0),), pad_a, device=input_ids.device)
                next_tokens_audio.append(next_token_audio)

            if eoa in next_tokens_audio or decode_text_only:
                audio_end = True
            if next_token_text == eot:
                text_end = True
            
            # Update input_ids for the next step
            current_input_text = next_token_text
            for i in range(self.code_layer):
                current_audio_tokens[i] = next_tokens_audio[i]

            attention_mask = torch.cat([attention_mask, torch.ones((input_ids.size(0), 1), device=input_ids.device)], dim=1)

            # Append generated tokens to the tensor
            for i in range(self.code_layer):
                generated_ids[i][step] = next_tokens_audio[i]  # Audio layers
            generated_ids[self.code_layer][step] = next_token_text  # Text layer

            if audio_end and text_end:
                for i in range(self.code_layer):
                    generated_ids[i] = generated_ids[i][:step+1]
                break

        # Concatenate the generated tokens to form the complete sequence
        text_tokens = generated_ids[self.code_layer]
        generated_ids[self.code_layer] = text_tokens[: (text_tokens == eot).nonzero(as_tuple=True)[0][0]] if eot in text_tokens else text_tokens

        if eoa in generated_ids[self.code_layer - 1] and do_layershift:
            end_ids = (generated_ids[self.code_layer - 1] == eoa).nonzero(as_tuple=True)[0][0]
            for i in range(self.code_layer):
                audio_tokens = generated_ids[i]
                generated_ids[i] = audio_tokens[:end_ids]

        if upsampling_factor > 1:
            generated_ids[self.code_layer] = generated_ids[self.code_layer][::upsampling_factor]
            
        return generated_ids



    @torch.no_grad()
    def stream_generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Generator:
        kwargs["inference_mode"] = True
        inputs_embeds, attention_mask = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        max_new_tokens = kwargs.get("max_new_tokens", 360)
        generated_ids = [torch.zeros((max_new_tokens,), dtype=torch.long, device=input_ids.device) for _ in range(self.code_layer + 1)]
        current_input_text = None
        current_audio_tokens = [None for _ in range(self.code_layer)]
        past_key_values = None

        text_vocab_size = self.model_config.vocab_config.padded_text_vocabsize
        text_vocab_size = self.model_config.vocab_config.padded_audio_vocabsize

        print('vocab',text_vocab_size,text_vocab_size)

        mini_omni_modeling = kwargs.get("mini_omni_modeling", False)
        text_repetition_penalty = kwargs.get("text_repetition_penalty", 1.0)
        audio_repetition_penalty = kwargs.get("audio_repetition_penalty", 1.0)
        decode_text_only = kwargs.get("decode_text_only", False)
        upsampling_factor = kwargs.get("upsampling_factor", 1)
        do_layershift = kwargs.get("do_layershift", True)
        if do_layershift:
            layershift = layer_shift
        else:
            layershift = simple_shift

        pad_t = self.model_config.vocab_config.pad_t
        pad_a = self.model_config.vocab_config.pad_a
        eot = self.model_config.vocab_config.eot
        eoa = self.model_config.vocab_config.eoa

        text_end = False
        audio_end = False
        begin_generate = False
        text_stream_end = False

        stream_stride = kwargs.get("stream_stride", 4)
        current_index = 0
        index = 0
        last_text_index = 0

        for step in tqdm(range(max_new_tokens), desc="Generating"):
            if current_input_text is not None:
                audio_tokens = torch.cat([layershift(current_audio_tokens[i], i).unsqueeze(1) for i in range(self.code_layer)], dim=1)
                combined_input_ids = torch.cat([audio_tokens, current_input_text.unsqueeze(1)], dim=1)
                if hasattr(self.llm.model, "embed_tokens"):
                    inputs_embeds = self.llm.model.embed_tokens(combined_input_ids)
                elif hasattr(self.llm.model.model, "embed_tokens"):
                    inputs_embeds = self.llm.model.model.embed_tokens(combined_input_ids)
                else:
                    inputs_embeds = self.llm.model.model.model.embed_tokens(combined_input_ids)
                inputs_embeds = torch.mean(inputs_embeds, dim=1).unsqueeze(1)
            
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            
            logits = outputs.logits[0]
            past_key_values = outputs.past_key_values

            xt_logits = logits[..., :text_vocab_size]
            if self.group_decode_adapter is not None:
                xa_logits = self.group_decode_adapter(logits[..., text_vocab_size:])
                xa_logits = [xa_logits[..., i * audio_vocab_size : (i + 1) * audio_vocab_size] for i in range(self.code_layer)]
            else:
                xa_logits = [logits[..., text_vocab_size + audio_vocab_size * i : text_vocab_size + audio_vocab_size * (i + 1)] for i in range(self.code_layer)]

            xt_logits = self.repetition_penalty(xt_logits, generated_ids[self.code_layer][:step], text_repetition_penalty)
            for i in range(self.code_layer):
                xa_logits[i] = self.repetition_penalty(xa_logits[i], generated_ids[i][:step], audio_repetition_penalty)

            if not text_end:
                next_token_text = self.sample_next_token(xt_logits[-1, :], **kwargs)
            else:
                next_token_text = torch.tensor([pad_t], device=input_ids.device)

            next_tokens_audio = []
            for i in range(self.code_layer):
                if not audio_end and not decode_text_only:
                    next_token_audio = self.sample_next_token(xa_logits[i][-1, :], **kwargs)
                else:
                    next_token_audio = torch.full((input_ids.size(0),), pad_a, device=input_ids.device)
                next_tokens_audio.append(next_token_audio)

            if eoa in next_tokens_audio or decode_text_only:
                audio_end = True
            if next_token_text == eot:
                text_end = True
            
            current_input_text = next_token_text
            for i in range(self.code_layer):
                current_audio_tokens[i] = next_tokens_audio[i]

            attention_mask = torch.cat([attention_mask, torch.ones((input_ids.size(0), 1), device=input_ids.device)], dim=1)

            for i in range(self.code_layer):
                generated_ids[i][step] = next_tokens_audio[i]
            generated_ids[self.code_layer][step] = next_token_text

            if audio_end and text_end:
                for i in range(self.code_layer):
                    generated_ids[i] = generated_ids[i][:step+1]
                break

            if index == self.code_layer:
                begin_generate = True

            if begin_generate and not decode_text_only and mini_omni_modeling:
                current_index += 1
                if current_index == stream_stride:
                    current_index = 0
                    snac = get_snac(generated_ids, index, stream_stride)
                    audio_stream = generate_audio_data(snac, self.codec_decoder, input_ids.device)
                    text_stream = generated_ids[self.code_layer][last_text_index:index] if not text_stream_end else None

                    if text_stream is not None and eot in text_stream:
                        text_stream_end = True
                        text_stream = text_stream[:text_stream.index(eot)]

                    last_text_index = index
                    yield {
                        "audio_stream": audio_stream,
                        "text_stream": text_stream,
                    }
            
            if not mini_omni_modeling and not decode_text_only:
                yield {
                    "audio_tokens": [next_tokens_audio[i].item() for i in range(self.code_layer)],
                    "text_token": next_token_text.item(),
                }

            if decode_text_only:
                yield {
                    "audio_stream": None,
                    "text_stream": next_token_text,
                }
            
            index += 1

        text_tokens = generated_ids[self.code_layer]
        generated_ids[self.code_layer] = text_tokens[: (text_tokens == eot).nonzero(as_tuple=True)[0][0]] if eot in text_tokens else text_tokens

        if eoa in generated_ids[self.code_layer - 1] and do_layershift:
            end_ids = (generated_ids[self.code_layer - 1] == eoa).nonzero(as_tuple=True)[0][0]
            for i in range(self.code_layer):
                audio_tokens = generated_ids[i]
                generated_ids[i] = audio_tokens[:end_ids]

        if upsampling_factor > 1:
            generated_ids[self.code_layer] = generated_ids[self.code_layer][::upsampling_factor]
            
        return generated_ids


    @torch.no_grad()
    def sample_next_token(self, logits, **kwargs):
        """
        Generate the next token based on the model output logits.
        Supports both greedy decoding, top-k sampling, and top-p (nucleus) sampling.
        """
        do_sample = kwargs.get("do_sample", False)
        temperature = kwargs.get("temperature", 1.0)
        top_k = kwargs.get("top_k", 0)
        top_p = kwargs.get("top_p", 1.0)
        num_samples = kwargs.get("num_samples", 1)

        # Adjust logits with temperature
        logits = logits.squeeze(0)
        logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))  # Make sure top_k is within the vocab size
            values, indices = torch.topk(logits, top_k)
            logits[logits < values[..., [-1]]] = -float('Inf')  # Filter tokens not in top_k

        # Top-p filtering (nucleus sampling)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -float('Inf')

        if do_sample:
            # Perform sampling
            return torch.multinomial(F.softmax(logits, dim=-1), num_samples=num_samples)
        else:
            # Greedy decoding (argmax)
            return torch.argmax(logits, dim=-1, keepdim=True)


    def repetition_penalty(self, logits, generated_ids, repetition_penalty):
        """
        Apply repetition penalty to the logits.
        """
        if repetition_penalty == 1.0:
            return logits

        # Gather the logits for generated_ids
        score = torch.gather(logits, -1, generated_ids.unsqueeze(0))

        # Apply penalty
        score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)

        # Scatter the updated scores back into logits
        logits.scatter_(-1, generated_ids.unsqueeze(0), score)

        return logits
