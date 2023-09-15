import torch
import torch.nn.functional as F
from .pfx_encoder import PrefixEncoder
from transformers.models.blip.modeling_blip import BlipPreTrainedModel, BlipConfig, \
                                                   BlipVisionModel, BlipEncoder, BlipVisionConfig
from transformers.models.blip.modeling_blip_text import BlipTextLMHeadModel

class ConCapModel(BlipPreTrainedModel):
    config_class=BlipConfig

    def __init__(self, 
                 config: BlipConfig, 
                 pfx_config_dict: dict,
                 *inputs,
                 **kwargs):
        
        super().__init__(config, *inputs, **kwargs)
        self.config=config
        # decoder for LM
        self.text_decoder=BlipTextLMHeadModel(config.text_config)
        # vision model
        self.vision_model=BlipVisionModel(config.vision_config)
        self.init_prefix(pfx_config_dict)

        # [ENC] and [BOS]
        # self.enc_token_id=getattr(config.text_config, "enc_token_id", 30523)
        self.decoder_bos_id=config.text_config.bos_token_id
        self.pad_token_id=config.text_config.pad_token_id

        # VisAbstractor
        self.vis_abstr_config=BlipVisionConfig(
            hidden_size=self.config.vision_config.hidden_size,
            num_hidden_layers=3,
            image_size=self.config.vision_config.image_size,
            layer_norm_eps=self.config.vision_config.layer_norm_eps
        )
        self.vis_abstr=BlipEncoder(self.vis_abstr_config)

        # post init work
        self.post_init()
        
    def get_pfx_status(self):
        '''get the status of the prefix modules'''
        return {"ViT_prefix": self.__pfx_vit, 
                "Text_dec_prefix": self.__pfx_decoder}

    def prefix_vit_(self, status:bool=None):
        if status is not None:
            self.__pfx_vit=status
            self.vision_prefix.requires_grad_(status)
        
        return self.__pfx_vit
    
    def prefix_decoder_(self, status:bool=None):
        if status is not None:
            self.__pfx_decoder=status
            self.decoder_prefix.requires_grad_(status)

        return self.__pfx_decoder
    
    def init_prefix(self, pfx_dict: dict):
        self.prefix_hid=pfx_dict["prefix_hidden_size"]
        self.pfx_len=pfx_dict["prefix_len"] 
        self.n_style=pfx_dict["n_cls"]
        self.prefix_proj=pfx_dict["prefix_projection"]
        n_emb=self.pfx_len*self.n_style

        pfx_decoder_config={
            **pfx_dict,
            "hidden_size": self.config.text_config.hidden_size,
            "num_hidden_layers": self.config.text_config.num_hidden_layers
        }

        # init prefix
        self.vision_prefix=torch.nn.Embedding(n_emb, self.config.vision_config.hidden_size)
        self.decoder_prefix=PrefixEncoder(pfx_decoder_config)

        # whether to add prefix to ViT and/or BERT
        self.__pfx_vit=pfx_dict.get("prefix_vit", False)
        self.vision_prefix.requires_grad_(self.__pfx_vit)
        self.__pfx_decoder=pfx_dict.get("prefix_decoder", False)
        self.decoder_prefix.requires_grad_(self.__pfx_decoder)

        # whether to drop the img_aware style feature
        self.shrink_sv=pfx_dict.get("shrink_sv", True)


    def _pfx_img_embed(self, img_embeds, prefix_ids):
        if prefix_ids is None or not self.__pfx_vit:
            return img_embeds
        
        vis_prefix=self.vision_prefix(prefix_ids)
        # easy way: just concat
        img_embeds=torch.cat([img_embeds, vis_prefix], dim=1)
        
        # hard way: through the visual abstractor
        vis_abstr_output=self.vis_abstr(inputs_embeds=img_embeds, 
                                        return_dict=True)
        img_embeds=vis_abstr_output.last_hidden_state

        if self.shrink_sv:
            # we add an additional linear layer to project instead
            img_embeds=img_embeds[:, :-self.pfx_len, :]

        return img_embeds

    def _past_kv_from_pfxid(self, prefix_ids, attention_mask):
        if (prefix_ids is None) or (not self.__pfx_decoder):
            return None, attention_mask
        
        # obtain batch_size
        batch_size=prefix_ids.shape[0]
        prefix_emb=self.decoder_prefix(prefix_ids)

        n_head = self.config.text_config.num_attention_heads
        n_emb_each = self.config.text_config.hidden_size // n_head

        past_kvs=prefix_emb.view(
            batch_size,
            self.pfx_len,
            -1,
            n_head,
            n_emb_each
        )
        # permute to [n_layer*2, bsz, heads, pfx_len, emb_each]
        past_kvs=past_kvs.permute([2,0,3,1,4]).split(2)

        # don't forget to concat attn_mask!!
        if attention_mask is not None:
            attention_mask=torch.cat([
                torch.ones(batch_size, self.pfx_len).to(prefix_ids.device),
                attention_mask
            ],dim=-1)
    
        return past_kvs, attention_mask

    def forward(self, 
                input_ids, 
                attention_mask, 
                pixel_values, 
                labels, 
                prefix_ids=None
        )->dict:

        # batch size
        batch=pixel_values.shape[0]
        # ---------------Obtain image feature using ViT Encoder---------------
        vision_outputs=self.vision_model(pixel_values)
        img_embeds=vision_outputs[0]
        # insert pfx embedding here
        img_embeds=self._pfx_img_embed(img_embeds, prefix_ids)
        img_attn=torch.ones(img_embeds.shape[:-1], dtype=torch.long)

        # ------------------Get text prefix-------------------
        dec_pastkv, attn_mask=self._past_kv_from_pfxid(prefix_ids,
                                                       attention_mask)
        #-----------------------LM----------------------------
        lm_input_ids=input_ids.clone()
        # LM starts with [BOS]
        lm_input_ids[:,0]=self.decoder_bos_id
        lm_attn_mask=attn_mask
        # mask fill
        labels=labels.masked_fill(labels==self.pad_token_id, -100)

        text_lm_output=self.text_decoder(
            input_ids=lm_input_ids,
            attention_mask=lm_attn_mask,
            past_key_values=dec_pastkv,
            labels=labels,
            encoder_hidden_states=img_embeds,
            encoder_attention_mask=img_attn,
            reduction="mean",
            return_dict=True
        )

        loss_lm=text_lm_output.loss
        
        return {"loss_lm": loss_lm,
                "img_embeds": img_embeds,
                "dec_logits": text_lm_output.logits}

    @torch.no_grad()
    def generate(self, pixel_values, 
                 input_ids=None, prefix_ids=None, 
                 attention_mask=None, **generate_kwargs):
        # Img feature
        batch_size = pixel_values.shape[0]
        vision_outputs = self.vision_model(
            pixel_values=pixel_values
        )

        image_embeds=self._pfx_img_embed(vision_outputs[0], prefix_ids)
        image_attention_mask = torch.ones(image_embeds.size()[:-1], 
                                          dtype=torch.long).to(image_embeds.device)
        
        # prepare input ids
        if isinstance(input_ids, list):
            input_ids = torch.LongTensor(input_ids)
        elif input_ids is None:
            input_ids = (
                torch.LongTensor([[self.decoder_bos_id, self.config.text_config.eos_token_id]])
                .repeat(batch_size, 1)
                .to(image_embeds.device)
            )

        input_ids[:, 0] = self.config.text_config.bos_token_id
        attention_mask = attention_mask[:, :-1] if attention_mask is not None else None

        # text prefix
        dec_pastkvs, attention_mask =self._past_kv_from_pfxid(prefix_ids, attention_mask)
        if self.__pfx_decoder and attention_mask is None:
            attention_mask=torch.ones([batch_size, self.pfx_len+1], device=pixel_values.device)

        outputs = self.text_decoder.generate(
            input_ids=input_ids[:, :-1],
            eos_token_id=self.config.text_config.sep_token_id,
            pad_token_id=self.config.text_config.pad_token_id,
            past_key_values=dec_pastkvs,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            **generate_kwargs,
        )

        return outputs