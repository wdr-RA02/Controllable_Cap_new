import torch
import torch.nn.functional as F
from .utils import pretrain_utils
from .pfx_encoder import PrefixEncoder
from transformers.models.blip.modeling_blip import BlipPreTrainedModel, BlipConfig, \
                                                   BlipVisionModel, BlipEncoder, BlipVisionConfig
from transformers.models.blip.modeling_blip_text import BlipTextModel, BlipTextLMHeadModel

class ConCapPretrainedModel(BlipPreTrainedModel):
    config_class=BlipConfig

    def __init__(self, 
                 config: BlipConfig, 
                 pfx_config_dict: dict,
                 *inputs,
                 **kwargs):
        
        super().__init__(config, *inputs, **kwargs)
        self.config=config
        # encoder for ITC/ITM (anyway...)
        self.text_encoder=BlipTextModel(config.text_config, add_pooling_layer=False)
        # decoder for LM
        self.text_decoder=BlipTextLMHeadModel(config.text_config)
        # projection layers for ITC
        self.vision_proj=torch.nn.Linear(config.vision_config.hidden_size, config.image_text_hidden_size)
        self.text_proj=torch.nn.Linear(config.text_config.hidden_size, config.image_text_hidden_size)
        self.itc_temp=torch.nn.Parameter(torch.tensor(0.08), requires_grad=False)
        # vision model
        self.vision_model=BlipVisionModel(config.vision_config)
        self.init_prefix(pfx_config_dict)

        # [ENC] and [BOS]
        self.enc_token_id=getattr(config.text_config, "enc_token_id", 30523)
        self.decoder_bos_id=config.text_config.bos_token_id
        self.pad_token_id=config.text_config.pad_token_id

        # VisAbstractor
        self.vis_abstr_config=BlipVisionConfig(
            hidden_size=self.config.vision_config.hidden_size,
            num_hidden_layers=4,
            image_size=self.config.vision_config.image_size,
            layer_norm_eps=self.config.vision_config.layer_norm_eps
        )
        self.vis_abstr=BlipEncoder(self.vis_abstr_config)

        # post init work
        self.post_init()
        self.tie_enc_dec_weights()
        

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


    def tie_enc_dec_weights(self):
        pretrain_utils.tie_encoder_decoder_weights(self.text_encoder, 
                                            self.text_decoder.bert, 
                                            "", "/attention")

    def _pfx_img_embed(self, img_embeds, prefix_ids):
        if prefix_ids is None or not self.__pfx_vit:
            return img_embeds
        
        vis_prefix=self.vision_prefix(prefix_ids)
        # easy way: just concat
        img_embeds=torch.cat([img_embeds,vis_prefix], dim=1)

        # hard way: through the visual abstractor
        vis_abstr_output=self.vis_abstr(inputs_embeds=img_embeds, 
                                        return_dict=True)
        img_embeds=vis_abstr_output.last_hidden_state

        if self.shrink_sv:
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

    def __itm(self, input_ids, prefix_ids, img_embeds, attention_mask, batch:int):
        # Image embeds = [b, p, d]
        # ------------------Img feature-------------------------
        img_embeds=img_embeds.repeat(3,1,1)
        prefix_ids=torch.cat([prefix_ids,prefix_ids[0:batch]], dim=0)
        img_embeds=self._pfx_img_embed(img_embeds, prefix_ids)
        img_attn=torch.ones(img_embeds.shape[:-1], dtype=torch.long)

        # ------------------Text feature-------------------------
        # requires to begin with [ENC]
        enc_input_ids=torch.cat([input_ids[0:batch],
                                 input_ids[0:batch],
                                 input_ids[batch:]], dim=0)
        enc_attn_mask=torch.cat([attention_mask[0:batch],
                                 attention_mask[0:batch],
                                 attention_mask[batch:]], dim=0)
        # enc_input_ids= [all_gts, fake_style, fake_cap]
        enc_input_ids[:,0]=self.enc_token_id

        text_itm_output=self.text_encoder(
            input_ids=enc_input_ids,
            attention_mask=enc_attn_mask,
            encoder_hidden_states=img_embeds,
            encoder_attention_mask=img_attn,
            return_dict=True
        )
        txt_enc_feature=text_itm_output.last_hidden_state[:,0,:]
        itm_output=self.itm_head(txt_enc_feature)
        # itm_label=[1*b, 0*2b]
        itm_label=torch.cat([torch.ones(batch,dtype=torch.long),\
                             torch.zeros(2*batch,dtype=torch.long)], dim=0).to(input_ids.device)
        loss_itm=F.cross_entropy(itm_output, itm_label)

        return {"txt_feature": txt_enc_feature, 
                "itm_output": itm_output,
                "loss": loss_itm}
    
    def __itc(self, input_ids, prefix_ids, img_embeds, attention_mask, batch:int):
        # image_feature = output of [CLS]
        
        # temp change: don't copy yet, since no negative example is used
        # img_embeds=img_embeds.repeat(2,1,1)
        # temp change ends

        img_embeds=self._pfx_img_embed(img_embeds, prefix_ids)
        img_cls=F.normalize(self.vision_proj(img_embeds[:,0,:]), dim=-1)
        # prefix id
        dec_pastkv, attn_mask=self._past_kv_from_pfxid(prefix_ids,
                                                       attention_mask)
        
        # ------------------Text feature-------------------------
        # requires to begin with [CLS]
        text_cls_output=self.text_encoder(
            input_ids=input_ids,
            # maybe insert pastkv here if we want to add triplet-like contrast
            past_key_values=dec_pastkv,
            attention_mask=attn_mask,
            return_dict=True
        )
        text_embeds=text_cls_output.last_hidden_state
        # text_feature = output of [CLS]
        text_cls=F.normalize(self.text_proj(text_embeds[:,0,:]), dim=-1)

        # calculate similarity, shape=[b,2b]
        # since we train the style pfx of both features
        # !!!! we require both prefixes to be on
        # gts text vs [gts img, fake style img]
        sim_t2i=text_cls[0:batch,:] @ img_cls.T / self.itc_temp
        # gts img vs [gts text, fake style text]
        sim_i2t=img_cls[0:batch,:] @ text_cls.T / self.itc_temp

        loss_t2i=F.cross_entropy(sim_t2i, torch.arange(batch, device=sim_t2i.device))
        loss_i2t=F.cross_entropy(sim_i2t, torch.arange(batch, device=sim_i2t.device))

        loss_itc=(loss_i2t+loss_t2i)/2.0

        return {"img_cls": img_cls,
                "text_cls": text_cls,
                "img_embeds": img_embeds[0:batch],
                "loss": loss_itc}


    def forward(self, input_ids, attention_mask, pixel_values, labels, prefix_ids=None):
        # batch size
        batch=pixel_values.shape[0]
        # ---------------Obtain image feature using ViT Encoder---------------
        vision_outputs=self.vision_model(pixel_values)
        img_embeds=vision_outputs[0]
        # insert pfx embedding here
        # pfx_id=[gts, fake, gts(fake_cap)]
        

        # ------------------Get text prefix-------------------
        dec_pastkv, attn_mask=self._past_kv_from_pfxid(prefix_ids[0:batch],
                                                       attention_mask[0:batch])
        dec_neg_pastkv, neg_attn_mask=self._past_kv_from_pfxid(prefix_ids[batch:],
                                                               attention_mask[batch:])

        # ------------------ITC or ITM------------------------
        # temp change: dont use negative examples for itc
        input_ids=input_ids[0:batch,:]
        prefix_ids=prefix_ids[0:batch, :]
        attention_mask=attention_mask[0:batch, :]
        # temp change ends

        mod_align_output=self.__itc(input_ids=input_ids,
                                    img_embeds=img_embeds, 
                                    prefix_ids=prefix_ids, 
                                    attention_mask=attention_mask, 
                                    batch=batch)
        
        loss_itc=mod_align_output["loss"]
        img_embeds_style=mod_align_output["img_embeds"]
        img_attn=torch.ones(img_embeds_style.shape[:-1], dtype=torch.long)

        #-----------------------LM----------------------------
        lm_input_ids=input_ids[0:batch,:].clone()
        # LM starts with [BOS]
        lm_input_ids[:,0]=self.decoder_bos_id
        lm_attn_mask=attn_mask[0:batch,:]
        # mask fill
        labels=labels.masked_fill(labels==self.pad_token_id, -100)

        text_lm_output=self.text_decoder(
            input_ids=lm_input_ids,
            attention_mask=lm_attn_mask,
            past_key_values=dec_pastkv,
            labels=labels,
            encoder_hidden_states=img_embeds_style,
            encoder_attention_mask=img_attn,
            reduction="mean",
            return_dict=True
        )

        loss_lm=text_lm_output.loss
        
        return {"loss_itc": loss_itc, 
                "loss_lm": loss_lm,
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
        dec_pastkvs, _ =self._past_kv_from_pfxid(prefix_ids, None)
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