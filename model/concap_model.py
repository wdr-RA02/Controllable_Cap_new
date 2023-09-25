import torch
import torch.nn.functional as F
from .pfx_encoder import PrefixEncoder
from transformers.models.blip.modeling_blip import BlipPreTrainedModel, BlipConfig, \
                                                   BlipVisionModel, BlipEncoder, BlipVisionConfig
from transformers.models.blip.modeling_blip_text import BlipTextLMHeadModel, \
                                                        CausalLMOutputWithCrossAttentions
from .utils.generation import generate_nucleus

class ConCapModelOld(BlipPreTrainedModel):
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
            num_hidden_layers=4,
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

class ConCapTextModel(BlipTextLMHeadModel):
    '''
    a BlipTextLMHeadModel which allows the generation with prefix_embeds
    '''
    def __init__(self, config):
        super().__init__(config)
        
    def forward(self, 
                input_ids=None, 
                attention_mask=None, 
                position_ids=None, 
                head_mask=None, 
                inputs_embeds=None,
                prefix_embeds=None, 
                encoder_hidden_states=None, 
                encoder_attention_mask=None, 
                labels=None, 
                past_key_values=None, 
                use_cache=None,
                output_attentions=None, 
                output_hidden_states=None, 
                return_dict=None, 
                return_logits=False, 
                is_decoder=True, 
                reduction="mean"):
        '''
        Take care of attention_mask outside of this function! 
        '''
        prefix_len=0
        if prefix_embeds is not None:
            assert prefix_embeds.dim()==3, \
                "prefix_embeds expected to be of shape [b, p, h], got {}" \
                .format(prefix_embeds.shape)
            prefix_len=prefix_embeds.shape[1]

            # concat prefix to the head of input_(whatever)
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("Both input_ids and inputs_embeds exist, it's not allowed")
            elif input_ids is not None and past_key_values is None:
                inputid_embeds=self.get_input_embeddings()(input_ids)
                inputs_embeds=torch.cat([prefix_embeds, inputid_embeds], dim=1)
            elif inputs_embeds is not None and past_key_values is None:
                inputs_embeds=torch.cat([prefix_embeds, inputs_embeds], dim=1)
        
        # use inputs_embeds only
        # calculate loss outside, due to the intro of prefix_embeds

        outputs=self.bert(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            is_decoder=is_decoder,
        )

        if return_logits:
            return prediction_scores[:, :-1, :].contiguous()
        
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, prefix_len:-1, :].contiguous()
            labels = labels[:, 1:].contiguous().to(shifted_prediction_scores.device)
            loss_fct = torch.nn.CrossEntropyLoss(reduction=reduction, label_smoothing=0.1)
            lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            if reduction == "none":
                lm_loss = lm_loss.view(prediction_scores.size(0), -1).sum(1)

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )
    
    def prepare_inputs_for_generation(self, 
                                      input_ids, 
                                      past_key_values=None, 
                                      prefix_embs=None,
                                      attention_mask=None, **model_kwargs):
        
        old_stuff=super().prepare_inputs_for_generation(input_ids, 
                                                     past_key_values, 
                                                     attention_mask, 
                                                     **model_kwargs)
        input_ids=old_stuff.pop("input_ids")
        inputs_emb=self.get_input_embeddings()(input_ids)

        if prefix_embs is not None and past_key_values is None:
            inputs_emb=torch.cat([prefix_embs, inputs_emb], dim=1)
            
            # attn_mask=old_stuff.pop("attention_mask")
            # attn_mask=torch.cat([torch.ones(batch, pfx_len, dtype=torch.long),\
            #                      attn_mask], dim=1).to(self.device)
            
            # old_stuff["attention_mask"]=attn_mask
        
        old_stuff["inputs_embeds"]=inputs_emb

        return old_stuff

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
        self.text_decoder=ConCapTextModel(config.text_config)
        # vision model
        self.vision_model=BlipVisionModel(config.vision_config)
        self.init_prefix(pfx_config_dict)

        # [ENC] and [BOS]
        # self.enc_token_id=getattr(config.text_config, "enc_token_id", 30523)
        self.decoder_bos_id=config.text_config.bos_token_id
        self.pad_token_id=config.text_config.pad_token_id

        # post init work
        self.post_init()

    
    def init_prefix(self, pfx_dict: dict):
        self.prefix_hid=pfx_dict["prefix_hidden_size"]
        self.pfx_len=pfx_dict["prefix_len"] 
        self.n_style=pfx_dict["n_cls"]
        self.prefix_proj=pfx_dict["prefix_projection"]

        n_emb=self.pfx_len*self.n_style
        dec_hidden=self.config.text_config.hidden_size

        # init prefix
        # [b, pfx_len] => [b, pfx_len, hidden]
        self.decoder_prefix=torch.nn.Embedding(n_emb, dec_hidden)
    

    def embed_from_ids(self, input_ids, prefix_ids):
        # convert prefix_ids to prefix_embs, shape=[b, pfx_len, hidden]
        prefix_embs=self.decoder_prefix(prefix_ids)

        # input_ids -> input_embeds, shape=[b, seq_len, hidden]
        input_embs=self.text_decoder.get_input_embeddings()(input_ids)
        # concat
        input_embs=torch.cat([prefix_embs, input_embs], dim=1).to(input_embs.device)

        return input_embs
    

    def forward(self, 
                input_ids, 
                attention_mask, 
                pixel_values, 
                labels=None, 
                past_key_values=None,
                prefix_ids=None
        )->dict:
        
        batch, pfx_len=prefix_ids.shape
        # ---------------Obtain image feature using ViT Encoder---------------
        vision_outputs=self.vision_model(pixel_values)
        img_embeds=vision_outputs[0]
        img_attn=torch.ones(img_embeds.shape[:-1], dtype=torch.long)

        #-----------------------LM----------------------------
        lm_input_ids=input_ids.clone()
        # LM starts with [BOS]
        lm_input_ids[:,0]=self.decoder_bos_id

        # input_ids -> input_embeds, shape=[b, seq_len, hidden]
        # input_embs=self.embed_from_ids(input_ids=input_ids, prefix_ids=prefix_ids)
        # also concat attention mask (ie: ones w/ [b, pfx_len])
        prefix_embs=self.decoder_prefix(prefix_ids)
        pfx_attn_mask=torch.ones((batch, pfx_len)).to(self.device)
        attn_mask=torch.cat([pfx_attn_mask, attention_mask], dim=1).to(self.device)

        text_lm_output=self.text_decoder(
            input_ids=lm_input_ids,
            prefix_embeds=prefix_embs,
            attention_mask=attn_mask,
            past_key_values=past_key_values,
            encoder_hidden_states=img_embeds,
            encoder_attention_mask=img_attn,
            labels=labels,
            reduction="mean",
            return_dict=True
        )

        # calculate loss from the logits w.o. prefix logits
        dec_logits = text_lm_output.logits
        loss_lm = text_lm_output.loss
        
        return {"loss_lm": loss_lm,
                "img_embeds": img_embeds,
                "dec_logits": dec_logits}

    @torch.no_grad()
    def generate(self, 
                 pixel_values, 
                 input_ids=None, 
                 prefix_ids=None,
                 past_key_values=None, 
                 attention_mask=None, 
                 **generate_kwargs):
        # Img feature
        batch_size = pixel_values.shape[0]
        vision_outputs = self.vision_model(
            pixel_values=pixel_values
        )

        image_embeds=vision_outputs[0]
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

        # get embs
        prefix_embs = None
        if prefix_ids is not None:
            prefix_embs=self.decoder_prefix(prefix_ids)
            batch, pfx_len=prefix_embs.shape[:-1]
            pfx_attn_mask=torch.ones((batch, pfx_len)).to(prefix_embs.device)
            if attention_mask is None:
                attention_mask=torch.ones((batch,1)).to(prefix_embs.device)

            attention_mask=torch.cat([pfx_attn_mask, attention_mask], dim=1) \
                                .to(attention_mask.device)

        # seq=generate_nucleus(self.text_decoder, 
        #                      prefix_embeds=prefix_embs,
        #                      encoder_hidden_states=image_embeds,
        #                      encoder_attn_mask=image_attention_mask,
        #                      top_k=self.config.text_config.top_k,
        #                      top_p=self.config.text_config.top_p,
        #                      temp=0.75)

        outputs = self.text_decoder.generate(
            input_ids=input_ids[:, :-1],
            eos_token_id=self.config.text_config.sep_token_id,
            pad_token_id=self.config.text_config.pad_token_id,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            prefix_embs=prefix_embs,
            **generate_kwargs,
        )
        
        return outputs

        
