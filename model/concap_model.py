import torch
import torch.nn.functional as F
from .pfx_encoder import PrefixEncoder
from transformers.models.blip.modeling_blip import BlipPreTrainedModel, BlipConfig, \
                                                   BlipVisionModel, BlipEncoder, BlipVisionConfig
from transformers.models.blip.modeling_blip_text import BlipTextLMHeadModel, \
                                                        CausalLMOutputWithCrossAttentions
from .utils.generation import generate_nucleus

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

        
