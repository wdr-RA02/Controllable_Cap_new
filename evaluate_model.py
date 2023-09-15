import torch, math, json
import utils
import torch.nn.functional as F
from data import build_dataloader, build_pcap_dataset
from model.eval_metric import ImageCaptionMetric

def tqdm_module(is_notebook=False):
    if is_notebook:
        from tqdm import tqdm
    else:
        from tqdm.notebook import tqdm
    
    return tqdm

tqdm=tqdm_module()

@torch.no_grad()
def evaluate_lm(model, 
                config:dict, 
                preprocessor,
                device,
                beam_size:int=1,
                rep_penalty:float=1.0,
                **kwargs):
    print("\n------Enter evaluation------")
    model.eval()

    train_cfg=config["train"]
    batch=train_cfg["batch_size_per_gpu"]
    num_workers=train_cfg["num_workers"]

    eos_token=preprocessor.tokenizer.sep_token
    pad_token=preprocessor.tokenizer.pad_token
    evaluator=ImageCaptionMetric(mul_100=True, eos_token=eos_token, pad_token=pad_token)

    test_ds=build_pcap_dataset(config, preprocessor, device, "test")
    steps=math.ceil(len(test_ds)/batch)
    test_dl=build_dataloader(test_ds, batch=batch, num_workers=num_workers, dist_training=False)

    for i, test_item in tqdm(enumerate(test_dl), \
                                total=steps, desc="Processing test set"):
        pixels=test_item.pop("pixel_values").to(device)
        prefix_ids=test_item.pop("prefix_ids").to(device)

        outputs=model.generate(pixel_values=pixels, 
                                prefix_ids=prefix_ids,
                                num_beams=beam_size,
                                repetition_penalty=rep_penalty,
                                max_length=30)
        outputs=preprocessor.batch_decode(outputs, skip_special_tokens=True)
        
        evaluator.add(outputs, test_item, batch_id="batch%d" % i, from_dl=True)
    
    result=evaluator.evaluate()
    print("\n------Finished evaluation------")
    # round the results
    result={k:round(v, 3) for k,v in result.items()}
    return result

@torch.no_grad()
def inference(model, config, processor, device, inference_items, 
              beam_size=1, rep_penalty=1.0, **kwargs):
    '''
    format of inference_items should be
    ```
    [
        {
            "images": "path/to/image",
            "personality": "one of styles in PCap or its ID",
            "reference": "(optional, just for comparison)"
        },...
    ]
    ```
    
    '''
    # load blip.generation()
    model.eval()
    # load style_dict
    with open(config["dataset"]["style_dict"]) as f:
        style_dict=json.load(f)["items"]
    
    total_styles=config["text_model"]["n_cls"]
    pfx_len=config["text_model"]["prefix_len"]

    assert total_styles==model.n_style and pfx_len==model.pfx_len, \
           "n_style and/or pfx_len mismatch! \
            config: {0}, {2}; model: {1}, {3}".format(total_styles, 
                                                      model.n_style, 
                                                      pfx_len, 
                                                      model.pfx_len)
    
    print("------Inference start-------")
    desc, reference=utils.convert_inference_src(inference_items,
                                processor,
                                style_dict,
                                pfx_len)
    
    pixel_values=desc.pop("pixel_values").to(device)
    prefix_ids=desc.pop("prefix_ids").to(device)
    # obtain past_kvs
    # past_kvs=model.text_decoder.pfx_id_to_pfx_embedding(prefix_ids)
    
    # inference using model.generate()
    res_tokens=model.generate(pixel_values=pixel_values,
                                prefix_ids=prefix_ids,
                                num_beams=beam_size,
                                repetition_penalty=float(rep_penalty),
                                max_length=30)
    
    decoded_results=processor.batch_decode(res_tokens, skip_special_tokens=True)
    # package the results into a dict
    results=[{
        "personality": inference_items[i]["personality"],
        "images":inference_items[i]["images"],
        "comment_model": decoded_results[i],
        "comment_gndtruth": reference[i] 
    } for i in range(len(reference))]

    print("------Inference end-------")

    return results

@torch.no_grad()
def evaluate_itm_batch(model, input_ids, pixel_values, prefix_ids, attention_mask=None):
    batch=pixel_values.shape[0]
    # obtain vision_outputs first
    vision_outputs=model.vision_model(pixel_values)
    img_embeds=vision_outputs[0]
    # insert pfx embedding here
    # pfx_id=[gts, fake, gts(fake_cap)]
    if prefix_ids is not None and model.pfx_vit:
        img_embeds=model._pfx_img_embed(img_embeds, prefix_ids)

    img_attn=torch.ones(img_embeds.shape[:-1], dtype=torch.long)

    # replace the head with [ENC]
    enc_input_ids=input_ids.clone()
    enc_input_ids[:,0]=model.enc_token_id

    text_itm_output=model.text_encoder(
        input_ids=enc_input_ids,
        attention_mask=attention_mask,
        encoder_hidden_states=img_embeds,
        encoder_attention_mask=img_attn,
        return_dict=True
    )
    txt_enc_feature=text_itm_output.last_hidden_state[:,0,:]
    itm_output=model.itm_head(txt_enc_feature)

    return itm_output

@torch.no_grad()
def evaluate_itm(model, eval_dl, device, total_steps, epoch:int):
    last_outputs=None

    for item in tqdm(eval_dl, total=total_steps):
        # print(item)
        pixels=item.pop("pixel_values").to(device)
        input_ids=item.pop("input_ids").to(device)
        prefix_ids=item.pop("prefix_ids").to(device)
        attn_mask=item.pop("attention_mask").to(device)
        # obtain the output of itm
        outputs=evaluate_itm_batch(model, input_ids=input_ids, 
                                   pixel_values=pixels, 
                                   attention_mask=attn_mask, 
                                   prefix_ids=prefix_ids)
        
        if last_outputs is None:
            last_outputs=outputs.clone().detach()
            continue

        last_outputs=torch.cat([last_outputs, outputs], dim=0)
    
    # return classifier probs
    itm_pred=torch.argmax(F.softmax(last_outputs, dim=1),dim=1)

        # calculate itm loss in evaluate set
    itm_items=itm_pred.shape[0]
    itm_label=torch.ones((itm_items,)).long().to(device)

    itm_loss=F.cross_entropy(last_outputs, itm_label)
    tqdm.write("Epoch: {}".format(epoch))
    tqdm.write("ITM loss: {}".format(itm_loss.item()))

    # calculate accuracy
    accu_num=int(itm_pred.eq(1).int().sum())
    itm_accu=accu_num/itm_items
    tqdm.write("ITM accuracy: {:.2f}%".format(itm_accu*100))
    
    itm_result={
        "epoch": epoch,
        "accuracy": round(itm_accu*100, 4),
        "itm_loss": round(float(itm_loss.item()), 4)
    }
    
    return last_outputs, itm_result

@torch.no_grad()
def evaluate_itc_batch(model, pixel_values, input_ids, attention_mask, prefix_ids=None, **kwargs):
    model.eval()
    
    batch=pixel_values.shape[0]
    vision_outputs=model.vision_model(pixel_values)
    img_embeds=vision_outputs[0]
    img_embeds=model._pfx_img_embed(img_embeds, prefix_ids)

    img_cls=F.normalize(model.vision_proj(img_embeds[:,0,:]), dim=-1)
    # ------------------Text feature-------------------------
    # requires to begin with [CLS]
    text_cls_output=model.text_encoder(
        input_ids=input_ids,
        # maybe insert pastkv here if we want to add triplet-like contrast
        attention_mask=attention_mask,
        return_dict=True
    )

    text_embeds=text_cls_output.last_hidden_state
    # text_feature = output of [CLS]
    text_cls=F.normalize(model.text_proj(text_embeds[:,0,:]), dim=-1)
    # calculate similarity, shape=[b,2b]
    sim_t2i=text_cls @ img_cls.T

    return sim_t2i


@torch.no_grad()
def evaluate_itc(model, eval_dl, total_steps, device, epoch:int):
    last_outputs=None

    for item in tqdm(eval_dl, total=total_steps):
        # print(item)
        pixels=item.pop("pixel_values").to(device)
        input_ids=item.pop("input_ids").to(device)
        prefix_ids=item.pop("prefix_ids").to(device)
        attn_mask=item.pop("attention_mask").to(device)
        # obtain the output of itm
        outputs=evaluate_itc_batch(model, pixels, input_ids, attn_mask, prefix_ids)
        outputs=torch.diag(outputs)
        if last_outputs is None:    
            last_outputs=outputs.clone().detach()
            continue

        last_outputs=torch.cat([last_outputs, outputs], dim=0)
        
    # calculate itm loss in evaluate set
    itc_items=last_outputs.shape[0]
    itc_label=torch.tensor(1).to(device)
    
    itc_loss=F.cross_entropy(last_outputs/model.itc_temp, itc_label)
    non_match_num=int((last_outputs<0).int().sum())
    
    tqdm.write("Epoch: {}".format(epoch))
    tqdm.write("ITC loss: {:.4f}".format(itc_loss.item()))
    tqdm.write("Non-match item: {}".format(non_match_num))
    
    itc_result={
        "epoch": epoch,
        "itc_loss": round(float(itc_loss.item()), 6),
        "non_match_items": non_match_num,
        "probs": (last_outputs).cpu()
    }

    return last_outputs, itc_result


from torch.utils.data import Dataset, DataLoader

class StylePrefix(Dataset):
    def __init__(self, pfx_len:int, n_style:int) -> None:
        super().__init__()
        self.pfx_len=pfx_len
        self.n_style=n_style

        self.style_pfxs=torch.arange(self.pfx_len*self.n_style).reshape(n_style, pfx_len)

    def __getitem__(self, index):
        return self.style_pfxs[index, :]
    
    def __len__(self):
        return self.n_style

@torch.no_grad()
def evaluate_itc_fullstyle(model, eval_dl, total_steps, device):
    last_outputs=None
    model.eval()
    # setup style ds and dl
    prefix_len=model.pfx_len
    n_style=model.n_style-1

    style_pfx=StylePrefix(prefix_len, n_style)
    style_dl=DataLoader(style_pfx, batch_size=1, num_workers=4, pin_memory=True)
    # loss
    loss_s2i=torch.zeros([], device=device)
    corrent_num=torch.zeros([])

    for item in tqdm(eval_dl, total=total_steps):
        # print(item)
        pixels=item.pop("pixel_values").to(device)
        input_ids=item.pop("input_ids").to(device)
        attn_mask=item.pop("attention_mask").to(device)
        # get ground_truth style labels
        gts_prefix_ids=(item.pop("prefix_ids")[:,0]//prefix_len).long().to(device)

        # get unified txt_embeds of this batch
        text_embeds=model.text_encoder(input_ids=input_ids,
                                    attention_mask=attn_mask).last_hidden_state
        text_feature=F.normalize(model.text_proj(text_embeds[:,0,:]), dim=-1)

        # get unified img_embeds of this batch
        img_embeds=model.vision_model(pixels)[0]

        # get visual reps wro every style
        img_ft_b=None
        for i, prefixs in enumerate(style_dl):
            b=img_embeds.shape[0]
            # img_embeds=[s个b, h, d]
            prefixs=prefixs.repeat(b,1).to(device)
            # prefixs=[b个s, d]
            # get pfx_aware img feature
            pfx_img_embeds=model._pfx_img_embed(img_embeds, prefixs)
            img_feature=model.vision_proj(pfx_img_embeds[:,0,:])

            if img_ft_b is None:
                img_ft_b=img_feature.clone().detach()
            else:
                img_ft_b=torch.cat([img_ft_b, img_feature], dim=0)
        
        # reshape img_ft to [b, n_style, dim]
        dim=img_ft_b.shape[-1]
        # gather all batches of same style to dim0,
        # i.e. after this img_ft_b will be [n_s, b, d]
        img_ft_b=img_ft_b.view(n_style, -1, dim)
        # then transpose using permute
        img_ft_b=img_ft_b.permute(1,0,2).contiguous()

        # txt=[b, dim]
        # img_ft=[b, n_style, dim]
        match_t2i = torch.bmm(text_feature.unsqueeze(1), 
                              F.normalize(img_ft_b, dim=-1).transpose(-1,-2))
        match_t2i = match_t2i.squeeze(1)
        # loss
        loss_s2i+=F.cross_entropy(match_t2i/model.itc_temp, gts_prefix_ids, reduction="sum")

        # get the style_id accuracy
        corrent_b=(torch.argmax(match_t2i, dim=1).eq(gts_prefix_ids).cpu().sum().int())
        corrent_num+=corrent_b
        # tqdm.write("Correct: {}/{}".format(corrent_b, match_t2i.shape[0]))

    loss_s2i=(loss_s2i / len(eval_dl)).cpu()

    return float(loss_s2i), int(corrent_num)
    


        
