import torch
import torch.nn.functional as F

def topk_topp_filtering(logits, topk=10, topp=0, filter_value=-float('Inf')):
    """
    将topk以外的token的生成概率置为-inf
    :param logits: [b_size, dim]
    :param topk:
    :param filter_value:
    :return:
    """
    assert logits.dim() == 2  # batch size 1 for now - could be updated for more but the code would be less clear
    topk = min(topk, logits.size(-1))  # Safety check
    if topk > 0:
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        indices_to_remove = logits < torch.topk(logits, topk, dim=-1)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if topp > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > topp
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # todo check
        for i in range(sorted_indices_to_remove.size(0)):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i][indices_to_remove] = filter_value

    return logits


@torch.no_grad()
def generate_nucleus(model, 
                     prefix_embeds: torch.Tensor,
                     encoder_hidden_states: torch.Tensor,
                     encoder_attn_mask: torch.Tensor,
                     top_p:float,
                     top_k:float,
                     temp:float,
                     max_len: int=30,
                     **kwargs):
    '''
    args:
    - model: text_decoder
    - prefix_embeds: prefix embedding of ``[b, pfx_len, hidden]``
    - encoder_hidden_states, encoder_attn_mask: output from encdoer
    - top_p: sample from culminated prob
    - top_k
    - temp
    '''

    batch, pfx_len, hidden=prefix_embeds.shape
    device=model.device

    # bos, eos and pad
    bos=model.config.bos_token_id
    eos=model.config.sep_token_id
    pad=model.config.pad_token_id
    # embedding layer
    model_emb=model.get_input_embeddings()
    
    # flag
    done=torch.tensor([False for _ in range(batch)], device=device)
    cur_cap=torch.zeros(batch, 1, device=device, dtype=torch.long).fill_(bos)
    cur_attn_mask=torch.ones(batch, pfx_len, dtype=torch.long).to(device)
    
    cur_len=0
    while cur_len<=max_len:
        # decoding step
        # 1. concat attention_mask with current step
        cur_step_attn=(cur_cap[:,-1]!=pad).to(dtype=torch.long).unsqueeze(-1)
        cur_attn_mask=torch.cat([cur_attn_mask, cur_step_attn], dim=1).to(device)

        cur_cap_emb=model_emb(cur_cap)
        cur_emb=torch.cat([prefix_embeds, cur_cap_emb], dim=1).to(device)
        # 2. obtain next word prediction prob
        model_output=model(inputs_embeds=cur_emb, 
                        attention_mask=cur_attn_mask, 
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attn_mask)

        next_word_prob=model_output.logits[:, -1, :].contiguous()
        next_word_prob=next_word_prob/temp

        # 3. filter using topk_topp
        # ensure top_k <= vocab_size
        filtered_prob=topk_topp_filtering(next_word_prob, top_k, top_p).to(device)

        # 4. sample words using torch.multinomial
        sample=torch.multinomial(F.softmax(filtered_prob, dim=-1), 1)[:,0].to(device)
        # fill pad_token to finished sequence
        sample[done]=pad
        done.masked_fill_(sample==eos, torch.tensor(True, device=device))

        # 5. concat current step
        sample=sample.unsqueeze(-1)
        cur_cap=torch.cat([cur_cap, sample], dim=-1)
        cur_len+=1

        # if all finish, then break
        if False not in done:
            break
    
    return cur_cap
