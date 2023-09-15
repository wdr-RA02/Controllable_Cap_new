import torch

def check_foreign_past_kv(susp_pkv, n_layers: int, size_of_each_past: list):
    '''
    The past_kv is expected to be of shape ``(n_layers, 2, [bsz, n_heads, pfx_len, hidden_size//n_heads])``

    and we'll check that out.
    '''
    # dim_1==n_layers?
    err_msg="Dim {2} mismatch, expected {1}, got {0}"
    assert len(susp_pkv)==n_layers, err_msg.format(len(susp_pkv),n_layers,1)
    # all dim_2's == 2?
    dim_2s=[len(x) for x in susp_pkv]
    assert set(dim_2s)=={2}, err_msg.format(dim_2s,"all 2's",2)
    # dim_3=[bsz, n_heads, pfx_len, hidden_size//n_heads]
    dim_3=list(susp_pkv[0][0].shape)
    assert dim_3==size_of_each_past, err_msg.format(dim_3, size_of_each_past, 3)
    # now it's an all-pass


# Borrowed from P-Tuning_v2/PrefixEncoder.py
class PrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 4*layers*hidden)
    '''
    def __init__(self, config):
        super().__init__()

        # 将(bsz, seq_len) 投影到 (bsz, seq_len, 2*layers*hidden)
        # 查阅bliplayer源码知cross attention并没有利用past kv, 因此这里还是照旧写2
        # 总而言之就是先算past_kv啦
        
        # 由于需要对不同的class设计不同的embedding, 故num_embedding需要乘以n_cls
        n_emb=config["prefix_len"] * config["n_cls"]

        # p-tuning v2中提到直接embedding效果不太好
        # 但是本work中可以先按下不表, 最重要是先能跑起来
        self.hidden_size=config["hidden_size"]
        self.n_layers=config["num_hidden_layers"]

        self.prefix_projection=config["prefix_projection"]
        self.prefix_hidden=config["prefix_hidden_size"]

        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(n_emb, self.hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(self.hidden_size, self.prefix_hidden),
                torch.nn.Tanh(),
                torch.nn.Linear(self.prefix_hidden, self.n_layers * 2 * self.hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding(n_emb, self.n_layers * 2 * self.hidden_size)
        

    def forward(self, prefix: torch.Tensor):
        '''
        prefix: [bsz, prefix_len]
        '''
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)

        # out: [bsz, prompt_len, layer*2*hidden_dim]
        return past_key_values