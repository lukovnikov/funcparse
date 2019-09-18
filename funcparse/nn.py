from typing import Set

import torch


class TokenEmb(torch.nn.Module):
    def __init__(self, emb:torch.nn.Embedding, rare_token_ids:Set[int]=None, rare_id:int=None, **kw):
        super(TokenEmb, self).__init__(**kw)
        self.emb = emb
        self.rare_token_ids = rare_token_ids
        self.rare_id = rare_id
        if rare_id is not None and rare_token_ids is not None:
            # build id mapper
            id_mapper = torch.arange(emb.num_embeddings)
            for id in self.rare_token_ids:
                id_mapper[id] = self.rare_id
            self.register_buffer("id_mapper", id_mapper)
        else:
            self.register_buffer("id_mapper", None)

    def forward(self, x:torch.Tensor):
        if self.id_mapper is not None:
            x = self.id_mapper[x]
        ret = self.emb(x)
        return ret