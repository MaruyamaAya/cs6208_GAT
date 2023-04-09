import torch
from gat_layer import GAT_layer

class GAT_model(torch.nn.Module):
    def __init__(self, num_feature, hidden_dim, dropout, alpha, num_head, num_class):
        super().__init__()
        self.blocks = torch.nn.ModuleList([GAT_layer(num_feature, hidden_dim, dropout, alpha, concat=True) for _ in range(num_head)])
        self.out_block = GAT_layer(hidden_dim * num_head, num_class, dropout, alpha, concat=False)
        self.dropout = dropout

    def forward(self, x, attn_mask):
        x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        x = torch.cat([attn_block(x, attn_mask) for attn_block in self.blocks], dim=1)
        x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        x = torch.nn.functional.elu(self.out_block(x, attn_mask))
        x = torch.nn.functional.log_softmax(x, dim=1)
        return x

