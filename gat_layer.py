import torch

class GAT_layer(torch.nn.Module):
    def __init__(self, in_feature, out_feature, dropout, alpha, concat):
        super().__init__()
        self.W = torch.nn.Linear(in_feature, out_feature)
        torch.nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        self.a = torch.nn.Linear(out_feature * 2, 1)
        torch.nn.init.xavier_uniform_(self.a.weight, gain=1.414)
        self.non_linear = torch.nn.LeakyReLU(alpha)

        self.in_feature = in_feature
        self.out_feature = out_feature
        self.dropout = dropout
        self.concat = concat

    def forward(self, h, attn_mask):
        h_hat = self.W(h)
        N_shape = h_hat.shape[0]
        h_1 = h_hat.repeat_interleave(N_shape, dim=0)
        h_2 = h_hat.repeat(N_shape, 1)
        h_3 = torch.cat((h_1, h_2), dim=1).view(N_shape, N_shape, 2 * self.out_feature)
        # print(h_3.shape)
        e = self.non_linear(self.a(h_3).squeeze(2))
        attn = torch.where(attn_mask > 0, e, -1e15 * torch.ones_like(e))
        attn = torch.nn.functional.softmax(attn, dim=1)
        attn = torch.nn.functional.dropout(attn, self.dropout, training=self.training)
        res_h = torch.matmul(attn, h_hat)
        if self.concat:
            return torch.nn.functional.elu(res_h)
        else:
            return res_h

