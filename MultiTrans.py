import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from modules.transformer import TransformerEncoder


class MULTModel(nn.Module):
    def __init__(self, img_H=256, img_W=192, patch_size = 32, dim = 1024):
        """
        """
        super(MULTModel, self).__init__()

        assert img_H % patch_size == 0, 'Image dimensions must be divisible by the patch size H.'
        assert img_W % patch_size == 0, 'Image dimensions must be divisible by the patch size W.'

        num_patches = (img_H // patch_size) * (img_W // patch_size)  # (256 / 32) * (192 / 32) = 48
        patch_dim_22 = 22 * patch_size * patch_size                  # 22 * 32 * 32 = 22528
        patch_dim_3 = 3 * patch_size * patch_size                    # 3 * 32 * 32 = 3072
        patch_dim_1 = 1 * patch_size * patch_size                    # 1 * 32 * 32 = 1024


        self.to_patch_embedding_22 = nn.Sequential(
            # [B, 22, 256, 192] -> [B, 22, 8 * 32, 6 * 32] -> [B, 8 * 6, 32 * 32 * 22]
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            # [B, 48, 32*32*22] -> [B, 48, 2048]
            nn.Linear(patch_dim_22, 11264),
        )
        self.to_patch_embedding_3 = nn.Sequential(
            # [B, 3, 256, 192] -> [B, 3, 8 * 32, 6 * 32] -> [B, 8 * 6, 32 * 32 * 3]
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            # [B, 48, 3072] -> [B, 48, 1024]
            nn.Linear(patch_dim_3, dim),
        )

        self.to_patch_embedding_1 = nn.Sequential(
            # [B, 3, 256, 192] -> [B, 3, 8 * 32, 6 * 32] -> [B, 8 * 6, 32 * 32 * 3]
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            # [B, 48, 3072] -> [B, 48, 1024]
            nn.Linear(patch_dim_1, dim),
        )
        # [B, 48, 32 * 32 * 26]
        
        self.backRearrange = nn.Sequential(Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=8, w=6))

        self.d_l, self.d_a, self.d_v = 1024, 1024, 1024
        combined_dim = self.d_l + self.d_a + self.d_v

        output_dim = 32 * 32 * 26

        self.num_heads = 8
        self.layers = 3
        self.attn_dropout = nn.Dropout(0.1)
        self.attn_dropout_a = nn.Dropout(0.0)
        self.attn_dropout_v = nn.Dropout(0.0)
        self.relu_dropout = nn.Dropout(0.1)
        self.embed_dropout = nn.Dropout(0.25)
        self.res_dropout = nn.Dropout(0.1)
        self.attn_mask = True

        # 2. Crossmodal Attentions
        # if self.lonly:
        self.trans_l_with_a = self.get_network(self_type='la')
        self.trans_l_with_v = self.get_network(self_type='lv')
        # if self.aonly:
        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')
        # if self.vonly:
        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')

        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)    

        # Projection layers
        self.proj1 = nn.Linear(6144, 6144)
        self.proj2 = nn.Linear(6144, 6144)
        self.out_layer = nn.Linear(6144, output_dim)

        self.projConv1 = nn.Conv1d(11264, 1024, kernel_size=1, padding=0, bias=False)
        self.projConv2 = nn.Conv1d(1024, 1024, kernel_size=1, padding=0, bias=False)
        self.projConv3 = nn.Conv1d(1024, 1024, kernel_size=1, padding=0, bias=False)

        


    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)


    def forward(self, x1, x2, x3):
        # Input:
        # x1: [B, 22, 256, 192]
        # x2: [B, 3, 256, 192]
        # x3: [B, 1, 256, 192]

        # Step1: patch_embedding
        x1 = self.to_patch_embedding_22(x1)   # [B, 22, 256, 192] -> [B, 48, 11264]
        x2 = self.to_patch_embedding_3(x2)    # [B, 3, 256, 192]  -> [B, 48, 1024]
        x3 = self.to_patch_embedding_1(x3)    # [B, 1, 256, 192]  -> [B, 48, 1024]

        # Step2: Project & 1D Conv & Permute
        # [B, 48, 1024] -> [B, 1024, 48] 
        x1 = x1.transpose(1, 2)   # [B, 11264, 48]
        x2 = x2.transpose(1, 2)
        x3 = x3.transpose(1, 2)

        # [1024]
        proj_x1 = self.projConv1(x1)
        proj_x2 = self.projConv2(x2)
        proj_x3 = self.projConv3(x3)

        # [48, B, 1024]
        proj_x1 = proj_x1.permute(2, 0, 1)
        proj_x2 = proj_x2.permute(2, 0, 1)
        proj_x3 = proj_x3.permute(2, 0, 1)
  
        # Self_att first [48, B, 1024]
        proj_x1_trans = self.trans_l_mem(proj_x1)
        proj_x2_trans = self.trans_a_mem(proj_x2)
        proj_x3_trans = self.trans_v_mem(proj_x3)


        # Step3: Cross Attention
        # (x3,x2) --> x1
        h_l_with_as = self.trans_l_with_a(proj_x1, proj_x2_trans, proj_x2_trans)  # Dimension (L, N, d_l) [48, B, 1024]
        h_l_with_vs = self.trans_l_with_v(proj_x1, proj_x3_trans, proj_x3_trans)  # Dimension (L, N, d_l) [48, B, 1024]

        cross1 = torch.cat([h_l_with_as, h_l_with_vs], 2)             # [2048]

        # (x1, x3) --> x2
        h_a_with_ls = self.trans_a_with_l(proj_x2, proj_x1_trans, proj_x1_trans)
        h_a_with_vs = self.trans_a_with_v(proj_x2, proj_x3_trans, proj_x3_trans)

        cross2 = torch.cat([h_a_with_ls, h_a_with_vs], 2)

        # (x1,x2) --> x3
        h_v_with_ls = self.trans_v_with_l(proj_x3, proj_x1_trans, proj_x1_trans)
        h_v_with_as = self.trans_v_with_a(proj_x3, proj_x2_trans, proj_x2_trans)

        cross3 = torch.cat([h_v_with_ls, h_v_with_as], 2)

        # Combine by cat
        # 三个[48, B, 2048] -> [48, B, 6144]
        # last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1#[N,6144]
        last_hs = torch.cat([cross1, cross2, cross3], dim=2) #[48, B, 6144]

        # A residual block
        decompo_1 = self.proj1(last_hs)
        decompo_1_relu = F.relu(decompo_1)
        last_hs_proj = self.proj2(decompo_1_relu)
        last_hs_proj += last_hs

        # last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        # last_hs_proj += last_hs
        
        output = self.out_layer(last_hs_proj)      # [48, B, 26624(26 * 32 * 32)]
        output = output.permute(1, 0, 2)      # [B, 8 * 6, 32 * 32 * 26]
        output = self.backRearrange(output)
        
        return output



if __name__ == '__main__':
    encoder = TransformerEncoder(300, 4, 2)
    x = torch.tensor(torch.rand(20, 2, 300))
    print(encoder(x).shape)
