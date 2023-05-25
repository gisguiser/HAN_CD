import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
from GraphLayer import HANLayer,SemanticAttention

"""class HAN(nn.Module):
    def __init__(self,args,local_map):
        self.device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')
        self.knowledge_dim = args.knowledge_n
        self.exer_n = args.exer_n
        self.emb_num = args.student_n
        self.stu_dim = self.knowledge_dim

        #图结构
        self.stu_g = local_map['stu'].to(self.device)
        self.exer_g = local_map['exer'].to(self.device)
        self.kn_g = local_map['know'].to(self.device)

        super(HAN,self).__init__()

        #学生嵌入二部图(SES)的图卷积操作
        self.layer1 = nn.ModuleList()
        self.layer1.append(
            HANLayer([["se","es"]],self.stu_dim,self.stu_dim,0,0.6)
        )

        #练习嵌入三部图（ESE,EKE)的图卷积操作
        self.layer2 = nn.ModuleList()
        self.layer2.append(
            HANLayer([["es","se"],["ek","ke"]],self.exer_n,self.exer_n,0,0.6)
        )

        #概念嵌入二部图(KEK)的图卷积操作
        self.layer3 = nn.ModuleList()
        self.layer3.append(
            HANLayer([["ke", "ek"]], self.knowledge_dim, self.knowledge_dim, 0, 0.6)
        )

    def forward(self,kn_emb,exer_emb,all_Stu_emb):

        #更新概念的嵌入
        for gnn in self.layer3:
            kn_emb = gnn(self.kn_g,kn_emb)


        #更新学生的嵌入
        for gnn in self.layer1:
            all_Stu_emb = gnn(self.stu_g,all_Stu_emb )

        #跟新练习的嵌入
        for gnn in self.layer2:
            exer_emb = gnn(self.exer_g, exer_emb)

        return kn_emb,exer_emb,all_Stu_emb
"""
class HAN(nn.Module):
    def __init__(
        self, meta_paths, in_size, hidden_size, out_size, num_heads, dropout
    ):
        super(HAN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(
            HANLayer(meta_paths, in_size, hidden_size, num_heads[0], dropout)
        )
        for l in range(1, len(num_heads)):
            self.layers.append(
                HANLayer(
                    meta_paths,
                    hidden_size * num_heads[l - 1],
                    hidden_size,
                    num_heads[l],
                    dropout,
                )
            )
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)

        return self.predict(h)