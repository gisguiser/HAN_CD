import torch
import torch.nn as nn
import torch.nn.functional as F
from Fusion import HAN

class Net(nn.Module):
    def __init__(self, args, local_map):
        self.device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')
        self.knowledge_dim = args.knowledge_n
        self.exer_n = args.exer_n
        self.emb_num = args.student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256

        self.stu_g = local_map['stu'].to(self.device)
        self.know_g = local_map['know'].to(self.device)
        self.exer_g = local_map['exer'].to(self.device)


        super(Net, self).__init__()

        # network structure
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.knowledge_emb = nn.Embedding(self.knowledge_dim, self.knowledge_dim)
        self.exercise_emb = nn.Embedding(self.exer_n, self.knowledge_dim)

        self.k_index = torch.LongTensor(list(range(self.stu_dim))).to(self.device)
        self.stu_index = torch.LongTensor(list(range(self.emb_num))).to(self.device)
        self.exer_index = torch.LongTensor(list(range(self.exer_n))).to(self.device)
        #学生的嵌入
        self.FusionLayer1 = HAN([["se","es"]],args.knowledge_n,8,args.knowledge_n,[8],0.6)
        #self.FusionLayer2 = HAN([["se","es"]],self.emb_num,8,self.emb_num,[8],0.6)

        #练习的嵌入
        self.FusionLayer3 = HAN([["es","se"],["ek","ke"]],args.knowledge_n,8,args.knowledge_n,[8],0.6)
        #self.FusionLayer4 = HAN([["es","se"],["ek","ke"]], self.knowledge_dim, 8, self.knowledge_dim, [8], 0.6)

        #概念的嵌入
        self.FusionLayer5 = HAN([["ke", "ek"]], args.knowledge_n, 8, args.knowledge_n, [8], 0.6)
        #self.FusionLayer6 = HAN([["ke", "ek"]], self.knowledge_dim, 8, self.knowledge_dim, [8], 0.6)

        self.prednet_full1 = nn.Linear(2 * args.knowledge_n, args.knowledge_n, bias=False)
        # self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(2 * args.knowledge_n, args.knowledge_n, bias=False)
        # self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(1 * args.knowledge_n, 1)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        """这段代码是一个循环，用于对self对象中的参数进行初始化。具体来说，它使用了named_parameters()方法来遍历self对象的所有参数，并对参数名中包含'weight'的参数进行初始化。
           在循环的每一次迭代中，name表示参数的名称，param表示参数的值。通过判断参数名称中是否包含'weight'字符串，可以筛选出所有权重参数。
           对于筛选出的权重参数，代码使用了nn.init.xavier_normal_()函数进行参数初始化。xavier_normal_是一种初始化权重的方法，它将参数的值初始化为服从Xavier正态分布的随机数。
           该方法有助于在深度神经网络中保持梯度的稳定性。
           总结起来，这段代码的作用是对self对象中的权重参数进行Xavier正态分布的初始化。通过遍历参数并根据名称筛选权重参数，然后对选定的权重参数应用Xavier正态分布初始化方法。"""

    def forward(self, stu_id, exer_id, kn_r):
        all_stu_emb = self.student_emb(self.stu_index).to(self.device)
        exer_emb = self.exercise_emb(self.exer_index).to(self.device)
        kn_emb = self.knowledge_emb(self.k_index).to(self.device)

        # 学生、练习、概念的第一层卷积
        kn_emb1=self.FusionLayer5(self.know_g,kn_emb)
        exer_emb1=self.FusionLayer3(self.exer_g,exer_emb)
        all_stu_emb1 = self.FusionLayer1(self.stu_g,all_stu_emb)
        """#学生、练习、概念的第二层卷积
        kn_emb2, exer_emb2, all_stu_emb2 = self.FusionLayer2(kn_emb1, exer_emb1, all_stu_emb1)"""
        """
        stu_id:tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        exer_id:tensor([9914, 9915, 9916, 9917, 9918, 9919, 9824, 9825, 9826, 9827, 9828, 9829,
        9831, 9834, 9872, 9873, 9874, 9875, 9876, 9877, 9878, 9881, 9888, 9889,
        9890, 9891, 9892, 9893, 9894, 9895, 9743, 9744, 9745, 9746, 9747, 9748,
        9749, 9750, 9856, 9494, 9526, 9472, 9445, 9459, 9495, 9478, 9595, 9659,
        9649, 9663, 9623, 9622, 9607, 9857, 9858, 9861, 9863, 4235, 9473, 9511,
        9477, 9524, 9475, 9483])
        kn_r:tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]])
        """
        # get batch student data 每次获得64个学生，64个练习，和64*123维度的概念嵌入
        batch_stu_emb = all_stu_emb1[stu_id] #
        batch_stu_vector = batch_stu_emb.repeat(1, batch_stu_emb.shape[1]).reshape(batch_stu_emb.shape[0], batch_stu_emb.shape[1], batch_stu_emb.shape[1])

        # get batch exercise data
        batch_exer_emb = exer_emb1[exer_id]  # 32 123
        batch_exer_vector = batch_exer_emb.repeat(1, batch_exer_emb.shape[1]).reshape(batch_exer_emb.shape[0], batch_exer_emb.shape[1], batch_exer_emb.shape[1])

        # get batch student data
        batch_stu_emb = all_stu_emb1[stu_id]  # 32 123
        batch_stu_vector = batch_stu_emb.repeat(1, batch_stu_emb.shape[1]).reshape(batch_stu_emb.shape[0],
                                                                                   batch_stu_emb.shape[1],
                                                                                   batch_stu_emb.shape[1])

        # get batch knowledge concept data
        kn_vector = kn_emb1.repeat(batch_stu_emb.shape[0], 1).reshape(batch_stu_emb.shape[0], kn_emb1.shape[0], kn_emb1.shape[1])

        # Cognitive diagnosis
        preference = torch.sigmoid(self.prednet_full1(torch.cat((batch_stu_vector, kn_vector), dim=2)))
        diff = torch.sigmoid(self.prednet_full2(torch.cat((batch_exer_vector, kn_vector), dim=2)))
        o = torch.sigmoid(self.prednet_full3(preference - diff))

        sum_out = torch.sum(o * kn_r.unsqueeze(2), dim = 1)
        count_of_concept = torch.sum(kn_r, dim = 1).unsqueeze(1)
        output = sum_out / count_of_concept
        return output

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)
"""
在__call__方法中，首先检查module是否具有weight属性。如果存在，就获取该属性的数据，并使用torch.neg函数对其进行取负操作，
然后使用torch.relu函数将结果中的负值变为零。最后，使用add_方法将剪辑后的值加回到原始权重上。
总的来说，这个NoneNegClipper类的作用是将模块对象的权重值剪辑为非负数。它通过将负值变为零来实现这一目的。"""