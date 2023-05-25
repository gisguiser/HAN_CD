import argparse
from build_graph import build_graph

class CommonArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(CommonArgParser, self).__init__()
        #命令行解析功能模块
        self.add_argument('--exer_n', type=int, default=831,
                          help='The number for exercise.')
        self.add_argument('--knowledge_n', type=int, default=831,
                          help='The number for knowledge concept.')
        self.add_argument('--student_n', type=int, default=20,
                          help='The number for student.')
        self.add_argument('--gpu', type=int, default=1,
                          help='The id of gpu, e.g. 0.')
        self.add_argument('--epoch_n', type=int, default=5,
                          help='The epoch number of training')
        self.add_argument('--lr', type=float, default=0.0001,
                          help='Learning rate')
        self.add_argument('--test', action='store_true',
                          help='Evaluate the model on the testing set in the training process.')

def construct_local_map(args):
    local_map = {
        'stu': build_graph('stu'),
        'know': build_graph('know'),
        'exer': build_graph( 'exer'),
    }
    return local_map

