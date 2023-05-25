import json
import random

def build_local_map():
    data_file = './data/self/train_set.json'
    with open('config.txt') as i_f:
        i_f.readline()
        student_n, exer_n, knowledge_n = list(map(eval, i_f.readline().split(',')))

    # e
    # u
    temp_list = []
    with open(data_file, encoding='utf8') as i_f:
        data = json.load(i_f)
    u_from_e = '' # e(src) to k(dst)
    e_from_u = '' # k(src) to k(dst)
    print (len(data))
    for line in data:
            exer_id = line['exer_id'] - 1
            user_id = line['user_id'] - 1
            if (str(exer_id) + '\t' + str(user_id )) not in temp_list or (
                        str(user_id ) + '\t' + str(exer_id)) not in temp_list:
                    u_from_e += str(exer_id) + '\t' + str(user_id ) + '\n'
                    e_from_u += str(user_id ) + '\t' + str(exer_id) + '\n'
                    temp_list.append((str(exer_id) + '\t' + str(user_id )))
                    temp_list.append((str(user_id ) + '\t' + str(exer_id)))
    print("排序结束！")
    with open('./data/self/graph/u_from_e.txt', 'w') as f:
        f.write(u_from_e)
    with open('./data/self/graph/e_from_u.txt', 'w') as f:
        f.write(e_from_u)

if __name__ == '__main__':
    build_local_map()
