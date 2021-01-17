import networkx as nx
import numpy as np
import tqdm
import time
import pandas as pd
import random as ra
from eval_metrics import *

def load_training_data(checkin_file, friendship_file, istag, isweek, hour_interval):
    if istag:
        u_p_edges, p_p_edges = generate_edge_with_tag(checkin_file, isweek, hour_interval)
    else:
        u_p_edges, p_p_edges = generate_edge_without_tag(checkin_file, edge_interval= 24)
    u_u_edges = generate_u_u_edges(friendship_file)
    return u_p_edges, p_p_edges, u_u_edges

def judge_time_interval(cur_line, next_line, time_interval=24):

    cur_records = cur_line.strip(' \n').split('\t')
    next_records = next_line.strip(' \n').split('\t')

    cur_userid = cur_records[0]
    next_userid = next_records[0]
    cur_time = cur_records[1]
    next_time = next_records[1]

    cur_timestamp = str_to_timestamp(cur_time)
    next_timestamp = str_to_timestamp(next_time)
    if (cur_userid == next_userid) and ((next_timestamp - cur_timestamp) <= time_interval * 60 * 60):
        return True
    else:
        return False


def generate_edge_without_tag(f_name, edge_interval= 24):

    # print('We are loading data from:', f_name)
    u_p_edges = []
    p_p_edges = []
    readfile =  open(f_name, 'r', encoding='utf-8')
    f = readfile.readlines()
    for i in range(len(f)-1):
        try:
            f1 = f[i]
            f2 = f[i+1]
            user_id = f1.split('\t')[0]

            poi_id = f1.split('\t')[2]
            u_p_edges.append((user_id, poi_id))
            u_p_edges.append((poi_id, user_id))
            if (judge_time_interval(f1, f2, edge_interval)):
                p_p_edges.append((poi_id, f2.split('\t')[2]))

        except:
            pass
    u_p_edges.append((f[-1].split('\t')[0], f[-1].split('\t')[2]))
    u_p_edges.append((f[-1].split('\t')[2], f[-1].split('\t')[0]))

    return u_p_edges, p_p_edges


def generate_u_u_edges(f_name):

    u_u_edges = []
    with open(f_name, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n').split(',')
            u_u_edges.append((line[0], line[1]))
            u_u_edges.append((line[1], line[0]))
    return u_u_edges


def read_history(f_name):

    history = dict()
    with open(f_name, 'r', encoding= 'utf-8') as f:
        for line in f:
            line = line.strip(' \n').split('\t')
            if line[0] not in history.keys():
                history[line[0]] = set()
            history[line[0]].add(line[2])
    return history

def get_G_from_edges(u_p_edges, p_p_edges, u_u_edges, history, threshold):
    edge_dict = {}
    temp_G =nx.DiGraph()
    edges = [u_p_edges, p_p_edges]
    for elements in edges:
        for element in elements:
            edge_key = str(element[0]) + '>' + str(element[1])
            if edge_key not in edge_dict:
                edge_dict[edge_key] = 1
            else:
                edge_dict[edge_key] += 1

    u_u_edge_dict = {}
    for u_u in u_u_edges:
        edge_key = str(u_u[0]) + '>' + str(u_u[1])
        weight = get_weight_between_users(history, u_u[0], u_u[1], threshold)
        # if edge_key not in u_u_edge_dict:
        u_u_edge_dict[edge_key] = weight

    for edge_key in edge_dict:
        weight = edge_dict[edge_key]
        x = edge_key.split('>')[0]
        y = edge_key.split('>')[1]
        temp_G.add_edge(x, y)
        temp_G[x][y]['weight'] = weight


    for edge_key in u_u_edge_dict:

        weight = u_u_edge_dict[edge_key]
        x = edge_key.split('>')[0]
        y = edge_key.split('>')[1]
        temp_G.add_edge(x, y)
        temp_G[x][y]['weight'] = weight

    print('number of nodes G: ', temp_G.number_of_nodes())
    print('number of edges G: ', temp_G.number_of_edges())
    return temp_G

def get_weight_between_users(history, user1, user2, threshold):
    try:
        denominator = 1.0
        numerator = len(history[user1] & history[user2]) + threshold
        return numerator/denominator
    except:
        return threshold
        pass


def write_G_to_file(G, fname):
    with open(fname, 'w', encoding='utf-8') as f:
        for n,nbrs in G.adjacency():
            for nbr,eattr in nbrs.items():
                data = eattr['weight']
                f.write(n+'\t'+nbr+'\t'+str(data)+'\n')
                # print(n,'\t',nbr,'\t',data,'\n')
    print('Write G finished!!')

def write_G_same_edge_type_to_file(G, fname):
    with open(fname, 'w', encoding='utf-8') as f:
        for n,nbrs in G.adjacency():
            for nbr,eattr in nbrs.items():
                data = eattr['weight']
                f.write('1'+'\t'+n+'\t'+nbr+'\t'+str(data)+'\n')
                # print(n,'\t',nbr,'\t',data,'\n')
    print('Write G finished!!')


def get_node_type(fname):
    node_type = {}
    with open(fname, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n').split('\t')
            node_type[line[0]] = line[1]
    return node_type

def judge_edge_type(nodei, nodej, node_type):

    edge_type = None
    if (node_type[nodei] == 'U') & (node_type[nodej] == 'P'):
        edge_type = '0'
    if (node_type[nodei] == 'P') & (node_type[nodej] == 'U'):
        edge_type = '0'
    if (node_type[nodei] == 'P') & (node_type[nodej] == 'P'):
        edge_type = '1'
    if (node_type[nodei] == 'U') & (node_type[nodej] == 'U'):
        edge_type = '2'
    return edge_type

def write_G_diff_edge_type_to_file(G, fname, node_type):
    with open(fname, 'w', encoding='utf-8') as f:
        for n,nbrs in G.adjacency():
            for nbr,eattr in nbrs.items():
                data = eattr['weight']
                edge_type = judge_edge_type(n, nbr, node_type)
                f.write(edge_type +'\t'+n+'\t'+nbr+'\t'+str(data)+'\n')
                # f.write(edge_type +'\t'+n+'\t'+nbr +'\n')
                # print(n,'\t',nbr,'\t',data,'\n')
    print('Write G finished!!')

def write_dict_to_file(dict_data, fname):
    with open(fname, 'w', encoding='utf-8') as f:
        for k,v in dict_data.items():
            f.write(k + '\t' + v + '\n')


def gen_node_type_with_tag_from_G(G):
    node_type={}
    for n,nbrs in G.adjacency():
        for nbr in nbrs.keys():
            if '-' in nbr:
                node_type[nbr] = 'P'
            else:
                node_type[nbr] = 'U'
            # print(n,'\t',nbr,'\t',data,'\n')
    print("node_type dict has ",len(node_type), " nodes!")
    return node_type

def gen_node_type_without_tag_from_G(G, checkin_file, friendship_file):
    user_set = set()
    with open(checkin_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip(' \n').split('\t')
            user_id = line[0]
            if user_id not in user_set:
                user_set.add(user_id)

    with open(friendship_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip(' \n').split(',')
            user1_id = line[0]
            user2_id = line[1]
            if user1_id not in user_set:
                user_set.add(user1_id)
            if user2_id not in user_set:
                user_set.add(user2_id)
    node_type={}
    for n,nbrs in G.adjacency():
        for nbr in nbrs.keys():
            if nbr in user_set:
                node_type[nbr] = 'U'
            else:
                node_type[nbr] = 'P'
            # print(n,'\t',nbr,'\t',data,'\n')
    print("node_type dict has ",len(node_type), " nodes!")
    return node_type


def read_vec(fname):
    final_model = dict()
    flag = True
    with open(fname,'r',encoding='utf-8') as f:
        for line in f:
            if flag:
                flag = False
            else:
                line = line.strip(' \n').split(' ')
                node_id = line[0]
                node_vec = []
                for i in range(len(line)-1):
                    node_vec.append(eval(line[i+1]))
                final_model[node_id] = node_vec
    return final_model


def get_score_norm(local_model, node1, node2):
    try:
        vector1 = local_model[node1]
        vector2 = local_model[node2]
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    except Exception as e:
        pass

def get_score(vec1_list, vec2_list):
    vec1_list = np.array(vec1_list)
    vec2_list = np.array(vec2_list)
    score = np.dot(vec1_list, vec2_list)
    return score

def get_node_type(fname):
    node_type = {}
    with open(fname, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n').split('\t')
            node_type[line[0]] = line[1]
    return node_type

def read_records_sampling(fname, sample_num, random_state=0):

    records = []
    test_data = pd.read_csv(fname,dtype='str',header=None,sep='\t').sample(n=sample_num, random_state=random_state, axis=0)
    for row in range(sample_num):
        record=[]
        for elem in test_data.iloc[row]:
            record.append(elem)
        record = tuple(record)
        records.append(record)
    return records

def read_history(fname):

    history = dict()
    with open(fname, 'r', encoding= 'utf-8') as f:
        for line in f:
            line = line.strip(' \n').split('\t')
            if line[0] not in history.keys():
                history[line[0]] = set()
            history[line[0]].add(line[2])
    return history

def read_future_history_with_tag(fname):
    future_history = dict()
    with open(fname, 'r', encoding= 'utf-8') as f:
        for line in f:
            line = line.strip(' \n').split('\t')
            if line[0] not in future_history.keys():
                future_history[line[0]] = set()
            future_history[line[0]].add(line[1].split('-')[0])
            future_history[line[0]].add(line[2].split('-')[0])
    return future_history

def normalize(vec):

    norm = np.linalg.norm(np.array(vec), ord=2)
    norm_vec = []
    for elem in vec:
        # norm_elem = round(elem / norm, 6)
        norm_elem = elem / norm
        norm_vec.append(norm_elem)
    return norm_vec

def normalize_dict(final_model):
    for k,v in final_model.items():
        final_model[k] = normalize(final_model[k])
    return final_model

def connecate(model1, model2):

    for k,v in model1.items():
        model1[k] = v + model2[k]
    return model1

def read_records(fname):

    records = []
    with open(fname, 'r',encoding='utf-8') as f:
        for line in f:

            candidate_pois = []
            line = line.strip(' \n').split('\t')
            num_items = len(line)
            for i in range(2, num_items):
                candidate_pois.append(line[i])
            record = (line[0], line[1], candidate_pois)
            records.append(record)
    return records

def evaleate_all_index_LSTM_no_history(candidate_pois_array, LSTM_test_target_poi_list, node_type, final_model2):


    accuracy = []
    precision = []
    recall = []
    ndcg = []
    hit_ratio = []
    MAP = []
    index_main = -1
    candidate_poi_list = [[] for i in range(len(candidate_pois_array))]

    with open('train_log.txt', 'w', encoding='utf-8') as f_log:

        for poi_tensor in tqdm.tqdm(candidate_pois_array):
            index_main = index_main + 1
            poi_scores = {}
            score = 0.0

            for k1,v1 in final_model2.items():
                try:
                    if (node_type[k1]=='P'):
                        score = get_score(poi_tensor, v1)
                        poi_scores[k1] = score
                    else:
                        continue
                except:
                    pass
            sorted_poi_scores = sorted(poi_scores.items(), key=lambda item:item[1], reverse = True)
            for t in sorted_poi_scores:
                candidate_poi_list[index_main].append(t[0])
                if len(candidate_poi_list[index_main]) >= 20:
                    break
# ==========================================================================================================================
        for k in tqdm.tqdm([1, 5, 10, 15, 20]):
            accuracy.append(accuracy_at_k(LSTM_test_target_poi_list, candidate_poi_list, k))
            precision.append(precision_at_k(LSTM_test_target_poi_list, candidate_poi_list, k))
            recall.append(recall_at_k(LSTM_test_target_poi_list, candidate_poi_list, k))
            ndcg.append(ndcg_at_k(LSTM_test_target_poi_list, candidate_poi_list, k))
            hit_ratio.append(hit_ratio_at_k(LSTM_test_target_poi_list, candidate_poi_list, k))
            MAP.append(mapk(LSTM_test_target_poi_list, candidate_poi_list, k))
        print('accuracy: ', accuracy)
        print('precision: ', precision)
        print('recall: ', recall)
        print('ndcg: ', ndcg)
        print('hit_ratio: ', hit_ratio)
        print('MAP: ', MAP)
        f_log.write('accuracy: '+ str(accuracy)+'\n')
        f_log.write('precision: '+ str(precision)+'\n')
        f_log.write('recall: '+  str(recall)+ '\n')
        f_log.write('ndcg: '+ str(ndcg)+'\n')
        f_log.write('hit_ratio: '+ str(hit_ratio)+ '\n')
        f_log.write('MAP: '+ str(MAP)+'\n')

    return accuracy, precision, recall, ndcg, hit_ratio, MAP



# reorder the orign file
def reorder_the_file(f_name, reorder_file):

    user_and_index = generate_user_index(f_name)

    readfile = open(f_name, 'r', encoding='utf-8')
    f_all = readfile.readlines()

    with open(reorder_file, 'w', encoding='utf-8') as f:
        for k,v in user_and_index.items():
            user_and_index[k].sort(reverse=True)
            for i in user_and_index[k]:
                f.write(f_all[i])

def remove_unimportant_checkins(min_poi_checked_num, min_user_posted_num, checkin_file):

    unimportant_user = set()
    unimportant_poi  = set()
    user_records = generate_user_records(checkin_file)
    poi_records  = generate_poi_records(checkin_file)

    for user,pois in user_records.items():
        if len(user_records[user]) < min_user_posted_num:
            unimportant_user.add(user)

    for poi,users in poi_records.items():
        if len(poi_records[poi]) < min_poi_checked_num:
            unimportant_poi.add(poi)

    # rewrite
    # readfile =  open(checkin_file, 'r', encoding='utf-8')
    # fline = readfile.readlines()

    with open(checkin_file, 'r', encoding='utf-8') as f1:
        with open('after_removed_all_checkin_file.txt', 'w', encoding='utf-8') as f2:
            for line in f1:
                temp_line = line.strip(' \n').split('\t')
                user_id = temp_line[0]
                poi_id = temp_line[2]
                if (user_id in unimportant_user) or (poi_id in unimportant_poi):
                    continue
                else:
                    f2.write(line)

def generate_test_file(orign_file, train_file, test_file, threshold=10 ,percentage=0.8):

    user_index = generate_user_index(orign_file)

    test = []
    train = []
    train_percentage = 10 * percentage  # eg:8
    train_percentage = int(train_percentage)
    readfile =  open(orign_file, 'r', encoding='utf-8')
    f = readfile.readlines()

    for k,v in user_index.items():
        if len(v) >= threshold:

            remainder = len(v) % 10
            partion = len(v) // 10
            train_num = partion *  train_percentage + remainder
            test_num = len(v) - train_num

            for i in range(train_num):
                train.append(f[v[i]])
            for i in range(train_num, len(v)):
                test.append(f[v[i]])
        else:
            for i in range(len(v)):
                train.append(f[v[i]])
    # return train, test

    with open(train_file, 'w', encoding='utf-8') as f2:
        for s in train:
            f2.write(s)
    with open(test_file, 'w', encoding='utf-8') as f3:
        for s in test:
            f3.write(s)

def generate_poi_records(f_name):

    poi_records = {}
    with open(f_name, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n').split('\t')
            user_id = line[0]
            poi_id = line[2]
            if poi_id not in poi_records.keys():

                poi_records[poi_id] = list()
                poi_records[poi_id].append(user_id)
            else:
                poi_records[poi_id].append(user_id)
    return poi_records


def remove_unimportant_users_from_friendship_file(train_checkin_file, friendship_file):
    user_set = set()
    with open(train_checkin_file, 'r', encoding='utf-8') as f:
        for line in f:
            userid = line.split('\t')[0]
            if userid not in user_set:
                user_set.add(userid)

    with open(friendship_file, 'r', encoding='utf-8') as f1:
        with open('after_removed_'+friendship_file, 'w', encoding='utf-8') as f2:
            for line in f1:
                line = line.strip(' \n').split('\t')
                user1 = line[0]
                user2 = line[1]
                if (user1 in user_set) & (user2 in user_set):
                    f2.write(user1+','+user2+'\n')




def str_to_timestamp(time_string):
    temp1 = time.strptime(time_string, "%Y-%m-%dT%H:%M:%SZ")
    timestamp = time.mktime(temp1)
    return timestamp

def generate_poi_node(line, isweek, hour_interval):
    node = line.split('\t')[2]
    time = line.split('\t')[1]
    node_with_tag = node + "-" + str(get_tag(time, isweek, hour_interval))
    return node_with_tag

def gen_user_stamp_poi_with_tag_triple(fname, isweek=False, hour_interval=4):
    user_stamp_poi_triple = []
    with open(fname, 'r', encoding='utf-8') as f:
        for line in f:
            userid = line.strip(' \n').split('\t')[0]
            time_string = line.strip(' \n').split('\t')[1]
            timestamp = str_to_timestamp(time_string)
            poi_with_tag = generate_poi_node(line, isweek, hour_interval)
            user_stamp_poi_triple.append((userid, timestamp, poi_with_tag))
    return user_stamp_poi_triple

def gen_user_stamp_poi_without_tag_triple(fname):
    user_stamp_poi_triple = []
    with open(fname, 'r', encoding='utf-8') as f:
        for line in f:
            userid = line.strip(' \n').split('\t')[0]
            time_string = line.strip(' \n').split('\t')[1]
            timestamp = str_to_timestamp(time_string)
            poiid = line.strip(' \n').split('\t')[2]
            user_stamp_poi_triple.append((userid, timestamp, poiid))
    return user_stamp_poi_triple

def gen_user_next_pois_with_tag_in_given_ineravl(triple_array, delt_t = 24):
    test_records = []
    threshold_delt_t = delt_t * 60 * 60
    for index,value in enumerate(triple_array):
        candidate_pois_of_cur_user = []
        for next_index, next_value in enumerate(triple_array[index+1:]):

            if value[0] == next_value[0]:
                if next_value[1] - value[1] <= threshold_delt_t:
                    candidate_pois_of_cur_user.append(next_value[2])
                if next_value[1] - value[1] > threshold_delt_t:
                    break
            else:
                break
        test_records.append((value[0],value[2],candidate_pois_of_cur_user))
        for index,record in enumerate(test_records):
            if len(record[2]) == 0:
                del test_records[index]
    return test_records


def gen_user_next_pois_in_given_ineravl(triple_array, delt_t = 24):
    test_records = []
    threshold_delt_t = delt_t * 60 * 60
    for index,value in enumerate(triple_array):
        candidate_pois_of_cur_user = []
        for next_index, next_value in enumerate(triple_array[index+1:]):
            if value[0] == next_value[0]:
                if next_value[1] - value[1] <= threshold_delt_t:
                    candidate_pois_of_cur_user.append(next_value[2])
                if next_value[1] - value[1] > threshold_delt_t:
                    break
            else:
                break
        test_records.append((value[0],value[2],candidate_pois_of_cur_user))
        for index,record in enumerate(test_records):
            if len(record[2]) == 0:
                del test_records[index]
    return test_records

def form_dict(f_name):
    feature = set()
    # print('We are loading data from:', f_name)
    with open(f_name, 'r', encoding='utf-8') as f:
        for line in f:
            words_for_line = get_words(line)
            for word in words_for_line:
                feature.add(word)
    # print(words)
    feature = list(feature)
    print("feature number ",len(feature)) # 110
    feature.sort()
    # print(feature)
    return feature

def get_words(line):

    words = set()
    line = line.split('\t')
    info = line[-1]
    word =info.strip("}\n{,").split(',')
    for w in word:
        words.add(w)
    return words

def generate_user_records(checkin_file):

    user_records = {}
    with open(checkin_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n').split('\t')
            user_id = line[0]
            poi_id = line[2]
            if user_id not in user_records.keys():

                user_records[user_id] = list()
                user_records[user_id].append(poi_id)
            else:
                user_records[user_id].append(poi_id)
    return user_records  # without tag

def maxmin_norm(x):

    _range = np.max(x) - np.min(x)
    true_value = (x - np.min(x)) / _range
    round_value = np.round(true_value, decimals=4)
    return round_value

def generate_node_with_tag_feature(node_and_feature, node_type, user_records):

    for k,v in node_and_feature.items():
        len_feature = len(node_and_feature[k])
        break
    node_with_tag_and_feature = {}
    for poi_with_tag in node_type.keys():
        if node_type[poi_with_tag] == 'P':
            poi_without_tag = poi_with_tag.split('-')[0]
            node_with_tag_and_feature[poi_with_tag] = node_and_feature[poi_without_tag]
        if node_type[poi_with_tag] == 'U':
            u_feature = np.array([0 for i in range(len_feature)] )
            records = user_records[poi_with_tag]
            for record in records:
                u_feature = u_feature + np.array(node_and_feature[record])
            u_feature = maxmin_norm(u_feature)
            node_with_tag_and_feature[poi_with_tag] = list(u_feature)
    return node_with_tag_and_feature

def write_test_records_to_file(test_records, fname):
    with open(fname,'w', encoding='utf-8') as f:
        for item in test_records:
            len_candidate_pois = len(item[2])
            poi_str = ''
            for index,poi in enumerate(item[2]):
                if index == len_candidate_pois -1:
                    poi_str += str(poi)
                else:
                    poi_str += str(poi) + '\t'
            f.write(str(item[0]) + '\t' + str(item[1]) + '\t' + poi_str + '\n' )


def indexing(node_type):
    poi_list = []
    user_list = []

    for node,type_ in node_type.items():
        if type_ == 'P':
            poi_list.append(node)
        if type_ == 'U':
            user_list.append(node)
	# POI Dictionary
    poi2id = {"<PAD>":0}
    id2poi = ["<PAD>"]
    for poi in poi_list:
        if poi not in poi2id:

            poi2id[poi] = len(poi2id)
            # id2poi = ["<PAD>",'0', '1', .......]
            id2poi.append(poi)

	# Word Dictionary
    user2id = {"<PAD>":0}
    id2user = ["<PAD>"]
    for user in user_list:
        if user not in user2id:
            user2id[user] = len(user2id)
            id2user.append(user)

    print("user nodes plus poi_tag nodes equals : " ,len(id2poi)+len(id2user)-2)

    return poi2id, id2poi, user2id, id2user #

def generate_user_index(orign_file):

    # user_and_records_num = generate_user_and_num_records(orign_file)
    readfile =  open(orign_file, 'r', encoding='utf-8')
    user_index = { }
    f = readfile.readlines()
    for i in range(len(f)):
        user_id = f[i].split('\t')[0]
        if user_id not in user_index.keys():
            user_index[user_id] = []
            user_index[user_id].append(i)
        else:
            user_index[user_id].append(i)
    return user_index

def generate_valid_file(orign_file, train_file, test_file, threshold=2 ,percentage=0.5):

    user_index = generate_user_index(orign_file)

    test = []
    train = []
    train_percentage = 10 * percentage  # eg:8
    train_percentage = int(train_percentage)
    readfile =  open(orign_file, 'r', encoding='utf-8')
    f = readfile.readlines()

    for k,v in user_index.items():
        if len(v) >= threshold:

            train_num = int(len(v) * percentage)
            test_num = len(v) - train_num

            for i in range(train_num):
                train.append(f[v[i]])
            for i in range(train_num, len(v)):
                test.append(f[v[i]])
        else:
            for i in range(len(v)):
                train.append(f[v[i]])
    # return train, test

    with open(train_file, 'w', encoding='utf-8') as f2:
        for s in train:
            f2.write(s)
    with open(test_file, 'w', encoding='utf-8') as f3:
        for s in test:
            f3.write(s)

def indexing_user_stamp_poi_triple(user_stamp_poi_triple, poi2id, user2id):

    num_users = len(user2id) - 1
    indexing_user_stamp_poi_triple = []
    for item in user_stamp_poi_triple:
        userid, stamp, poi_with_tag = item
        try:
            triple_array = (user2id[userid], stamp, poi2id[poi_with_tag] + num_users)
            indexing_user_stamp_poi_triple.append(triple_array)
        except:
            continue
    return indexing_user_stamp_poi_triple

def read_records(fname):
    records = []
    with open(fname, 'r',encoding='utf-8') as f:
        for line in f:
            candidate_pois = []
            line = line.strip(' \n').split('\t')
            num_items = len(line)
            for i in range(2, num_items):
                candidate_pois.append(line[i])
            record = (line[0], line[1], candidate_pois)
            records.append(record)
    return records

def gen_user_next_pois_in_given_ineravl_triple(triple_array, delt_t = 24):
    candidate_pois_of_cur_user = []
    threshold_delt_t = delt_t * 60 * 60
    # future_triple_array = triple_array[position:]
    value = triple_array[0]
    for next_index, next_value in enumerate(triple_array[1:]):
        if next_value[1] - value[1] <= threshold_delt_t:
            candidate_pois_of_cur_user.append(next_value[2])
        if next_value[1] - value[1] > threshold_delt_t:
            break
    return candidate_pois_of_cur_user

def gen_LSTM_test_records(train_user_records, test_user_records, test_checkin_file, delt_t):
    user_stamp_poi_without_tag_triple = gen_user_stamp_poi_without_tag_triple(test_checkin_file)
    process_triple = {} # key = user , value = [triple, triple]
    for item in user_stamp_poi_without_tag_triple:
        if item[0] not in process_triple.keys():
            process_triple[item[0]] = []
            process_triple[item[0]].append(item)
        else:
            process_triple[item[0]].append(item)
    LSTM_test_records = []
    for user,poi_list in test_user_records.items():
        cur_user_train_poi_history = train_user_records[user]

        for i in range(len(poi_list)-1):
            cur_user_LSTM_input_pois = cur_user_train_poi_history + poi_list[:i+1]
            cur_user_LSTM_output_pois = gen_user_next_pois_in_given_ineravl_triple(process_triple[user][i:], delt_t = delt_t)
            if len(cur_user_LSTM_output_pois)!=0:
                LSTM_test_records.append((user, cur_user_LSTM_input_pois, cur_user_LSTM_output_pois))

    return LSTM_test_records

def gen_LSTM_test_records_next(train_user_records, test_user_records, test_checkin_file, delt_t):

    user_stamp_poi_without_tag_triple = gen_user_stamp_poi_without_tag_triple(test_checkin_file)
    process_triple = {} # key = user , value = [triple, triple]
    for item in user_stamp_poi_without_tag_triple:
        if item[0] not in process_triple.keys():
            process_triple[item[0]] = []
            process_triple[item[0]].append(item)
        else:
            process_triple[item[0]].append(item)


    LSTM_test_records = []
    for user,poi_list in test_user_records.items():
        cur_user_train_poi_history = train_user_records[user]
        for i in range(len(poi_list)-1):
            cur_user_LSTM_input_pois = cur_user_train_poi_history + poi_list[:i+1]
            cur_user_LSTM_output_pois = gen_user_next_pois_in_given_ineravl_triple(process_triple[user][i:], delt_t = delt_t)
            if len(cur_user_LSTM_output_pois)!=0:
                LSTM_test_records.append((user, cur_user_LSTM_input_pois, cur_user_LSTM_output_pois))

    LSTM_test_records_next = []
    for u, LSTM_input_pois, LSTM_output_pois in LSTM_test_records:
        LSTM_test_records_next.append((u, LSTM_input_pois, [LSTM_output_pois[0]]))
    return LSTM_test_records_next

def change_gowalla_checkin_col(fname):
    with open(fname, 'r', encoding='utf-8') as fin:
        with open('change_cols_'+fname, 'w', encoding='utf-8') as fout:
            temp_str = ''
            for line in fin:
                line = line.strip(' \n').split('\t')
                poi_id = line[4]
                poi_lat = line[2]
                temp_str = poi_id
                line[4] = poi_lat
                line[2] = temp_str
                fout.write(line[0] + '\t' + line[1] + '\t' + line[2] + '\t' + line[3] + '\t' + line[4] + '\n')

if __name__ == "__main__":
    pass
