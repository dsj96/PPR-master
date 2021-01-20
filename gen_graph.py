import networkx as nx
import numpy as np
import argparse

import os
import sys
import time
import collections

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', type=str, default='dataset/toyset/',
                        help='Input dataset path')

    parser.add_argument('--epsilon', type=float, default=0.5,
                        help='Set the epsilon')

    parser.add_argument('--theta', type=float, default=24,
                        help='Set the theta')

    return parser.parse_args()


def str_to_timestamp(time_string):
    # temp_str = time_string.split('+')
    # year = temp_str[1].lstrip('0 ')
    # Mon Jul 25 02:03:30 2011
    # format_time_string = temp_str[0] + year
    temp1 = time.strptime(time_string, "%Y-%m-%dT%H:%M:%SZ")
    timestamp = time.mktime(temp1)
    return timestamp

def load_training_data(checkin_file, friendship_file, istag, isweek, hour_interval, edge_interval):
    if istag:
        u_p_edges, p_p_edges = generate_edge_with_tag(checkin_file, isweek, hour_interval)
    else:
        u_p_edges, p_p_edges = generate_edge_without_tag(checkin_file, edge_interval)
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


def generate_edge_with_tag(f_name, isweek, hour_interval, edge_interval=24):
    u_p_edges = []
    p_p_edges = []
    readfile =  open(f_name, 'r', encoding='utf-8')
    f = readfile.readlines()
    for i in range(len(f)-1):
        try:
            f1 = f[i]
            f2 = f[i+1]
            user_id = f1.split('\t')[0]

            poi_with_tag = generate_poi_node(f1, isweek, hour_interval)
            u_p_edges.append((user_id, poi_with_tag))
            u_p_edges.append((poi_with_tag, user_id))
            if (judge_time_interval(f1, f2, edge_interval)):
                p_p_edges.append((poi_with_tag, generate_poi_node(f2, isweek, hour_interval)))
        except:
            pass
    u_p_edges.append((f[-1].split('\t')[0], generate_poi_node(f[-1], isweek, hour_interval)))
    u_p_edges.append((generate_poi_node(f[-1], isweek, hour_interval), f[-1].split('\t')[0]))
    return u_p_edges, p_p_edges


def generate_edge_without_tag(f_name, edge_interval= 24):
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


# def read_history(f_name):
#     history = dict()
#     with open(f_name, 'r', encoding= 'utf-8') as f:
#         for line in f:
#             line = line.strip(' \n').split('\t')
#             if line[0] not in history.keys():
#                 history[line[0]] = set()
#             history[line[0]].add(line[2])
#     return history

def read_history(f_name):
    history = dict()
    with open(f_name, 'r', encoding= 'utf-8') as f:
        for line in f:
            line = line.strip(' \n').split('\t')
            if line[0] not in history.keys():
                history[line[0]] = []
            history[line[0]].append(line[2])
    return history

def gen_poi_coordinate(fname_in):
    '''
    param {*}a file in format : userid time poiid longitude lantitude
    return {*} a dict: key=poiid value=poi_coordiinate eg: {poi1: '120.34\t-32.234'}
    '''
    with open(fname_in, 'r', encoding='utf-8') as fin:
        poi_longi_lanti = {}
        for line in fin:
            line = line.strip(' \n').split('\t')
            poiid = line[2]
            lanti = line[4]
            longi = line[3]
            geo_info =  longi + '\t' +lanti
            if poiid not in poi_longi_lanti.keys():
                poi_longi_lanti[poiid] = geo_info
        return poi_longi_lanti

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
        if edge_key not in u_u_edge_dict:
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

def add_geo_type_info(G, poi_longi_lanti, node_type):
    for n,nbrs in G.adjacency():
        G.nodes[n]['type'] = node_type[n]
        if n in poi_longi_lanti.keys():
            longi_lanti = poi_longi_lanti[n].split('\t')
            G.nodes[n]['longi'] = eval(longi_lanti[0])
            G.nodes[n]['lanti'] = eval(longi_lanti[1])
    return G

def calc_euclid_dist(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dist = np.linalg.norm(vec1-vec2)
    return dist


def update_weight_by_geo(G, kappa=-2):
    for n,nbrs in G.adjacency():
        if G.nodes[n]['type'] == 'P':
            out_nbrs_dist = dict()
            out_nbrs_seq_weight = dict()
            # calc all the out-neighbors
            for nbr in nbrs.keys():
                if G.nodes[nbr]['type'] == 'P':
                    dist = calc_euclid_dist([G.nodes[n]['longi'], G.nodes[n]['lanti']], [G.nodes[nbr]['longi'], G.nodes[nbr]['lanti']])
                    out_nbrs_dist[nbr] = dist
                    weight_seq = G[n][nbr]['weight']
                    out_nbrs_seq_weight[nbr] = weight_seq
            if len(out_nbrs_dist) != 0:
                sum_dist_power = 0.0
                for k,v in out_nbrs_dist.items():
                    sum_dist_power += pow(v+1.0, kappa)
                for k,v in out_nbrs_dist.items():
                    out_nbrs_dist[k] = pow(v+1, kappa) / sum_dist_power
                for k,v in out_nbrs_seq_weight.items():
                    G[n][k]['weight'] = v*out_nbrs_dist[k]
    return G


def get_weight_between_users(history, user1, user2, epsilon):
    user1_checkin_dict = collections.Counter(history[user1]) # {key=poi value=checkin times}
    user2_checkin_dict = collections.Counter(history[user2])
    common_pois = set(history[user1]) & set(history[user2])
    sum_common_times = epsilon
    for poi in common_pois:
        sum_common_times += min(user1_checkin_dict[poi], user2_checkin_dict[poi])
    return sum_common_times/(len(common_pois)+1.0)


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

if __name__ == "__main__":
    args = parse_args()
    input_path = args.input_path
    epsilon = args.epsilon
    theta = args.theta

    u_p_edges, p_p_edges, u_u_edges = load_training_data(input_path+'train_checkin_file.txt', input_path+'friendship_file.txt', istag=False, isweek=False, hour_interval=4, edge_interval = theta)
    history = read_history(input_path+'train_checkin_file.txt')
    poi_longi_lanti = gen_poi_coordinate(input_path+'train_checkin_file.txt')

    G = get_G_from_edges(u_p_edges, p_p_edges, u_u_edges, history, epsilon)
    node_type = gen_node_type_without_tag_from_G(G, input_path+'train_checkin_file.txt', input_path+'friendship_file.txt')
    G = add_geo_type_info(G, poi_longi_lanti, node_type)
    G = update_weight_by_geo(G)

    write_dict_to_file(node_type, input_path+'node_type.txt')
    write_G_to_file(G, input_path+'G_edges.txt')
