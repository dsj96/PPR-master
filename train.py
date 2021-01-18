
import sys
import os

import torch
import torch.nn as nn
import numpy as np
# import torchvision
from torch.utils.data import DataLoader
from datetime import datetime
import random
import argparse

from utils import *

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', type=str, default='dataset/toyset/',
                        help='Input dataset path')

    parser.add_argument('--input_size', type=int, default=16,
                        help='Dimention of the poi/user')

    parser.add_argument('--hidden_size', type=int, default=16,
                        help='Set the output_size of LSTM')

    parser.add_argument('--layers', type=int, default=2,
                        help='Set the layers of LSTM')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='Set the learning rate')

    parser.add_argument('--delt_t', type=float, default=6.0,
                        help='Set the delt_t')

    parser.add_argument('--epochs', type=int, default=20,
                        help='Set the epochs')

    parser.add_argument('--dr', type=float, default=0.2,
                        help='Set the drop rate')

    parser.add_argument('--seed', type=int, default=1,
                        help='Set the random seed')

    parser.add_argument('--test_sample_num', type=int, default=100,
                        help='Set the number of test records')

    return parser.parse_args()

def log(fname, s):
    f = open(fname, 'a')
    f.write(str(datetime.now()) + ': ' + s + '\n')
    f.close()

def switch_list_to_tensor():
    print(PATH)
    start_time = time.time()
    print('Loading configuration!')
    history = read_history(PATH+'train_checkin_file.txt')
    final_model2 = read_vec(PATH+'vec_2nd_wo_norm.txt')
    print('fnished read!')
    final_model2 = normalize_dict(final_model2)
    print('finished normalized!')
    # final_model= connecate(final_model1,final_model2)
    for k,v in final_model2.items():
        final_model2[k] = torch.Tensor(v)
    return final_model2 # final_model2['124'].shape[0]= 16

def switch_tensor_to_array(final_model2):
    for k,v in final_model2.items():
        final_model2[k] = v.numpy()
    return final_model2



def gen_train_data(final_model2):

    train_user_records = generate_user_records(PATH+'train_checkin_file.txt')
    less_three_records_user_list = []
    for k in train_user_records.keys():
        if len(train_user_records[k]) < 3:
            less_three_records_user_list.append(k)
    for k in less_three_records_user_list:
        train_user_records.pop(k)


    LSTM_train_records_output = []
    LSTM_train_records_input = []
    index = 0
    for userid,poi_list in train_user_records.items():
        userid_tensor = final_model2[userid]
        LSTM_train_records_input.append([])
        LSTM_train_records_output.append([])
        for poi in poi_list:
            LSTM_train_records_input[index].append(torch.cat((userid_tensor, final_model2[poi]), 0))
            LSTM_train_records_output[index].append(final_model2[poi])
        index = index + 1
    for index,item in enumerate(LSTM_train_records_input):
        LSTM_train_records_input[index] = item[:-1]
    for index,item in enumerate(LSTM_train_records_output):
        LSTM_train_records_output[index] = item[1:]

    LSTM_train_records_input_ = []
    LSTM_train_records_output_ = []
    for index,tensor_list in enumerate(LSTM_train_records_input):
        tensor = tensor_list[0]
        for item in tensor_list[1:]:
            tensor = torch.cat((tensor, item), 0)
        tensor = tensor.view(len(tensor_list), -1)
        LSTM_train_records_input_.append(tensor)
    for index,tensor_list in enumerate(LSTM_train_records_output):
        tensor = tensor_list[0]
        for item in tensor_list[1:]:
            tensor = torch.cat((tensor, item), 0)
        tensor = tensor.view(len(tensor_list), -1)
        LSTM_train_records_output_.append(tensor)
    print('gen_train_data')
    return LSTM_train_records_input_, LSTM_train_records_output_

def gen_test_data(final_model2):
    train_user_records = generate_user_records(PATH+'train_checkin_file.txt')
    test_user_records = generate_user_records(PATH+'test_checkin_file.txt')
    LSTM_test_records = gen_LSTM_test_records(train_user_records, test_user_records, PATH+'test_checkin_file.txt',delt_t=DELT_T)
    random.seed(SEED)
    LSTM_test_records = random.sample(LSTM_test_records, TEST_SAMPLE_NUM)
    LSTM_test_records_input = []
    LSTM_test_records_output = []
    LSTM_test_user_list = []
    for userid,input_poi_list,target_poi_list in LSTM_test_records:
        userid_tensor = final_model2[userid]
        input_ = []
        try:
            for poi in input_poi_list:
                input_.append(torch.cat((userid_tensor, final_model2[poi]), 0))
            LSTM_test_records_input.append(input_)
            LSTM_test_records_output.append(target_poi_list)
            LSTM_test_user_list.append(userid)
        except:
            continue
    LSTM_test_records_input_ = []

    for index,tensor_list in enumerate(LSTM_test_records_input):
        tensor = tensor_list[0]
        for item in tensor_list[1:]:
            tensor = torch.cat((tensor, item), 0)
        tensor = tensor.view(len(tensor_list), -1)
        LSTM_test_records_input_.append(tensor)
    print('gen_test_data')
    return LSTM_test_records_input_, LSTM_test_records_output, LSTM_test_user_list


class lstm(nn.Module):
    def __init__(self):
        super(lstm, self).__init__()
        self.rnn = nn.LSTM(
            input_size = INPUT_SIZE,
            hidden_size = HIDDEN_SIZE,
            num_layers = LAYERS,
            dropout = DROP_RATE,
            batch_first = False
        )
        self.hidden_out = nn.Linear(HIDDEN_SIZE, int(INPUT_SIZE/2))
        self.h_s = None
        self.h_c = None
    def forward(self, x):
        r_out, (h_s, h_c) = self.rnn(x)
        output = self.hidden_out(r_out)
        return output

if __name__ == "__main__":
    args = parse_args()
    PATH = args.input_path
    INPUT_SIZE = args.input_size * 2
    HIDDEN_SIZE = args.hidden_size
    DELT_T = args.delt_t
    TEST_SAMPLE_NUM = args.test_sample_num
    LAYERS = args.layers
    DROP_RATE = args.dr
    LR = args.lr
    EPOCHS = args.epochs
    SEED = args.seed
    random.seed(SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    final_model2 = switch_list_to_tensor()
    LSTM_train_records_input, LSTM_train_records_output = gen_train_data(final_model2)
    train = zip(LSTM_train_records_input, LSTM_train_records_output) # iterator
    train_ = []
    for pairs in train:
        train_.append((pairs[0].view(-1,1,INPUT_SIZE), pairs[1].view(-1,1,int(INPUT_SIZE/2))))

    LSTM_test_records_input, LSTM_test_target_poi_list, LSTM_test_user_list = gen_test_data(final_model2)

    rnn = lstm().to(device)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
    loss_func = nn.MSELoss()
    mult_step_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                milestones=[EPOCHS//2, EPOCHS//4*3], gamma=0.1)

    train_loss = []
    min_valid_loss = np.inf
    print('Training...')
    for i in range(EPOCHS):
        total_train_loss = []
        rnn.train()
        for step, (b_x, b_y) in enumerate(train_):

            b_x = b_x.type(torch.FloatTensor).to(device)
            b_y = b_y.type(torch.FloatTensor).to(device)
            prediction = rnn(b_x)

            loss = loss_func(prediction, b_y)
            optimizer.zero_grad()                   # clear gradients for this training step
            loss.backward()                         # backpropagation, compute gradients
            optimizer.step()                        # apply gradients
            total_train_loss .append(loss.item())
        train_loss.append(np.mean(total_train_loss ))
        random.shuffle(train_)

        log_string = ('iter: [{:d}/{:d}], train_loss: {:0.6f}, lr: {:0.7f}').format((i + 1), EPOCHS, train_loss[-1], optimizer.param_groups[0]['lr'])
        mult_step_scheduler.step()
        print(log_string)
        log('./LSTM.log', log_string)

    rnn = rnn.eval()
    candidate_pois_tensor = []
    for step, b_x in enumerate(LSTM_test_records_input):
        b_x = b_x.view(-1,1,int(INPUT_SIZE))
        b_x = b_x.type(torch.FloatTensor).to(device)
        prediction = rnn(b_x)

        prediction = prediction[-1][-1]
        candidate_pois_tensor.append(prediction)

    norm_candidate_pois_array = []
    for item in candidate_pois_tensor:
        item = item.detach().cpu().numpy()
        item = normalize(item)
        norm_candidate_pois_array.append(item)
    final_model2 = switch_tensor_to_array(final_model2)


    node_type = get_node_type(PATH+'node_type.txt')
    history = read_history(PATH+'train_checkin_file.txt')
    print('evaluate!')

    accuracy, precision, recall, ndcg, hit_ratio, MAP = evaleate_all_index_LSTM_no_history(norm_candidate_pois_array, LSTM_test_target_poi_list, node_type, final_model2)
