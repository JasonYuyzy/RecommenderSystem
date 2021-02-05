import argparse
import torch
import math
import random
import progressbar
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.utils.data as Data

from model import Autorec

p1 = progressbar.ProgressBar()


def rating_dataloader():
    print("Collecting the reviews data...")
    reviews = pd.read_csv("./data_file/review_bars.csv")[['review_id', 'user_id', 'business_id', 'stars', 'date']]
    reviews = reviews[reviews['date'].apply(lambda x: int(x.split('-')[0]) > 2018)]
    business = dict()
    users_train = dict()
    users_test = dict()
    user_train_set = set()
    user_test_set = set()
    item_train_set = set()
    item_test_set = set()
    for bus_id in set(reviews['business_id']):
        #if len(business) >= 4000:
            #continue
        if bus_id not in business:
            business[bus_id] = len(business)

    for user_id in set(reviews['user_id']):
        if user_id not in users_train and random.random() < 0.8:
            #if len(users_train) >= 24000:
                #print(user_id)
                #exit()
            users_train[user_id] = len(users_train)
        elif user_id not in users_test:
            #if len(users_test) >= 6000:
                #continue
            users_test[user_id] = len(users_test)

    num_items = len(business)
    test_num_users = len(users_test)
    train_num_users = len(users_train)

    print("business number:", num_items)
    print("train number:", train_num_users)
    print("test number:", test_num_users)

    train_r = np.zeros((train_num_users, num_items))
    test_r = np.zeros((test_num_users, num_items))

    train_mask_r = np.zeros((train_num_users, num_items))
    test_mask_r = np.zeros((test_num_users, num_items))

    reid_barid = dict(zip(reviews['review_id'], reviews['business_id']))
    reid_barar = dict(zip(reviews['review_id'], reviews['stars']))
    reid_userid = dict(zip(reviews['review_id'], reviews['user_id']))
    del reviews
    # training set
    p1.start(len(reid_barid))
    count = 0
    total = 0
    for re_id in reid_barid:
        b_id = reid_barid[re_id]
        u_id = reid_userid[re_id]
        star = int(reid_barar[re_id])
        count += 1
        if u_id in users_train and b_id in business:
            train_r[users_train[u_id]][business[b_id]] = star
            train_mask_r[users_train[u_id]][business[b_id]] = 1
            user_train_set.add(users_train[u_id])
            item_train_set.add(business[b_id])
            total += 1

        elif u_id in users_test and b_id in business:
            test_r[users_test[u_id]][business[b_id]] = star
            test_mask_r[users_test[u_id]][business[b_id]] = 1
            user_test_set.add(users_test[u_id])
            item_test_set.add(business[b_id])
            total += 1

        p1.update(count)

    num_users = len(user_train_set) + len(user_test_set)
    num_items = len(business)
    print("number of users:", num_users)
    print("number of bar:", num_items)
    print("total of rating:", total)
    dic_new = dict(zip(business.values(), business.keys()))
    del reid_barid
    del reid_barar
    del reid_userid
    del business
    #del users_train
    #del users_test
    exit()
    return users_train, users_test, dic_new, num_users, num_items, total, train_r, train_mask_r, test_r, test_mask_r, user_train_set, item_train_set, user_test_set, item_test_set


#DL training recommender
parser = argparse.ArgumentParser(description='I-AutoRec ')
parser.add_argument('--hidden_units', type=int, default=800)  # hidden unit
parser.add_argument('--lambda_value', type=float, default=1)

parser.add_argument('--train_epoch', type=int, default=100)  # training epoch
parser.add_argument('--batch_size', type=int, default=1500)  # batch_size

parser.add_argument('--optimizer_method', choices=['Adam', 'RMSProp'], default='Adam')
parser.add_argument('--grad_clip', type=bool, default=False)
parser.add_argument('--base_lr', type=float, default=2e-5)  # learning rate
parser.add_argument('--decay_epoch_step', type=int, default=10, help="decay the learning rate for each n epochs")

parser.add_argument('--random_seed', type=int, default=1000)
parser.add_argument('--display_step', type=int, default=1)

args = parser.parse_args()
np.random.seed(args.random_seed)
users_train, users_test, dict_new, num_users, num_items, num_total_ratings, train_r, train_mask_r, test_r, test_mask_r, user_train_set, item_train_set, user_test_set, item_test_set = rating_dataloader()
rec = Autorec(args, num_users, num_items)

if torch.cuda.is_available():
    args.cuda = True
    rec.cuda()
else:
    args.cuda = False

optimer = optim.Adam(rec.parameters(), lr=args.base_lr, weight_decay=1e-4)

num_batch = int(math.ceil(num_users / args.batch_size))

torch_dataset = Data.TensorDataset(torch.from_numpy(train_r), torch.from_numpy(train_mask_r), torch.from_numpy(train_r))
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=args.batch_size,
    shuffle=True
)


def train(epoch):
    RMSE = 0
    cost_all = 0
    for step, (batch_x, batch_mask_x, batch_y) in enumerate(loader):
        if args.cuda == True:
            batch_x = batch_x.type(torch.FloatTensor).cuda()
            batch_mask_x = batch_mask_x.type(torch.FloatTensor).cuda()
        else:
            batch_x = batch_x.type(torch.FloatTensor)
            batch_mask_x = batch_mask_x.type(torch.FloatTensor)

        decoder = rec(batch_x)
        loss, rmse = rec.loss(decoder=decoder, input=batch_x, optimizer=optimer, mask_input=batch_mask_x)
        optimer.zero_grad()
        loss.backward()
        optimer.step()
        cost_all += loss
        RMSE += rmse

    RMSE = np.sqrt(RMSE.detach().cpu().numpy() / (train_mask_r == 1).sum())
    print('epoch ', epoch, ' train RMSE : ', RMSE)

def test(epoch):
    if args.cuda == True:
        test_r_tensor = torch.from_numpy(test_r).type(torch.FloatTensor).cuda()
        test_mask_r_tensor = torch.from_numpy(test_mask_r).type(torch.FloatTensor).cuda()
    else:
        test_r_tensor = torch.from_numpy(test_r).type(torch.FloatTensor)
        test_mask_r_tensor = torch.from_numpy(test_mask_r).type(torch.FloatTensor)

    decoder = rec(test_r_tensor)
    if args.cuda == True:
        decoder = torch.from_numpy(np.clip(decoder.detach().cpu().numpy(),a_min=1,a_max=5)).cuda()

    unseen_user_test_list = list(user_test_set - user_train_set)  # the user list not in training list
    unseen_item_test_list = list(item_test_set - item_train_set)  # the bar list not in training list

    for user in unseen_user_test_list:
        for item in unseen_item_test_list:
            if test_mask_r[user, item] == 1:  # if recorded decoder[user,item]=3
                decoder[user, item] = 3

    mse = ((decoder - test_r_tensor) * test_mask_r_tensor).pow(2).sum()
    RMSE = mse.detach().cpu().numpy() / (test_mask_r == 1).sum()
    RMSE = np.sqrt(RMSE)

    print('epoch ', epoch, ' test RMSE : ', RMSE)

    if epoch == args.train_epoch:
        rec.saveModel('./AutoRec.model')

def train_model(is_train, u_id):
    if is_train:
        for epoch in range(args.train_epoch):
            train(epoch=epoch)
            test(epoch=epoch)
        print("Training finished")
        return True
    else:
        new_p_rate = dict()
        rec = Autorec(args, num_users, num_items)
        rec.loadModel('./AutoRec.model', map_location=torch.device('cpu'))
        if u_id in users_train:
            num_p_rate_dict = rec.recommend_user(train_r[users_train[u_id]])
        else:
            num_p_rate_dict = rec.recommend_user(test_r[users_test[u_id]])
        for num_id in num_p_rate_dict:
            new_p_rate[dict_new[num_id]] = num_p_rate_dict[num_id]
        del num_p_rate_dict
        return new_p_rate