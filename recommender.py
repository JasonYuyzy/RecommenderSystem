import os
import numpy as np
import pandas as pd
import json
import math
import progressbar
import random
import time, datetime
from ast import literal_eval
from model import Autorec
import argparse
import torch
import torch.optim as optim
import torch.utils.data as Data

from surprise import SVD, SVDpp
from surprise import accuracy
from surprise import Dataset, Reader
from surprise.model_selection import KFold
from surprise.model_selection import GridSearchCV
from surprise.model_selection import cross_validate, train_test_split

import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

today = datetime.date.today()

p1 = progressbar.ProgressBar()
p2 = progressbar.ProgressBar()
p3 = progressbar.ProgressBar()
p4 = progressbar.ProgressBar()
p5 = progressbar.ProgressBar()
p6 = progressbar.ProgressBar()
p7 = progressbar.ProgressBar()
p8 = progressbar.ProgressBar()
p9 = progressbar.ProgressBar()

user_id = 'QaELAmRcDc5TfJEylaaP8g'

def restaurant_tips_categories_combiner(rest):
    print("Collecting categories...")
    categories_sum = dict()
    with open('./rest_data/business_restaurants.json', mode='r', encoding='utf-8') as f:
        dicts = json.load(f)
        #print("Restaurant number:", len(dicts))
        for i in dicts:
            if i['business_id'] not in rest:
                continue
            if i['business_id'] is None and i['user_id'] is None:
                continue
            text_lst = i['categories'].split(',')
            text_lst = [word.strip() for word in text_lst if random.random() > 0.7]
            text_lst = list(set(text_lst))
            categories_sum[i['business_id']] = text_lst
            rest[i['business_id']] = i['name']
    f.close()
    del dicts
    #with open('./rest_data/tips_summarized.json', mode='r', encoding='utf-8') as f:
        #dicts = json.load(f)
        #for busi_id in dicts:
            #text_lst = dicts[busi_id].split(' ')
            #text_lst = list(set(text_lst))
            #if busi_id not in categories_sum:
                #continue
            #c_lst = categories_sum[busi_id]
            #for word in text_lst:
                #c_lst.append(word)
            #list(set(c_lst))
            #categories_sum.update({busi_id: c_lst + text_lst})
        #f.close()

    len_lst = []
    for id in categories_sum:
        len_lst.append(len(categories_sum[id]))
        categories_sum.update({id: str(categories_sum[id])})

    dataf_description = pd.DataFrame(categories_sum, index=[0]).transpose().reset_index()
    dataf_rest_id = pd.DataFrame(rest, index=[0]).transpose().reset_index()
    del categories_sum
    #del rest_id

    dataf_description.columns = ['business_id', 'description']
    dataf_rest_id.columns = ['business_id', 'name']
    dataf_description['description'] = dataf_description['description'].apply(literal_eval)
    dataf_description['description'] = dataf_description['description'].apply(lambda x: ' '.join(x))

    #print(len(dataf_description['business_id']))
    print("Categories collecting done")
    # print(dataf_description)

    return dataf_description, dataf_rest_id

def csv_restaurant_categories_collect():
    print("Start collecting the restaurants' information...")
    restaurant = pd.read_csv('./data_file/restaurant.csv')[['business_id', 'name', 'categories','is_open', 'review_count']]
    restaurant['business_id'].dropna(axis=0)
    restaurant['name'].dropna(axis=0)
    restaurant['categories'].dropna(axis=0)
    restaurant['is_open'].dropna(axis=0)
    restaurant = restaurant[~restaurant['is_open'].isin([0])]
    restaurant = restaurant[restaurant['review_count'].apply(lambda x: int(x) > 400)]
    categories_dic = dict(zip(restaurant['business_id'], restaurant['categories']))
    categories_description = dict()
    p8.start(len(restaurant))
    count = 0
    for rest_id in set(restaurant['business_id']):
        text = categories_dic[rest_id]
        text_lst = ''.join(text.split(','))
        text_lst = text_lst.split(' ')
        text_lst = [word.strip() for word in text_lst if word != "&"]
        text_lst = list(set(text_lst))
        if "Food" in text_lst:
            text_lst.remove("Food")
        if "Restaurants" in text_lst:
            text_lst.remove("Restaurants")
        categories_description[rest_id] = str(text_lst)
        count += 1
        p8.update(count)

    name = restaurant.drop(['categories','is_open', 'review_count'], axis=1)
    del text_lst
    del text
    del restaurant


    rest_categories = pd.DataFrame(categories_description, index=[0]).transpose().reset_index()
    rest_categories.columns = ['business_id', 'categories']
    #rest_categories['categories'] = rest_categories['categories'].apply(literal_eval)
    #rest_categories['categories'] = rest_categories['categories'].apply(lambda x: ' '.join(x))
    restaurant = pd.merge(rest_categories, name, how='left', on='business_id')
    #print(rest_categories)

    del categories_dic
    del categories_description
    print("Information collected")
    return restaurant

def csv_tips_cocategories_combination(rest_dict):
    print("Collecting the tips info...")
    stemmer = SnowballStemmer('english')
    stop = set(stopwords.words('english'))
    tips_df = pd.read_csv('./data_file/tips.csv')
    tips_df['text'].dropna(axis=0)
    categories_dict = dict(zip(rest_dict['business_id'], rest_dict['categories']))
        #print(type(eval(categories_dict[i])))
    del rest_dict
    tips_df = tips_df[tips_df['business_id'].apply(lambda x: x in categories_dict)]
    user_rest = dict(zip(tips_df['user_id'], tips_df['business_id']))
    user_tips = dict(zip(tips_df['user_id'], tips_df['text']))
    del tips_df
    rest_tips = dict()
    p9.start(len(user_rest))
    count = 0
    for user_id in user_rest:
        rest_id = user_rest[user_id]
        tips = user_tips[user_id]
        text_lst = nltk.regexp_tokenize(tips, pattern=r'\w+|\S\w')
        text_lst = list(set(text_lst))
        text_lst = [stemmer.stem(w) for w in text_lst if len(w.strip()) > 3]
        text_lst = [w for w in text_lst if w.strip() not in stop]
        text_lst = eval(categories_dict[rest_id]) + text_lst
        text_lst = list(set(text_lst))
        if "Food" in text_lst:
            text_lst.remove("Food")
        if rest_id not in rest_tips:
            rest_tips[rest_id] = text_lst
            count += 1
            continue
        if len(rest_tips[rest_id]) > 70:
            count += 1
            continue

        rest_tips[rest_id] += text_lst
        count += 1
        p9.update(count)

    for i in rest_tips:
        rest_tips[i] = str(rest_tips[i])

    del user_rest
    del user_tips
    del stop
    rest_tips_df = pd.DataFrame(rest_tips, index=[0]).transpose().reset_index()
    del rest_tips
    rest_tips_df.columns = ['business_id', 'description']
    rest_tips_df['description'] = rest_tips_df['description'].apply(literal_eval)
    rest_tips_df['description'] = rest_tips_df['description'].apply(lambda x: ' '.join(x))
    return rest_tips_df

def csv_covid_info_collect(restaurant_dcit):
    print("Collecting the covid information...")
    covid = pd.read_csv('./data_file/covid_19.csv')
    rest_covid_info = covid[covid.apply(covid_filtering, axis=1, args=(today.year, today.month))]
    rest_covid_info = pd.merge(restaurant_dcit, rest_covid_info, how='left', on='business_id')
    rest_covid_info.dropna()
    #rest_covid_info['review_id'].dropna(axis=0)
    #rest_covid_info['business_id'].dropna(axis=0)
    #rest_covid_info['stars'].dropna(axis=0)
    print("Covid info collected")
    return rest_covid_info
def covid_filtering(covid_info, year, month):
    date_check = covid_info['Temporary Closed Until'].split('-')
    if len(date_check) == 1:
        return True
    if int(date_check[0]) >= year and int(date_check[1]) >= month:
        return False
    else:
        return True

def csv_restaurant_rating_average(restaurant_dcit):
    print("Start collecting the restaurants' rating average...")
    reviews = pd.read_csv('./data_file/reviews.csv')[['review_id', 'business_id', 'stars', 'date', 'user_id']]
    reviews['date'].dropna(axis=0)
    reviews['review_id'].dropna(axis=0)
    reviews['business_id'].dropna(axis=0)
    reviews['stars'].dropna(axis=0)
    reviews['user_id'].dropna(axis=0)
    #filtering the year
    reviews = reviews[reviews['date'].apply(lambda x: int(x.split('-')[0]) > 2018)]
    reviews = reviews[reviews['business_id'].apply(lambda x: x in restaurant_dcit)]
    idre_idres = dict(zip(reviews['review_id'], reviews['business_id']))
    reid_restar = dict(zip(reviews['review_id'], reviews['stars']))
    reid_userid = dict(zip(reviews['review_id'], reviews['user_id']))
    users_review = dict()
    rating_average = dict()
    rating_count = dict()
    p7.start(len(idre_idres))
    count = 0
    for re_id in idre_idres:
        rest_id = idre_idres[re_id]
        user_id = reid_userid[re_id]
        users_review[user_id] = {}
        if rest_id in restaurant_dcit:
            if rest_id not in rating_average:
                rating_average[rest_id] = reid_restar[re_id]
                rating_count[rest_id] = 1
                users_review[user_id].update({rest_id: reid_restar[re_id]})
                count += 1
                p7.update(count)
                continue
            rating_average.update({rest_id : round((rating_average[rest_id] + reid_restar[re_id])/2, 3)})
            rating_count.update({rest_id : rating_count[rest_id] + 1})
            #print(user_id, users_review[user_id])
            users_review[user_id].update({rest_id: reid_restar[re_id]})
            count += 1
            p7.update(count)

    users_rating = reviews[['user_id', 'business_id', 'stars']]
    del reviews
    del idre_idres
    del reid_restar
    del reid_userid
    ra_count_df = pd.DataFrame(rating_count, index=[0]).transpose().reset_index()
    ra_average_df = pd.DataFrame(rating_average, index=[0]).transpose().reset_index()
    ra_count_df.columns = ['business_id', 'ra_count']
    ra_average_df.columns = ['business_id', 'ra_average']
    rest_rating_average_df = pd.merge(ra_average_df, ra_count_df, how='left', on='business_id')

    m = rest_rating_average_df['ra_count'].quantile(0.95)
    rest_rating_average_df = rest_rating_average_df[rest_rating_average_df['ra_count'] >= m]
    c = rest_rating_average_df['ra_average'].mean()
    rest_rating_average_df['cb_ra_weight'] = rest_rating_average_df.apply(cb_ra_weight, axis=1, args=(m, c))
    rest_rating_average_df = rest_rating_average_df.sort_values('cb_ra_weight', ascending=False)

    del rating_average
    del rating_count
    del ra_count_df
    del ra_average_df
    del restaurant_dcit
    print("Rating average collected")
    return rest_rating_average_df, users_review, users_rating
def cb_ra_weight(restaurant_dcit, m, c):
    v = restaurant_dcit['ra_count']
    r = restaurant_dcit['ra_average']
    return v / (v + m) * r + m / (v + m) * c

def user_rating_filtering(users_review, rest_dict):
    print("Filtering the users rating...")
    users_review = users_review[users_review['business_id'].apply(lambda x: x in rest_dict)]
    #users_review.set_index('user_id', inplace=True)
    print("Finished filtering")
    return users_review

def csv_user_collecting(restaurant_dcit, rest_dict):
    print('Start collecting user rating matrix...')
    user = pd.read_csv('./data_file/users.csv')[['user_id', 'review_count', 'restaurant_review_count']]
    #user = user[user['restaurant_review_count'].apply(lambda x: int(x) >= 100)]
    kf = KFold(n_splits=5, shuffle=True)
    user = user.sort_values('restaurant_review_count', ascending=False)
    user_dict = set(user['user_id'])
    del user
    rest_lst = [i for i in rest_dict]
    final_rating_matrix = [rest_lst]
    p1.start(len(user_dict))
    count = 0
    Head = True
    #user-rest rating matrix
    for user in user_dict:
        if user in restaurant_dcit:
            if Head:
                final_rating_matrix = pd.DataFrame([restaurant_dcit[user]], columns=[key for key in rest_dict])
                Head = False
            final_rating_matrix = final_rating_matrix.append([restaurant_dcit[user]], ignore_index=True)
            '''
            user_rates_lst = [user]
            for rest in rest_dict:
                if rest in restaurant_dcit[user]:
                    user_rates_lst.append(int(restaurant_dcit[user][rest]))
                    continue
                user_rates_lst.append(0)
            final_rating_matrix.append(user_rates_lst)
            '''
        count += 1
        p1.update(count)
    final_rating_matrix.fillna(0, inplace=True)
    final_rating_matrix = final_rating_matrix.values
    #del user_rates_lst

    for train_index, test_index in kf.split(final_rating_matrix):
        print(train_index, test_index)
    del rest_lst
    del restaurant_dcit
    del rest_dict
    #final_rating_matrix = np.array(final_rating_matrix, dtype=object)
    #print(final_rating_matrix)
    return final_rating_matrix

#DL part
def rating_dataloader():
    print("Collecting the reviews data...")
    reviews = pd.read_csv("./data_file/reviews.csv")[['review_id', 'user_id', 'business_id', 'stars', 'date']]
    reviews = reviews[reviews['date'].apply(lambda x: int(x.split('-')[0]) > 2018)]
    business = dict()
    users_train = dict()
    users_test = dict()
    user_train_set = set()
    user_test_set = set()
    item_train_set = set()
    item_test_set = set()
    for bus_id in set(reviews['business_id']):
        if len(business) >= 4000:
            continue
        if bus_id not in business:
            business[bus_id] = len(business)

    for user_id in set(reviews['user_id']):
        if user_id not in users_train and random.random() < 0.8:
            if len(users_train) >= 24000:
                continue
            users_train[user_id] = len(users_train)
        elif user_id not in users_test:
            if len(users_test) >= 6000:
                continue
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

    reid_restid = dict(zip(reviews['review_id'], reviews['business_id']))
    reid_restar = dict(zip(reviews['review_id'], reviews['stars']))
    reid_userid = dict(zip(reviews['review_id'], reviews['user_id']))
    del reviews
    # training set
    p1.start(len(reid_restid))
    count = 0
    total = 0
    for re_id in reid_restid:
        b_id = reid_restid[re_id]
        u_id = reid_userid[re_id]
        star = int(reid_restar[re_id])
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
    print("number of rest:", num_items)
    print("total of rating:", total)
    dic_new = dict(zip(business.values(), business.keys()))
    del reid_restid
    del reid_restar
    del reid_userid
    del business
    del users_train
    del users_test

    return dic_new, num_users, num_items, total, train_r, train_mask_r, test_r, test_mask_r, user_train_set, item_train_set, user_test_set, item_test_set

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
    unseen_item_test_list = list(item_test_set - item_train_set)  # the restaurant list not in training list

    for user in unseen_user_test_list:
        for item in unseen_item_test_list:
            if test_mask_r[user, item] == 1:  # if recorded decoder[user,item]=3
                decoder[user, item] = 3

    mse = ((decoder - test_r_tensor) * test_mask_r_tensor).pow(2).sum()
    RMSE = mse.detach().cpu().numpy() / (test_mask_r == 1).sum()
    RMSE = np.sqrt(RMSE)

    print('epoch ', epoch, ' test RMSE : ', RMSE)


def CF_SVD_rating_prediction(rest_data_df, users_rating_df, user_id):
    print("Start prediction by CF...")
    user_watched = set(users_rating_df[users_rating_df['user_id'].apply(lambda x: x==user_id)]['business_id'])
    reader = Reader(rating_scale=(0, 5))#
    rating_data = Dataset.load_from_df(users_rating_df, reader=reader)

    #First SVD to filter the unaccerry ratings
    cross_validation = KFold(n_splits=3)
    model = SVD(n_factors=100)
    for trainset, testset in cross_validation.split(rating_data):
        model.fit(trainset)
        predictions = model.test(testset)
        accuracy.rmse(predictions, verbose=True)

    rest_dict = set(rest_data_df['business_id'])
    user_rating_predic = dict()
    for rest in rest_dict:
        predict = model.predict(user_id, rest)
        user_rating_predic[rest] = round(predict.est,3)

    user_rates = pd.DataFrame(user_rating_predic, index=[0]).transpose().reset_index()
    user_rates.columns = ['business_id', 'cf_prediction']
    user_rates = user_rates.sort_values('cf_prediction', ascending=False)

    print("Prediction finished")
    return user_rates

def CF_cal_restaurant_recommend(rest_data_df):
    top_rest = rest_data_df.sort_values('cf_prediction', ascending=False)
    recommend_rest = top_rest[['name', 'description', 'cf_prediction']][:10]
    return recommend_rest


def HY_cal_restaurant_recommend(rest_data_df, users_rating_df, user_id):
    print("Start prediction by CF...")
    user_watched = set(users_rating_df[users_rating_df['user_id'].apply(lambda x: x == user_id)]['business_id'])
    rest_dict = set(rest_data_df['business_id'])
    reader = Reader(rating_scale=(0, 5))
    rating_data = Dataset.load_from_df(users_rating_df, reader=reader)

    # First SVD to filter the unaccerry ratings
    cross_validation = KFold(n_splits=3)
    model1 = SVD(n_factors=100)

    for trainset, testset in cross_validation.split(rating_data):
        model1.fit(trainset)
        predictions = model1.test(testset)
        accuracy.rmse(predictions, verbose=True)

    user_rating_pre = dict()
    for rest in rest_dict:
        predict = model1.predict(user_id, rest)
        user_rating_pre[rest] = round(predict.est, 3)

    user_rates = pd.DataFrame(user_rating_pre, index=[0]).transpose().reset_index()
    user_rates.columns = ['business_id', 'cf_prediction']
    user_rates = user_rates.sort_values('cf_prediction', ascending=False)
    users_rating_df = users_rating_df[users_rating_df['business_id'].apply(lambda x: x in set(user_rates['business_id'].head(40)))]
    del model1

    # second SVD filtering to predict better
    reader = Reader(rating_scale=(3, 4.5))  #
    rating_data = Dataset.load_from_df(users_rating_df, reader=reader)

    # trainset, testset = train_test_split(rating_data, test_size=.25)
    cross_validation = KFold(n_splits=3)
    model2 = SVDpp(n_factors=20)

    for trainset, testset in cross_validation.split(rating_data):
        model2.fit(trainset)
        predictions = model2.test(testset)
        accuracy.rmse(predictions, verbose=True)

    #final_df = pd.merge(rest_data_df, user_rates, how='left', on='business_id')
    #TF-IDF
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(rest_data_df['description'])
    cosine_similar = linear_kernel(tfidf_matrix, tfidf_matrix)
    matrix_sim = pd.DataFrame(cosine_similar, columns=rest_data_df['business_id'], index=rest_data_df['business_id'])
    rank = {}
    #recommend prediction
    for rest in rest_dict:
        for bus_id in user_watched:
            score = 0
            predict = model2.predict(user_id, rest)
            p_v = round(predict.est, 3)
            sim = matrix_sim[bus_id][rest]
            if bus_id == rest or sim == 0:
                continue
            else:
                score += (p_v * sim) / abs(sim)
        score = round(score / len(user_watched), 3)
        rank[rest] = score

    del model2
    hybrid_rates = pd.DataFrame(rank, index=[0]).transpose().reset_index()
    del rank
    hybrid_rates.columns = ['business_id', 'hy_rate']
    final_df = pd.merge(rest_data_df, hybrid_rates, how='left', on='business_id')
    top_rest = final_df.sort_values('hy_rate', ascending=False)
    recommend_rest = top_rest[['name', 'description', 'hy_rate']][:10]
    return recommend_rest

def CB_cal_restaurant_recommend(restaurant, visited):
    print("Starting recommend...")
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')

    tfidf_matrix = tf.fit_transform(restaurant['description'])

    cosine_similar = linear_kernel(tfidf_matrix, tfidf_matrix)

    matrix_sim = pd.DataFrame(cosine_similar, columns=restaurant['business_id'], index=restaurant['business_id'])
    rec = set()
    count = 0
    p6.start(len(visited))
    for visited_id in visited:
        count += 1
        if visited_id in matrix_sim.columns:
            num_sim = list(matrix_sim.sort_values(visited_id, ascending=False).index[1:10])
            for i in num_sim:
                if i not in visited:
                    rec.add(i)
        p6.update(count)

    rec_df = pd.DataFrame(list(rec), columns=['business_id'])
    top_rest = rec_df.merge(restaurant, how='left', on='business_id')
    #top_rest = top_rest.sort_values('ra_average', ascending=False)
    top_rest = top_rest.sort_values('cb_ra_weight', ascending=False)
    recommend_rest = top_rest[['name', 'description', 'cb_ra_weight', 'ra_average']][:10]
    return recommend_rest



if __name__ == '__main__':
    is_train = False
    #CB recommender
    #visited = {'xfWdUmrz2ha3rcigyITV0g': 1, 'OETh78qcgDltvHULowwhJg': 1, '4JNXUYY8wbaaDmk3BPzlWw': 1, 'sKA6EOpxvBtCg7Ipuhl1RQ': 1}
    #rest_data_df = csv_restaurant_categories_collect()
    #covid_info_df = csv_covid_info_collect(rest_data_df['business_id'])
    #rest_data_df = pd.merge(covid_info_df, rest_data_df, how='left', on='business_id')
    #combination_df = csv_tips_cocategories_combination(rest_data_df[['business_id', 'categories']])
    #rest_data_df = pd.merge(rest_data_df, combination_df, how='left', on='business_id')
    #rating_average_df, users_review, users_rating = csv_restaurant_rating_average(set(rest_data_df['business_id']))
    #rest_data_df = pd.merge(rating_average_df, rest_data_df, how='left', on='business_id')
    #top_10_restaurant = CB_cal_restaurant_recommend(rest_data_df, visited)
    #print("CB:", top_10_restaurant)

    #DL data testing
    #user_features_df = csv_user_collecting(users_review, set(rest_data_df['business_id']))
    #print(user_features_df)
    #exit()

    #CF recommender
    #user_id = 'yPv39tqbBwsiMj6M7hQjWQ'
    #users_rating_df = user_rating_filtering(users_rating, set(rest_data_df['business_id']))
    #cf_rest_data_df = CF_SVD_rating_prediction(rest_data_df, users_rating_df, user_id)
    #rest_data_df = pd.merge(rest_data_df, cf_rest_data_df, how='left', on='business_id')
    #top_10_restaurant = CF_cal_restaurant_recommend(rest_data_df)
    #print("CF:", top_10_restaurant)

    #Hybrid recommender
    #top_10_restaurant = HY_cal_restaurant_recommend(rest_data_df, users_rating_df, user_id)
    #print("HY:", top_10_restaurant)
    #visited = user_visited(user_id)
    #print(restaurant)

    #DL training recommender
    parser = argparse.ArgumentParser(description='I-AutoRec ')
    # hidden unit
    parser.add_argument('--hidden_units', type=int, default=500)  # hidden unit
    parser.add_argument('--lambda_value', type=float, default=1)

    parser.add_argument('--train_epoch', type=int, default=3)  # training epoch
    parser.add_argument('--batch_size', type=int, default=800)  # batch_size

    parser.add_argument('--optimizer_method', choices=['Adam', 'RMSProp'], default='Adam')
    parser.add_argument('--grad_clip', type=bool, default=False)
    parser.add_argument('--base_lr', type=float, default=5e-4)  # learning rate
    parser.add_argument('--decay_epoch_step', type=int, default=70, help="decay the learning rate for each n epochs")

    parser.add_argument('--random_seed', type=int, default=1000)
    parser.add_argument('--display_step', type=int, default=1)

    args = parser.parse_args()
    np.random.seed(args.random_seed)
    dict_new, num_users, num_items, num_total_ratings, train_r, train_mask_r, test_r, test_mask_r, user_train_set, item_train_set, user_test_set, item_test_set = rating_dataloader()
    rec = Autorec(args, num_users, num_items)

    if torch.cuda.is_available():
        args.cuda = True
        rec.cuda()
    else:
        args.cuda = False

    if is_train:
        optimer = optim.Adam(rec.parameters(), lr=args.base_lr, weight_decay=1e-4)

        num_batch = int(math.ceil(num_users / args.batch_size))

        torch_dataset = Data.TensorDataset(torch.from_numpy(train_r), torch.from_numpy(train_mask_r),
                                           torch.from_numpy(train_r))
        loader = Data.DataLoader(
            dataset=torch_dataset,
            batch_size=args.batch_size,
            shuffle=True
        )

        for epoch in range(args.train_epoch):
            train(epoch=epoch)
            test(epoch=epoch)

        rec.saveModel('./AutoRec.model')
    else:
        rec = Autorec(args, num_users, num_items)
        rec.loadModel('./AutoRec.model', map_location=torch.device('cpu'))
        bus_id, rate_dict = rec.recommend_user(test_r[0], 5)
        for i in bus_id:
            print(dict_new[i])
