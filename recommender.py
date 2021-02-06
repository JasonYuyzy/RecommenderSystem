import os
import numpy as np
import pandas as pd
import json
import math
import progressbar
import random
import time, datetime
from ast import literal_eval
from train import train_model
from operator import itemgetter


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
p10 = progressbar.ProgressBar()
p11 = progressbar.ProgressBar()

user_id = 'QaELAmRcDc5TfJEylaaP8g'

def csv_bar_categories_collect():
    print("Start collecting the bars' information...")
    bar = pd.read_csv('./data_file/bars.csv')[['business_id', 'name', 'categories','is_open', 'review_count']]
    bar['business_id'].dropna(axis=0)
    bar['name'].dropna(axis=0)
    bar = bar.fillna(value={'categories': '', 'is_open': 0})
    bar = bar[~bar['is_open'].isin([0])]
    categories_dic = dict(zip(bar['business_id'], bar['categories']))
    categories_description = dict()
    p8.start(len(bar))
    count = 0
    for bar_id in set(bar['business_id']):
        text = categories_dic[bar_id]
        text_lst = ''.join(text.split(','))
        text_lst = text_lst.split(' ')
        text_lst = [word.strip() for word in text_lst if word != "&"]
        text_lst = list(set(text_lst))
        count += 1
        p8.update(count)
        if "Bars" in text_lst:
            text_lst.remove("Bars")
        if "Bar" in text_lst:
            text_lst.remove("Bar")
        categories_description[bar_id] = str(text_lst)

    name = bar.drop(['categories','is_open', 'review_count'], axis=1)
    del text_lst
    del text
    del bar

    bar_categories = pd.DataFrame(categories_description, index=[0]).transpose().reset_index()
    bar_categories.columns = ['business_id', 'categories']
    bar = pd.merge(bar_categories, name, how='left', on='business_id')

    del categories_dic
    del categories_description
    print("Information collected")
    return bar

def csv_tips_cocategories_combination(bar_dict):
    print("Collecting the tips info...")
    stemmer = SnowballStemmer('english')
    stop = set(stopwords.words('english'))
    tips_df = pd.read_csv('./data_file/tip_bars.csv')
    categories_dict = dict(zip(bar_dict['business_id'], bar_dict['categories']))
    del bar_dict
    tips_df = tips_df[tips_df['business_id'].apply(lambda x: x in categories_dict)]
    user_bar = dict(zip(tips_df['user_id'], tips_df['business_id']))
    user_tips = dict(zip(tips_df['user_id'], tips_df['text']))
    del tips_df
    bar_tips = dict()
    p9.start(len(user_bar))
    count = 0
    for user_id in user_bar:
        bar_id = user_bar[user_id]
        tips = user_tips[user_id]
        text_lst = nltk.regexp_tokenize(tips, pattern=r'\w+|\S\w')
        text_lst = list(set(text_lst))
        text_lst = [stemmer.stem(w) for w in text_lst if len(w.strip()) > 3]
        text_lst = [w for w in text_lst if w.strip() not in stop]
        text_lst = eval(categories_dict[bar_id]) + text_lst
        text_lst = list(set(text_lst))
        if "Bars" in text_lst:
            text_lst.remove("Bars")
        if bar_id not in bar_tips:
            bar_tips[bar_id] = text_lst
            count += 1
            continue
        if len(bar_tips[bar_id]) > 70:
            count += 1
            continue

        bar_tips[bar_id] += text_lst
        count += 1
        p9.update(count)

    for i in bar_tips:
        bar_tips[i] = str(bar_tips[i])
    #fill nan with the no-tips bar
    for bus in categories_dict:
        if bus not in bar_tips:
            bar_tips[bus] = str(categories_dict[bus])

    del user_bar
    del user_tips
    del stop
    bar_tips_df = pd.DataFrame(bar_tips, index=[0]).transpose().reset_index()
    del bar_tips
    bar_tips_df.columns = ['business_id', 'description']
    bar_tips_df['description'] = bar_tips_df['description'].apply(literal_eval)
    bar_tips_df['description'] = bar_tips_df['description'].apply(lambda x: ' '.join(x))
    return bar_tips_df

def csv_covid_info_collect(bar_dcit):
    print("Collecting the covid information...")
    covid = pd.read_csv('./data_file/covid_bars.csv')
    bar_set = set(bar_dcit)
    bar_covid_info = covid[covid.apply(covid_filtering_open_date, axis=1, args=(today.year, today.month))]
    bar_covid_info = bar_covid_info[bar_covid_info['business_id'].apply(lambda x: x in bar_set)]
    bar_covid_info = bar_covid_info.replace({True: 1, False: 0})
    bar_covid_info = bar_covid_info.merge(bar_dcit, how='left', on='business_id')
    #del bar_covid_info['Covid Banner']
    del bar_covid_info['Temporary Closed Until']
    del bar_covid_info['Virtual Services Offered']
    print("Covid info collected")
    return bar_covid_info
def covid_filtering_open_date(covid_info, year, month):
    date_check = covid_info['Temporary Closed Until'].split('-')
    if len(date_check) == 1:
        return True
    if int(date_check[0]) >= year and int(date_check[1]) >= month:
        return False
    else:
        return True

def CB_cal_bar_simularity(bar):
    print("Calculating bar similarity with CB...")
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(bar['description'])
    cosine_similar = linear_kernel(tfidf_matrix, tfidf_matrix)
    matrix_sim = pd.DataFrame(cosine_similar, columns=bar['business_id'], index=bar['business_id'])

    print('Similarity CB finished')
    return matrix_sim

def user_rating_account(bar_set):
    print("Start collecting the bars' rating average...")
    reviews = pd.read_csv('./data_file/review_bars.csv')[['review_id', 'business_id', 'stars', 'user_id']]
    reviews['review_id'].dropna(axis=0)
    reviews = reviews[reviews['business_id'].apply(lambda x: x in bar_set)]
    reid_idres = dict(zip(reviews['review_id'], reviews['business_id']))
    reid_barra = dict(zip(reviews['review_id'], reviews['stars']))
    reid_userid = dict(zip(reviews['review_id'], reviews['user_id']))
    users_review = dict()
    for re_id in reid_userid:
        u_id = reid_userid[re_id]
        b_id = reid_idres[re_id]
        star = reid_barra[re_id]
        if u_id not in users_review:
            users_review.setdefault(u_id, {b_id: star})
            continue
        users_review[u_id].update({b_id: star})

    return users_review

def CF_cal_bar_simularity(views_dict):
    print("Calculating bar similarity with CF...")
    bars_sim_matrix = dict()
    bars_popularity = dict()
    for u_id in views_dict:
        for b_id in views_dict[u_id]:
            if b_id not in bars_popularity:
                bars_popularity[b_id] = 1
                continue
            bars_popularity[b_id] += 1
            for b_id2 in views_dict[u_id]:
                if b_id == b_id2:
                    continue
                if b_id not in bars_sim_matrix:
                    bars_sim_matrix[b_id] = {b_id2: 1}
                    continue
                if b_id2 not in bars_sim_matrix[b_id]:
                    bars_sim_matrix[b_id].update({b_id2: 1})
                    continue
                bars_sim_matrix[b_id][b_id2] += 1

    for b1 in bars_sim_matrix:
        for b2 in bars_sim_matrix[b1]:
            bars_sim_matrix[b1][b2] = bars_sim_matrix[b1][b2] / math.sqrt(bars_popularity[b1] * bars_popularity[b2])

    #print(bars_sim_matrix)
    return bars_sim_matrix

def recommender(CB_sim_matrix, CF_sim_matrix, DL_prate_dict, bar_data_df, reviewed, targets):
    print("Start racommending...")
    bar_set = set(bar_data_df['business_id'])
    prate = dict()
    for bar in bar_set:
        if bar not in DL_prate_dict:
            continue
        P_rate = 0
        count = 0
        for re_bar in reviewed:
            if bar == re_bar:
                continue
            try:
                sim_cf = CF_sim_matrix[re_bar][bar]
            except:
                sim_cf = CB_sim_matrix[re_bar][bar]
            cb_sim = CB_sim_matrix[re_bar][bar]
            sim = cb_sim * 0.7 + sim_cf * 0.3
            if sim == 0:
                continue
            dl_rate = DL_prate_dict[bar]
            P_rate += (sim * dl_rate) / abs(sim)
            count += 1
        if count == 0:
            prate[bar] = 0
        else:
            prate[bar] = round((P_rate / count), 3)

    hybrid_rates = pd.DataFrame(prate, index=[0]).transpose().reset_index()
    del prate
    hybrid_rates.columns = ['business_id', 'hy_rate']
    final_df = pd.merge(bar_data_df, hybrid_rates, how='left', on='business_id')
    top_bar = final_df.sort_values('hy_rate', ascending=False)
    recommend_bar = top_bar[['name', 'categories', 'hy_rate']][:targets]
    return recommend_bar

def recommender2(CB_sim_matrix, CF_sim_matrix, reviewed, targets):
    print("Start combine similarity...")
    CF_result = dict()
    for b_id, rating in reviewed.items():
        cf_n_sim = sorted(CF_sim_matrix[b_id].items(), key=itemgetter(1), reverse=True)
        for m, similarity in cf_n_sim:
            if m not in reviewed:
                CF_result.setdefault(m, 0)
                CF_result[m] += similarity * float(rating)

    for cf_id in CF_result:
        for b_id in reviewed:
            CB_sim_matrix[b_id][cf_id] = CB_sim_matrix[b_id][cf_id] * 0.7 + CF_result[cf_id] * 0.3

    final_sim_matrix = CB_sim_matrix.copy()

    return final_sim_matrix

def final_recommender(bar_data_df, final_sim_matrix, reviewed, targets, delivery_option):
    print("Getting the final recommender...")
    recommend = set()
    for bar_id in reviewed:
        if bar_id in final_sim_matrix.columns:
            CB_n_sim = list(final_sim_matrix.sort_values(bar_id, ascending=False).index[1:targets * 2])
            for i in CB_n_sim:
                if i not in reviewed:
                    recommend.add(i)

    final_df = pd.DataFrame(list(recommend), columns=['business_id'])
    if delivery_option:
        bar_data_df = bar_data_df[bar_data_df['delivery or takeout'].apply(lambda x: x == 1)]
    else:
        bar_data_df = bar_data_df[bar_data_df['delivery or takeout'].apply(lambda x: x == 0)]

    top_bars = final_df.merge(bar_data_df, how='left', on='business_id')
    top_bars = top_bars.sort_values('dl_rate', ascending=False)
    recommend_bar = top_bars[['name', 'categories', 'dl_rate', 'delivery or takeout', 'Covid Banner']][:targets]

    return recommend_bar





def HY_cal_bar_recommend(bar_data_df, users_rating_df, user_id):
    print("Start prediction by CF...")
    user_watched = set(users_rating_df[users_rating_df['user_id'].apply(lambda x: x == user_id)]['business_id'])
    bar_dict = set(bar_data_df['business_id'])
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
    for bar in bar_dict:
        predict = model1.predict(user_id, bar)
        user_rating_pre[bar] = round(predict.est, 3)

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

    #final_df = pd.merge(bar_data_df, user_rates, how='left', on='business_id')
    #TF-IDF
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(bar_data_df['description'])
    cosine_similar = linear_kernel(tfidf_matrix, tfidf_matrix)
    matrix_sim = pd.DataFrame(cosine_similar, columns=bar_data_df['business_id'], index=bar_data_df['business_id'])
    rank = {}
    #recommend prediction
    for bar in bar_dict:
        for bus_id in user_watched:
            score = 0
            predict = model2.predict(user_id, bar)
            p_v = round(predict.est, 3)
            sim = matrix_sim[bus_id][bar]
            if bus_id == bar or sim == 0:
                continue
            else:
                score += (p_v * sim) / abs(sim)
        score = round(score / len(user_watched), 3)
        rank[bar] = score

    del model2
    hybrid_rates = pd.DataFrame(rank, index=[0]).transpose().reset_index()
    del rank
    hybrid_rates.columns = ['business_id', 'hy_rate']
    final_df = pd.merge(bar_data_df, hybrid_rates, how='left', on='business_id')
    top_bar = final_df.sort_values('hy_rate', ascending=False)
    recommend_bar = top_bar[['name', 'description', 'hy_rate']][:10]
    return recommend_bar

def csv_bar_rating_average(bar_dcit):
    print("Start collecting the bars' rating average...")
    reviews = pd.read_csv('./data_file/review_bars.csv')[['review_id', 'business_id', 'stars', 'date', 'user_id']]
    reviews['review_id'].dropna(axis=0)
    reviews = reviews[reviews['business_id'].apply(lambda x: x in bar_dcit)]
    idre_idres = dict(zip(reviews['review_id'], reviews['business_id']))
    reid_barar = dict(zip(reviews['review_id'], reviews['stars']))
    reid_userid = dict(zip(reviews['review_id'], reviews['user_id']))
    users_review = dict()
    rating_average = dict()
    rating_count = dict()
    p7.start(len(idre_idres))
    count = 0
    for re_id in idre_idres:
        bar_id = idre_idres[re_id]
        user_id = reid_userid[re_id]
        users_review[user_id] = {}
        if bar_id in bar_dcit:
            if bar_id not in rating_average:
                rating_average[bar_id] = reid_barar[re_id]
                rating_count[bar_id] = 1
                users_review[user_id].update({bar_id: reid_barar[re_id]})
                count += 1
                p7.update(count)
                continue
            rating_average.update({bar_id : round((rating_average[bar_id] + reid_barar[re_id])/2, 3)})
            rating_count.update({bar_id : rating_count[bar_id] + 1})
            #print(user_id, users_review[user_id])
            users_review[user_id].update({bar_id: reid_barar[re_id]})
            count += 1
            p7.update(count)

    users_rating = reviews[['user_id', 'business_id', 'stars']]
    del reviews
    del idre_idres
    del reid_barar
    del reid_userid
    ra_count_df = pd.DataFrame(rating_count, index=[0]).transpose().reset_index()
    ra_average_df = pd.DataFrame(rating_average, index=[0]).transpose().reset_index()
    ra_count_df.columns = ['business_id', 'ra_count']
    ra_average_df.columns = ['business_id', 'ra_average']
    bar_rating_average_df = pd.merge(ra_average_df, ra_count_df, how='left', on='business_id')

    m = bar_rating_average_df['ra_count'].quantile(0.95)
    bar_rating_average_df = bar_rating_average_df[bar_rating_average_df['ra_count'] >= m]
    c = bar_rating_average_df['ra_average'].mean()
    bar_rating_average_df['cb_ra_weight'] = bar_rating_average_df.apply(cb_ra_weight, axis=1, args=(m, c))
    bar_rating_average_df = bar_rating_average_df.sort_values('cb_ra_weight', ascending=False)

    del rating_average
    del rating_count
    del ra_count_df
    del ra_average_df
    del bar_dcit
    print("Rating average collected")
    return bar_rating_average_df, users_review, users_rating
def cb_ra_weight(bar_dcit, m, c):
    v = bar_dcit['ra_count']
    r = bar_dcit['ra_average']
    return v / (v + m) * r + m / (v + m) * c

def user_rating_filtering(users_review, bar_dict):
    print("Filtering the users rating...")
    users_review = users_review[users_review['business_id'].apply(lambda x: x in bar_dict)]
    print("Finished filtering")
    return users_review

def csv_user_collecting(bar_dcit, bar_dict):
    print('Start collecting user rating matrix...')
    user = pd.read_csv('./data_file/user_bars.csv')[['user_id', 'review_count', 'bar_review_count']]
    #user = user[user['bar_review_count'].apply(lambda x: int(x) >= 100)]
    kf = KFold(n_splits=5, shuffle=True)
    user = user.sort_values('bar_review_count', ascending=False)
    user_dict = set(user['user_id'])
    del user
    bar_lst = [i for i in bar_dict]
    final_rating_matrix = [bar_lst]
    p1.start(len(user_dict))
    count = 0
    Head = True
    #user-bar rating matrix
    for user in user_dict:
        if user in bar_dcit:
            if Head:
                final_rating_matrix = pd.DataFrame([bar_dcit[user]], columns=[key for key in bar_dict])
                Head = False
            final_rating_matrix = final_rating_matrix.append([bar_dcit[user]], ignore_index=True)
            '''
            user_rates_lst = [user]
            for bar in bar_dict:
                if bar in bar_dcit[user]:
                    user_rates_lst.append(int(bar_dcit[user][bar]))
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
    del bar_lst
    del bar_dcit
    del bar_dict
    #final_rating_matrix = np.array(final_rating_matrix, dtype=object)
    #print(final_rating_matrix)
    return final_rating_matrix

#DL part

def CF_SVD_rating_prediction(bar_data_df, users_rating_df, user_id):
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

    bar_dict = set(bar_data_df['business_id'])
    user_rating_predic = dict()
    for bar in bar_dict:
        predict = model.predict(user_id, bar)
        user_rating_predic[bar] = round(predict.est,3)

    user_rates = pd.DataFrame(user_rating_predic, index=[0]).transpose().reset_index()
    user_rates.columns = ['business_id', 'cf_prediction']
    user_rates = user_rates.sort_values('cf_prediction', ascending=False)

    print("Prediction finished")
    return user_rates

def CF_cal_bar_recommend(bar_data_df):
    top_bar = bar_data_df.sort_values('cf_prediction', ascending=False)
    recommend_bar = top_bar[['name', 'description', 'cf_prediction']][:10]
    return recommend_bar






if __name__ == '__main__':
    is_train = False
    u_id = input("Please input the user id:")
    targets = int(input("How many bars you want to recommended:"))
    delivery_option = int(input("Eat in or take-away/delivery: 0 eat-in, 1 take-away/delivery"))
    u_id = '5VESqAgYsL9vzLEIA_xgnw'
    targets = 10
    #CB recommender
    #visited = {'xfWdUmrz2ha3rcigyITV0g': 1, 'OETh78qcgDltvHULowwhJg': 1, '4JNXUYY8wbaaDmk3BPzlWw': 1, 'sKA6EOpxvBtCg7Ipuhl1RQ': 1}
    bar_data_df = csv_bar_categories_collect()
    covid_info_df = csv_covid_info_collect(bar_data_df['business_id'])
    bar_data_df = pd.merge(covid_info_df, bar_data_df, how='left', on='business_id')
    combination_df = csv_tips_cocategories_combination(bar_data_df[['business_id', 'categories']])
    bar_data_df = pd.merge(bar_data_df, combination_df, how='left', on='business_id')
    bar_data_df['categories'] = bar_data_df['categories'].apply(literal_eval)
    bar_data_df['categories'] = bar_data_df['categories'].apply(lambda x: ' '.join(x))
    user_rate_account_dict = user_rating_account(set(bar_data_df['business_id']))
    visited = user_rate_account_dict[u_id]
    CF_sim_matrix = CF_cal_bar_simularity(user_rate_account_dict)
    CB_sim_matrix = CB_cal_bar_simularity(bar_data_df)
    #DL AutoREC training recommender
    p_rate_dict = train_model(is_train, u_id, set(bar_data_df['business_id']))
    p_rate_df = pd.DataFrame(p_rate_dict, index=[0]).transpose().reset_index()
    p_rate_df.columns = ['business_id', 'dl_rate']
    bar_data_df = bar_data_df.merge(p_rate_df, how='left', on='business_id')
    #bar_data_df = recommender(CB_sim_matrix, CF_sim_matrix, p_rate_dict, bar_data_df, visited, targets)
    final_sim_matrix = recommender2(CB_sim_matrix, CF_sim_matrix, visited, targets)
    final_recommend = final_recommender(bar_data_df, final_sim_matrix, visited, targets, delivery_option)
    print("Here is your recommendation:")
    print(final_recommend)


    #rating_average_df, users_review, users_rating = csv_bar_rating_average(set(bar_data_df['business_id']))

