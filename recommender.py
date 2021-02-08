import os
import numpy as np
import pandas as pd
import math
import random
import time, datetime
from ast import literal_eval
from train import train_model
from operator import itemgetter
#import matplotlib.pyplot as plt


#from surprise import SVD
#from surprise import accuracy
#from surprise import Dataset, Reader
#from surprise.model_selection import KFold

import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

today = datetime.date.today()


def csv_collecting_users():
    print("Collecting the user's information...")
    user = pd.read_csv('./data_file/user_bars.csv')[['user_id', 'name']]
    return user

def csv_bar_categories_collect():
    print("Collecting the bar's information...")
    bar = pd.read_csv('./data_file/bars.csv')[['business_id', 'name', 'categories','is_open', 'review_count']]
    bar['business_id'].dropna(axis=0)
    bar['name'].dropna(axis=0)
    bar = bar.fillna(value={'categories': '', 'is_open': 0})
    bar = bar[~bar['is_open'].isin([0])]
    categories_dic = dict(zip(bar['business_id'], bar['categories']))
    categories_description = dict()
    #p8.start(len(bar))
    #count = 0
    for bar_id in set(bar['business_id']):
        #count += 1
        #p8.update(count)
        text = categories_dic[bar_id]
        text_lst = ''.join(text.split(','))
        text_lst = text_lst.split(' ')
        text_lst = [word.strip() for word in text_lst if word != "&"]
        text_lst = list(set(text_lst))
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
    #p9.start(len(user_bar))
    #count = 0
    for user_id in user_bar:
        #count += 1
        #p9.update(count)
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
            continue
        if len(bar_tips[bar_id]) > 70:
            continue
        bar_tips.setdefault(bar_id, [])
        bar_tips[bar_id] += text_lst


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

    return matrix_sim

def user_rating_account(bar_set, user_set):
    print("Collecting the bars' rating average...")
    reviews = pd.read_csv('./data_file/review_bars.csv')[['review_id', 'business_id', 'stars', 'user_id']]
    reviews['review_id'].dropna(axis=0)
    reviews = reviews[reviews['business_id'].apply(lambda x: x in bar_set)]
    user_df = user_set.merge(reviews['user_id'], how='left', on='user_id')
    user_df = user_df.drop_duplicates(['user_id'])
    user_df = user_df.reset_index(drop=True)
    print("Please input the id number in range 0 ~ {0}".format(len(user_df)-1), '\n')
    id_num = input()
    user_id = user_df['user_id'][int(id_num)]
    user_name = user_df['name'][int(id_num)]
    print("Hello", user_name)
    reid_idres = dict(zip(reviews['review_id'], reviews['business_id']))
    reid_barra = dict(zip(reviews['review_id'], reviews['stars']))
    reid_userid = dict(zip(reviews['review_id'], reviews['user_id']))
    del reviews
    users_review_train = dict()
    users_review_test = dict()
    for re_id in reid_userid:
        u_id = reid_userid[re_id]
        b_id = reid_idres[re_id]
        star = reid_barra[re_id]
        if random.random() < 0.9 or u_id == user_id:
            users_review_train.setdefault(u_id, {})
            users_review_train[u_id].update({b_id: star})
        else:
            users_review_test.setdefault(u_id, {})
            users_review_test[u_id].update({b_id: star})

    del reid_idres
    del reid_barra
    del reid_userid
    return users_review_train, users_review_test, user_id

def CF_cal_bar_simularity(views_dict):
    print("Calculating bar similarity with CF...")
    bars_sim_matrix = dict()
    bars_popularity = dict()
    for u_id in views_dict:
        for b_id in views_dict[u_id]:
            bars_popularity.setdefault(b_id, 0)
            bars_popularity[b_id] += 1
            for b_id2 in views_dict[u_id]:
                if b_id == b_id2:
                    continue
                bars_sim_matrix.setdefault(b_id, {})
                bars_sim_matrix[b_id].setdefault(b_id2, 0)
                bars_sim_matrix[b_id][b_id2] += 1

    for b1 in bars_sim_matrix:
        for b2 in bars_sim_matrix[b1]:
            bars_sim_matrix[b1][b2] = bars_sim_matrix[b1][b2] / math.sqrt(bars_popularity[b1] * bars_popularity[b2])

    #print(bars_sim_matrix)
    return bars_sim_matrix

def matrix_combination(CB_sim_matrix, CF_sim_matrix, reviewed):
    print("Combining similarity...")
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
    recommend = set()
    for bar_id in reviewed:
        if bar_id in final_sim_matrix.columns:
            CB_n_sim = list(final_sim_matrix.sort_values(bar_id, ascending=False).index[1:targets * 2])
            for i in CB_n_sim:
                if i not in reviewed:
                    recommend.add(i)

    final_df = pd.DataFrame(list(recommend), columns=['business_id'])

    top_bars = final_df.merge(bar_data_df, how='left', on='business_id')
    top_bars = top_bars.sort_values('dl_rate', ascending=False)
    if delivery_option == 1:
        delivery_top_bars = top_bars[top_bars['delivery or takeout'].apply(lambda x: x == 1)]
        recommend_bar = delivery_top_bars[['name', 'categories', 'dl_rate', 'delivery or takeout', 'Covid Banner', 'business_id']][:targets]
    elif delivery_option == 0:
        eat_in_top_bars = top_bars[top_bars['delivery or takeout'].apply(lambda x: x == 0)]
        recommend_bar = eat_in_top_bars[['name', 'categories', 'dl_rate', 'delivery or takeout', 'Covid Banner', 'business_id']][:targets]
    else:
        recommend_bar = top_bars[['name', 'categories', 'dl_rate', 'delivery or takeout', 'Covid Banner', 'business_id']][:targets]

    if len(recommend_bar) == 0:
        print("Sorry, there is no suitable Bars for your option, other recommendations are presented below, hope you would like it")
        recommend_bar = top_bars[['name', 'categories', 'dl_rate', 'delivery or takeout', 'Covid Banner', 'business_id']][:10]

    return recommend_bar


'''
def evalution(train_dict, test_dict, target_test):
    print("Evaluate...")
    used = 0
    recommend_count = 0
    test_count = 0
    bar_all_recommend = set()

    #p12.start(len(train_dict))
    count = 0
    y = [0,0]
    yp = [0,0]
    yr = [0,0]
    #yc = [0,0]
    print(len(train_dict))
    for u_id in train_dict:
        count += 1
        y[1] = count
        #p12.update(count)

        bar_test = test_dict.get(u_id, {})
        bar_recommend = final_recommender(bar_data_df, final_sim_matrix, train_dict[u_id], target_test, 3)
        bar_recommend = list(bar_recommend['business_id'])
        for bar_id in bar_recommend:
            if bar_id in bar_test:
                used += 1
                print("IN")
            bar_all_recommend.add(bar_id)
        recommend_count += target_test
        test_count += len(bar_test)
        if test_count == 0 or recommend_count == 0:
            continue

        precision = used / recommend_count
        recall = used / test_count
        coverage = len(bar_all_recommend) / len(bar_data_df)
        yp[1] = precision
        yr[1] = recall
        #yc[1] = coverage
        print("precision:", precision)
        print("recall:", recall)
        print("coverage:", coverage)
        print('\n')
        plt.plot(y, yp, color='green', label='precision')
        plt.plot(y, yr, color='blue', label='recall')
        #plt.plot(y, yc, color='yellow', label='recall')
        #plt.draw()
        #plt.show()
        y[0] = y[1]
        yp[0] = yp[1]
        yr[0] = yr[1]
        #yc[0] = yc[1]
        plt.pause(0.0000001)

    coverage = len(bar_all_recommend) / len(bar_data_df)
    result = (precision, recall, coverage)

    print('Precision = %.4f\nRecall = %.4f\nCoverage = %.4f' % result)

def CF_SVD_rating_prediction(bar_data_df, users_rating_df, user_id):
    print("Prediction by CF...")
    reader = Reader(rating_scale=(0, 5))
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
    top_bars = user_rates.merge(bar_data_df, how='left', on='business_id')
    recommend_bar = top_bars[['name', 'categories', 'dl_rate', 'delivery or takeout', 'Covid Banner', 'business_id']][
                    :targets]

    return recommend_bar
'''


if __name__ == '__main__':
    is_train = True
    if is_train:
        print("You are running in the train mode, the trained model will be saved in AutoRec_train.nodel")
    else:
        print("You are running in the normal mode, the model you are using is AutoRec.model")
    time.sleep(2)
    #u_id = 'ACMYTmlycF2-HgSZ8lyC0A'
    #targets = 10
    #data preparetion
    user_set = csv_collecting_users()
    bar_data_df = csv_bar_categories_collect()
    covid_info_df = csv_covid_info_collect(bar_data_df['business_id'])
    bar_data_df = pd.merge(covid_info_df, bar_data_df, how='left', on='business_id')
    combination_df = csv_tips_cocategories_combination(bar_data_df[['business_id', 'categories']])
    bar_data_df = pd.merge(bar_data_df, combination_df, how='left', on='business_id')
    bar_data_df['categories'] = bar_data_df['categories'].apply(literal_eval)
    bar_data_df['categories'] = bar_data_df['categories'].apply(lambda x: ' '.join(x))


    user_rate_account_dict_train, user_rate_account_dict_test, u_id = user_rating_account(set(bar_data_df['business_id']), user_set)
    visited = user_rate_account_dict_train[u_id]
    print("The bar you had visited:")
    v = bar_data_df.loc[bar_data_df['business_id'].isin([i for i in visited])]
    print(v[['name', 'categories']], '\n')
    print("How many target you want for final recommendation? (suggest 10)")
    targets = int(input())
    print("Eat in or take-away/delivery:", "\n", "0:eat-in", "\n", "1:take-away/delivery", "\n", "3:don't mind")
    delivery_option = int(input())
    #calculating the similarity matrix
    CF_sim_matrix = CF_cal_bar_simularity(user_rate_account_dict_train)
    CB_sim_matrix = CB_cal_bar_simularity(bar_data_df)
    #DL AutoREC training recommender
    p_rate_dict = train_model(is_train, u_id)
    p_rate_df = pd.DataFrame(p_rate_dict, index=[0]).transpose().reset_index()
    p_rate_df.columns = ['business_id', 'dl_rate']
    bar_data_df = bar_data_df.merge(p_rate_df, how='left', on='business_id')
    #combine the similarity matrix
    final_sim_matrix = matrix_combination(CB_sim_matrix, CF_sim_matrix, visited)
    print('Getting the final recommender...')
    final_recommend = final_recommender(bar_data_df, final_sim_matrix, visited, targets, delivery_option)
    print("Here is your recommendation:")
    print(final_recommend[['name', 'categories', 'Covid Banner']])
    #evalution(user_rate_account_dict_train, user_rate_account_dict_test, targets)

