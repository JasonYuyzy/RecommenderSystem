import os
import numpy as np
import pandas as pd
import json
import math
import progressbar
import random
import time, datetime
from ast import literal_eval
#from train import train_model


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

def csv_restaurant_categories_collect():
    print("Start collecting the restaurants' information...")
    restaurant = pd.read_csv('./data_file/bars.csv')[['business_id', 'name', 'categories','is_open', 'review_count']]
    restaurant['business_id'].dropna(axis=0)
    restaurant['name'].dropna(axis=0)
    restaurant['categories'].dropna(axis=0)
    restaurant['is_open'].dropna(axis=0)
    restaurant = restaurant[~restaurant['is_open'].isin([0])]
    #restaurant = restaurant[restaurant['review_count'].apply(lambda x: int(x) > 400)]
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
        count += 1
        p8.update(count)
        if "Food" in text_lst:
            text_lst.remove("Food")
        if "Restaurants" in text_lst:
            text_lst.remove("Restaurants")
        categories_description[rest_id] = str(text_lst)


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
    tips_df = pd.read_csv('./data_file/tip_bars.csv')
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
    covid = pd.read_csv('./data_file/covid_bars.csv')
    rest_covid_info = covid[covid.apply(covid_filtering_open_date, axis=1, args=(today.year, today.month))]
    print(rest_covid_info)
    rest_covid_info = rest_covid_info.replace({True: 1, False: 0})#, inplace=True)

    rest_covid_info = pd.merge(restaurant_dcit, rest_covid_info, how='left', on='business_id')
    del rest_covid_info['Covid Banner']
    del rest_covid_info['Temporary Closed Until']
    del rest_covid_info['Virtual Services Offered']
    print(rest_covid_info)
    print(len(restaurant_dcit))
    exit()
    rest_covid_info.dropna()
    #rest_covid_info['review_id'].dropna(axis=0)
    #rest_covid_info['business_id'].dropna(axis=0)
    #rest_covid_info['stars'].dropna(axis=0)
    print("Covid info collected")
    return rest_covid_info
def covid_filtering_open_date(covid_info, year, month):
    date_check = covid_info['Temporary Closed Until'].split('-')
    if len(date_check) == 1:
        return True
    if int(date_check[0]) >= year and int(date_check[1]) >= month:
        return False
    else:
        return True

def csv_restaurant_rating_average(restaurant_dcit):
    print("Start collecting the restaurants' rating average...")
    reviews = pd.read_csv('./data_file/review_bars.csv')[['review_id', 'business_id', 'stars', 'date', 'user_id']]
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
    user = pd.read_csv('./data_file/user_bars.csv')[['user_id', 'review_count', 'restaurant_review_count']]
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
    u_id = '5VESqAgYsL9vzLEIA_xgnw'
    #CB recommender
    visited = {'xfWdUmrz2ha3rcigyITV0g': 1, 'OETh78qcgDltvHULowwhJg': 1, '4JNXUYY8wbaaDmk3BPzlWw': 1, 'sKA6EOpxvBtCg7Ipuhl1RQ': 1}
    rest_data_df = csv_restaurant_categories_collect()
    covid_info_df = csv_covid_info_collect(rest_data_df['business_id'])
    rest_data_df = pd.merge(covid_info_df, rest_data_df, how='left', on='business_id')
    print(rest_data_df)
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
    #p_rate_dict = train_model(is_train, u_id)
    #DL_prate_df = pd.DataFrame(p_rate_dict, index=[0]).transpose().reset_index()
    #DL_prate_df.columns = ['business_id', 'dl_prate']
    #print(DL_prate_df)
