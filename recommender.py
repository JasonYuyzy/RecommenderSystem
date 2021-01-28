import os
import numpy as np
import pandas as pd
import json
import progressbar
import random
import time, datetime
from ast import literal_eval

from surprise import SVD
from surprise import accuracy
from surprise import Dataset, Reader
from surprise.model_selection import KFold
from surprise.model_selection import cross_validate, train_test_split


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

user_id = 'QaELAmRcDc5TfJEylaaP8g'
#collaborative filtering item-item

def restaurant_user_finder(user_id): #friends so far
    print("Finding user groups...")
    users_group = {}
    count = 0
    with open('./rest_data/users_restaurants.json', mode='r', encoding='utf-8') as f:
        dicts = json.load(f)
        p1.start(len(dicts))
        for i in dicts:
            count += 1
            #if len(i['friends'].split(',')) > 30:
                #user_list.append(i['user_id'])
            #if i['user_id'] == user_id:
            if len(i['friends'].split(',')) > 200:
                friend_lst = i['friends'].split(',')
                for friend in friend_lst:
                    users_group[friend] = 1
                #break
                #if len(users_group) > 100000:
                    #break
                    #p1.update(len(dicts))
            p1.update(count)

    f.close()
    del dicts
    #print(user_list)
    print("Found friends:", len(users_group))
    return users_group

def user_visited(user_id):
    with open('./rest_data/users_reviews_restaurants.json', mode='r', encoding='utf-8') as f:
        dicts = json.load(f)
        visited = {}
        print("Review number:", len(dicts))
        for all in dicts:
            for i in all:
                if len(all[i]) == 40:
                    for busi in all[i]:
                        if all[i][busi] ==5 :
                            visited[busi] = all[i][busi]
                            break
                    break
            break


    f.close()
    del dicts
    print("Visited collected", visited)
    return visited

def restaurant_review_finder(users_group):
    print("Finding restaurant and reviews...")
    review_group = {}
    restaurant_reviews_average = {}
    num_of_review = 0
    with open('./rest_data/reviews_restaurants.json', mode='r', encoding='utf-8') as f:
        dicts = json.load(f)
        print("Review number:", len(dicts))
        for i in dicts:
            if i['user_id'] in users_group:
                if i['user_id'] not in review_group:
                    review_group.update({i['user_id']: {}})
                review_group[i['user_id']][i['business_id']] = i['stars']
                if i['business_id'] not in restaurant_reviews_average:
                    restaurant_reviews_average.update({i['business_id']: i['stars']})
                origin_star = restaurant_reviews_average[i['business_id']]
                restaurant_reviews_average.update({i['business_id']: (i['stars']+origin_star)/2})
                num_of_review += 1

    f.close()
    print("Found review:", num_of_review, "from ", len(review_group), "users")
    del dicts
    return review_group, restaurant_reviews_average

def restaurant_feature_finder():
    print("Collecting features...")
    features_group = {}
    features = {}
    with open('./rest_data/restaurants_features.json', mode='r', encoding='utf-8') as f:
        dicts = json.load(f)
        print("Restaurant number:", len(dicts))
        for i in dicts:
            features_group.update({i: {}})
            for j in dicts[i]: #go through each feature
                #filtering features details
                if j not in features:
                    features[j]={}
                    if dicts[i][j] == True and type(dicts[i][j]) != int:
                        features[j][dicts[i][j]] = 1
                    elif dicts[i][j] == False:
                        features[j][dicts[i][j]] = 0
                    else:
                        features[j][dicts[i][j]] = 0

                if dicts[i][j] not in features[j]:
                    if dicts[i][j] == True and type(dicts[i][j]) != int:
                        features[j][dicts[i][j]] = 1
                    elif dicts[i][j] == False:
                        features[j][dicts[i][j]] = 0
                    else:
                        features[j][dicts[i][j]] = len(features[j])

                if dicts[i][j] == True:
                    features_group[i].update({j: 1})
                elif dicts[i][j] == False:
                    features_group[i].update({j: 0})
                else:
                    features_group[i].update({j: features[j][dicts[i][j]]})

        return features_group, features

def restaurant_categories_finder():
    print("Collecting categories...")
    categories_group = {}
    categories = {}
    with open('./rest_data/restaurants_categories.json', mode='r', encoding='utf-8') as f:
        dicts = json.load(f)
        print("Restaurant number:", len(dicts))
        for i in dicts:
            categories_group.update({i: {}})
            for j in dicts[i]:  # go through each feature
                # filtering the repeat
                char = j.split()
                # filtering features details
                if ''.join(char) not in categories:
                    categories[''.join(char)] = 1
                else:
                    categories[''.join(char)] = 1 + categories[''.join(char)]

                categories_group[i][''.join(char)] = 1
        del dicts
        categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
        #print(categories[2:202])
        return categories_group, categories[2:202]

def restaurant_tips_finder():
    print("Collecting tips...")
    tips_group = {}
    tips = {}
    count = 0
    with open('./rest_data/tips_restaurants.json', mode='r', encoding='utf-8') as f:
        dicts = json.load(f)
        stemmer = SnowballStemmer('english')
        stop = set(stopwords.words('english'))
        print("Tips number:", len(dicts))
        p4.start(63961)
        for i in dicts:
            try:
                #count += 1
                p4.update(len(tips_group))
                if len(tips_group) == 63961:
                    break
                if len(tips_group[i['business_id']]) > 50:
                    continue
            except:
                text_lst = [w for w in i['text'].split(' ') if w not in stopwords.words('english')]
                text_lst = list(set(text_lst))
                text = ' '.join(text_lst)
                if i['business_id'] not in tips_group:
                    tips_group.update({i['business_id']: stemmer.stem(text)})
                    continue
                tips_group.update({i['business_id']: stemmer.stem(text + " " + tips_group[i['business_id']])})
                #count += 1
                p4.update(len(tips_group))


        #print(tips_group)
        #exit()
        # print(categories[2:202])
        return tips_group

def restaurant_rating_average_finder():
    print("Collecting rating average...")
    rating_count = dict()
    rating_sum = dict()
    rating_average = dict()
    rest = dict()
    count = 0
    with open('./rest_data/reviews_restaurants.json', mode='r', encoding='utf-8') as f:
        dicts = json.load(f)
        p5.start(len(dicts))
        print("Review number:", len(dicts))
        for i in dicts:
            #if int(i['date'].split('-')[0]) < 2015:
                #count += 1
                #p5.update(count)
                #continue
            if i['business_id'] is None and i['user_id'] is None:
                continue
            if i['business_id'] not in rating_count:
                rating_sum[i['business_id']] = i['stars']
                rating_count[i['business_id']] = 1
                rest[i['business_id']] = 1
                continue
            rating_sum.update({i['business_id']: rating_sum[i['business_id']]+i['stars']})
            rating_count.update({i['business_id']: rating_count[i['business_id']]+1})
            count += 1
            p5.update(count)

        del dicts
        f.close()

        for r in rating_sum:
            rating_average[r] = round(rating_sum[r]/rating_count[r], 3)
        del rating_sum
        dataf_count = pd.DataFrame(rating_count, index=[0]).transpose().reset_index()
        dataf_count.columns = ['business_id', 'r_count']
        dataf_average = pd.DataFrame(rating_average, index=[0]).transpose().reset_index()
        dataf_average.columns = ['business_id', 'r_average']
        rating_count_average = pd.merge(dataf_count, dataf_average, how='left', on='business_id')
        del dataf_count
        del dataf_average
        m = rating_count_average['r_count'].quantile(0.95)
        rating_count_average = rating_count_average[rating_count_average['r_count'] >= m]
        c = rating_count_average['r_average'].mean()

        rating_count_average['r_weight'] = rating_count_average.apply(ra_weight, axis=1, args=(m, c))
        rating_count_average = rating_count_average.sort_values('r_weight', ascending=False)


        print("Rating average collecting done")
        return rating_count_average, rest

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

def building_review_matrix(review_group, restaurant_reviews_average):
    print("Creating review matrix...")
    restaurant_array = [i for i in restaurant_reviews_average]
    final_review_array = [restaurant_array]
    p2.start(len(review_group))
    count = 0
    for user_id in review_group:
        user_array = [user_id]
        for j in restaurant_reviews_average:
            if j in review_group[user_id]:
                user_array.append(review_group[user_id][j])
            else:
                user_array.append(None)
        final_review_array.append(user_array)
        count += 1
        p2.update(count)
    final_array = np.array(final_review_array, dtype=object)
    #final_array = final_array.pivot_table(index='user_id', columns='product_id', values='order_id', aggfunc='count')

    print(final_array[0][3])
    print(final_array[4][0])
    print(final_array[7][12])
    print("Done review matrix")
    return final_array

def building_restaurant_feature_matrix(feature_group, feature):
    print("Creating restaurant feature matrix...")
    restaurant_array = [i for i in feature]
    final_array = [restaurant_array]
    p3.start(len(feature_group))
    count = 0
    for rest_id in feature_group:
        rest_array = [rest_id]
        for j in feature:
            if j in feature_group[rest_id]:
                rest_array.append(feature_group[rest_id][j])
            else:
                rest_array.append(None)
        final_array.append(rest_array)
        count += 1
        p3.update(count)

    final_array = np.array(final_array, dtype=object)
    print("Done features matrix")
    return final_array

def building_restaurant_categories_matrix(categories_group, categories):
    print("Creating restaurant feature matrix...")
    categories_array = [i[0] for i in categories]
    final_array = [categories_array]
    p3.start(len(categories_group))
    count = 0
    for rest_id in categories_group:
        categories_array = [rest_id]
        for j in categories:
            if j[0] in categories_group[rest_id]:
                categories_array.append(categories_group[rest_id][j[0]])
            else:
                categories_array.append(0)
        final_array.append(categories_array)
        count += 1
        p3.update(count)

    final_array = np.array(final_array, dtype=object)
    print("Done categories matrix")
    return final_array



def csv_restaurant_categories_combiner():
    print("Start collecting the restaurants' information...")
    restaurant = pd.read_csv('./data_file/restaurant.csv')[['business_id', 'name', 'categories','is_open', 'review_count']]
    restaurant['business_id'].dropna(axis=0)
    restaurant['name'].dropna(axis=0)
    restaurant['categories'].dropna(axis=0)
    restaurant['is_open'].dropna(axis=0)
    restaurant = restaurant[~restaurant['is_open'].isin([0])]
    restaurant = restaurant[restaurant['review_count'].apply(lambda x: int(x) > 500)]
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
        categories_description[rest_id] = str(text_lst)
        count += 1
        p8.update(count)

    name = restaurant.drop(['categories','is_open', 'review_count'], axis=1)
    del text_lst
    del text
    del restaurant


    rest_categories = pd.DataFrame(categories_description, index=[0]).transpose().reset_index()
    rest_categories.columns = ['business_id', 'categories']
    rest_categories['categories'] = rest_categories['categories'].apply(literal_eval)
    rest_categories['categories'] = rest_categories['categories'].apply(lambda x: ' '.join(x))
    restaurant = pd.merge(rest_categories, name, how='left', on='business_id')
    #print(rest_categories)

    del categories_dic
    del categories_description
    print("Information collected")
    return restaurant

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
    user = user[user['restaurant_review_count'].apply(lambda x: int(x) >= 100)]
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
    del rest_lst
    del restaurant_dcit
    del rest_dict
    #final_rating_matrix = np.array(final_rating_matrix, dtype=object)
    #print(final_rating_matrix)
    return final_rating_matrix

def CF_SVD_rating_prediction(rest_data_df, users_rating_df, user_id):
    print("Start prediction by CF...")
    reader = Reader(rating_scale=(1, 5))
    rating_data = Dataset.load_from_df(users_rating_df, reader)

    cross_validation = KFold(n_splits=5)
    model = SVD()
    for trainset, testset in cross_validation.split(rating_data):
        model.fit(trainset)
        predictions = model.test(testset)

        print("Learning rate:", accuracy.rmse(predictions, verbose=True))

    rest_dict = set(rest_data_df['business_id'])
    user_rating_predic = dict()
    for rest in rest_dict:
        predict = model.predict(user_id, rest)
        user_rating_predic[rest] = round(predict.est,3)

    user_rates = pd.DataFrame(user_rating_predic, index=[0]).transpose().reset_index()
    user_rates.columns = ['business_id', 'cf_prediction']
    final_df = pd.merge(rest_data_df, user_rates, how='left', on='business_id')
    print("Prediction finished")
    return final_df

def CF_cal_restaurant_recommend(rest_data_df):
    top_rest = rest_data_df.sort_values('cf_prediction', ascending=False)
    recommend_rest = top_rest[['name', 'categories', 'cf_prediction']][:10]
    return recommend_rest


def CB_cal_restaurant_recommend(restaurant, visited):
    print("Starting recommend...")
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(restaurant['categories'])

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
    recommend_rest = top_rest[['name', 'categories', 'cb_ra_weight', 'ra_average']][:10]
    return recommend_rest


#users_group = restaurant_user_finder(user_id)
#review_group, restaurant_reviews_average = restaurant_review_finder(users_group)
#feature_group, feature = restaurant_feature_finder()
#categories_group, categories = restaurant_categories_finder()
#review_matrix = building_review_matrix(review_group, restaurant_reviews_average)
#feature_matrix = building_restaurant_feature_matrix(feature_group, feature)
#categories_matrix = building_restaurant_categories_matrix(categories_group, categories)
#restaurant_tips_finder()
#restaurant_rating_average, rest = restaurant_rating_average_finder()
#restaurant_description, rests = restaurant_tips_categories_combiner(rest)

if __name__ == '__main__':
    #CB recommender
    visited = {'Yl05MqCs9xRzrJFkGWLpgA': 1, '2iTsRqUsPGRH1li1WVRvKQ': 1, 'fuW-VCynECpKukrm-9nxdg': 1}
    rest_data_df = csv_restaurant_categories_combiner()
    covid_info_df = csv_covid_info_collect(rest_data_df['business_id'])
    rest_data_df = pd.merge(covid_info_df, rest_data_df, how='left', on='business_id')
    rating_average_df, users_review, users_rating = csv_restaurant_rating_average(set(rest_data_df['business_id']))
    rest_data_df = pd.merge(rating_average_df, rest_data_df, how='left', on='business_id')
    top_10_restaurant = CB_cal_restaurant_recommend(rest_data_df, visited)
    print(top_10_restaurant)
    #user_features_df = csv_user_collecting(users_review, set(rest_data_df['business_id']))

    #CF recommender
    user_id = 'HoyH3jYg9wgReyVmaTxlTg'
    users_rating_df = user_rating_filtering(users_rating, set(rest_data_df['business_id']))
    rest_data_df = CF_SVD_rating_prediction(rest_data_df, users_rating_df, user_id)
    top_10_restaurant = CF_cal_restaurant_recommend(rest_data_df)
    print(top_10_restaurant)
    #restaurant = pd.merge(rating_average_with_weight_df, restaurant_info_df, how='left', on='business_id')
    #visited = user_visited(user_id)
    #print(restaurant)

