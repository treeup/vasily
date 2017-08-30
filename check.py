#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
import math

users_number = 0
items_number = 0

def make_user_item_pairs():
    # データからitem/userの配列を用意したい。
    df = pd.read_csv("./ml-100k/u1.base", sep="\t", names=["user_id","item_id","rating", "timestamp"])

    global users_number
    global items_number
    users_number = df.max().ix["user_id"]
    items_number = df.max().ix["item_id"]

    # userとitemのpairを表す二次元配列
    # 縦user943行、横item1682列の零配列
    user_item_pairs = np.zeros([users_number,items_number])

    # 映画の情報のデータ
    # df_movie = pd.read_csv("./ml-100k/u.item", sep="|")

    # user_id,item_idに対応するratingの値を、user_item_pairsに代入していく。
    # user_id,item_idは1から始まっているので、それぞれ-1している。
    for i in range(len(df)):
        user_item_pairs[df.ix[i]["user_id"]-1][df.ix[i]["item_id"]-1] = df.ix[i]["rating"]

    return user_item_pairs

def search_movie_title(id_list,score_list):
    movie_title_list=[]
    score_list_reverse = sorted(score_list)[::-1]
    # 映画の情報のデータ
    df_movie = pd.read_csv("./ml-100k/u.item", sep='|', encoding="ISO-8859-1",header=None).ix[:,0:1]
    for x,i in enumerate(id_list):
        movie_title_list.append(list([df_movie.ix[i][1],score_list_reverse[x]]))
    return movie_title_list

#コサイン類似度
def cos_similarity(x,y):
    return np.dot(x,y) / (np.linalg.norm(x) * np.linalg.norm(y))

#ユーザー同士の類似度を配列で返す
def user_user_similarity(pairs):
    users_sim = np.zeros((users_number, users_number))
    
    for i in range(users_number):
        for j in range(i,users_number):
            if i == j:
                users_sim[i][j] = 1.0
            else:
                sim_score = cos_similarity(pairs[i], pairs[j])
                users_sim[i][j] = sim_score
                users_sim[j][i] = sim_score
                
    return users_sim

#  user_idとユーザー間類似度と、user/itemの対応リストを与えた時に、user_idのユーザーに映画をリコメンドする。
# user_idは-1して用いるのに注意
# kは推薦する映画の本数
# lは上位L人の高い類似度をもつユーザーを対象にする
def movie_recommend(users_sim, user_item_pairs):
    
    # l = int(input("対象にする類似ユーザー数を入力してください : "))
    # if l > 943:
    #     print("Error with number of users.")
    #     sys.exit(1)
    l = 40
    l = l + 1

    df_real = pd.read_csv("./ml-100k/u1.test", sep="\t", names=["user_id","item_id","rating", "timestamp"])
    array_real = df_real.ix[:,0:3].as_matrix()
    test_users_list = list(set(array_real.T[0]))
    
    for u in test_users_list:
    
        #user_idの人との類似度が高いL人の人を対象に考える
        #それ以外の人の類似度は考えないため、０で置き換える
        high_sim_users_indices = np.argpartition(-users_sim[u-1], l)[:l]       
        for x in range(users_number):
            if x in high_sim_users_indices:
                pass
            else:
                users_sim[u-1][x] = 0
    
    ###ここまでは共通###
    #array_realにあるtestデータの評価を行い、比較を行う
    #そのために、まず訓練データの配列を用意する必要がある
    
    #正解データにあるユーザーとアイテムの組み合わせから予測値を計算する
    sim_scores = []
    for i in array_real:
        #     重みの合計
        total_weight = 0
    
        total_weight = np.dot(users_sim[i[0]-1],user_item_pairs.T[i[1]])
        
        for_normalization = 0
        for j in range(users_number):
            if user_item_pairs[j][i[1]] != 0:
                for_normalization += users_sim[i[0]-1][j]
        if for_normalization != 0:
            similarity_score = total_weight / for_normalization
        else:
            similarity_score = 0
        sim_scores.append(similarity_score)
     
    #RMSEを計算する
    real_scores = array_real.T[2]
    num = len(real_scores)
    rmse = 0
    for i in range(0,num-1):
        dif = real_scores[i]-sim_scores[i]
        rmse += dif * dif
    rmse = rmse/num
    rmse = math.sqrt(rmse)

    return rmse

if __name__ == '__main__':
    u_i_pair = make_user_item_pairs()
    users_similarity = user_user_similarity(u_i_pair)
    result = movie_recommend(users_similarity, u_i_pair)
    print(result)
