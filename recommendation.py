#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
import pprint


users_number = 0
items_number = 0

def make_user_item_pairs():
	# データからitem/userの配列を用意したい。
	df = pd.read_csv("./ml-100k/u.data", sep="\t", names=["user_id","item_id","rating", "timestamp"])

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

#movie_idに対応するmovie_titleをリストで返す
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
    
    user_id = int(input("user_idを入力してください : "))
    # エラー処理
    if user_id-1 not in range(users_number):
        print("No such a user_id. You need to update \"u.data\" for recommendation.")
        sys.exit(1)

    k = int(input("推薦する映画の本数を入力してください : "))
    if k >1682:
        print("Error with number of recommendation.")
        sys.exit(1)

    l = int(input("対象にする類似ユーザー数を入力してください : "))
    if l > 943:
        print("Error with number of users.")
        sys.exit(1)
    l = l + 1

    #     重みの合計
    total_weight = 0
    sim_scores = []
    
    #user_idの人との類似度が高いL人の人を対象に考える
    #それ以外の人の類似度は考えないため、０で置き換える
    high_sim_users_indices = np.argpartition(-users_sim[user_id-1], l)[:l]       
    for x in range(users_number):
        if x in high_sim_users_indices:
            pass
        else:
            users_sim[user_id-1][x] = 0
    
    
    for i in range(items_number):
        # まだ評価のない（０）ものを対象に処理を行う
        if user_item_pairs[user_id-1][i] == 0:
            
            # 作品iへの評価値と、user_idの人への類似度を掛け合わせた重みの合計を内積を用いて計算
            # 自分自身への類似度は1になるため、その分を差し引いている
            total_weight = np.dot(users_sim[user_id-1],user_item_pairs.T[i])

            #たくさんの人に評価された作品の重みは大きくなる
            #作品iを評価している(!=0)評価者の類似度の合計を求め、正規化する
            for_normalization = 0            
            for j in range(users_number):
                if user_item_pairs[j][i] != 0:
                    for_normalization += users_sim[user_id-1][j]
            if for_normalization != 0:
                similarity_score = total_weight / for_normalization
            else:
                similarity_score  = 0
            sim_scores.append(similarity_score)

    #正規化した類似度スコアの内、上位K件のitem_idを降順にソートして取得する
    sim_scores_array = np.array(sim_scores)
    unsorted_max_indices = np.argpartition(-sim_scores_array, k)[:k]
    k_max = sim_scores_array[unsorted_max_indices]
    indices = np.argsort(-k_max)
    max_k_indices = unsorted_max_indices[indices]
    
    return max_k_indices, k_max


if __name__ == '__main__':
	u_i_pair = make_user_item_pairs()
	users_similarity = user_user_similarity(u_i_pair)
	movie_id = movie_recommend(users_similarity, u_i_pair)
	result = search_movie_title(movie_id[0],movie_id[1])
	# print(result)
	pprint.pprint(result)
