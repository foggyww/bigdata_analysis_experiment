import pickle
import sys

import pandas as pd
from math import log
import numpy as np
import math
from collections import defaultdict
import random


def generate_affine_hash_function(n):
    while True:
        a = random.randint(1, n-1)
        if math.gcd(a, n) == 1:
            b = random.randint(0, n-1)
            return lambda x: (a * x + b) % n

def generate_affine_hash_functions(n, num_functions):
    hash_functions = []
    for _ in range(num_functions):
        hash_functions.append(generate_affine_hash_function(n))
    return hash_functions


def get_user_similar(user_anime_matrix, anime_id_all, user_id_all):
    try:
        return np.load('temp/user_similar_2.npy')
    except:
        user_num = max(user_id_all)
        anime_num = len(anime_id_all)
        anime_index = {}
        index = 0
        for anime_id in anime_id_all:
            anime_index[anime_id] = index
            index += 1
        # 01 处理效用矩阵
        user_anime_matrix_01 = {}
        for user_id, anime_dict in user_anime_matrix.items():
            user_anime_matrix_01[user_id] = {anime_id: 1 if score >= 5 else 0 for anime_id, score in anime_dict.items()}

        num_hash = 100
        hash_matrix = np.zeros((anime_num,num_hash),dtype=int)
        hash_functions = generate_affine_hash_functions(anime_num,num_hash)
        for i in range(anime_num):
            for j in range(num_hash):
                hash_matrix[i][j] = hash_functions[j](i)

        user_minhash_matrix = np.zeros((num_hash,user_num+1))
        for user_id, anime_dict in user_anime_matrix_01.items():
            temp_min_array = [random.randint(100001, sys.maxsize)]*num_hash
            for anime_id,score in anime_dict.items():
                if score == 1:
                    for i in range(num_hash):
                        temp_min_array[i] = min(hash_matrix[anime_index[anime_id]][i],temp_min_array[i])
            for i in range(num_hash):
                user_minhash_matrix[i][user_id] = temp_min_array[i]

        # 计算 Jaccard 相似度矩阵
        user_similarity_matrix = np.eye(user_num+1)
        for i in range(1,user_num+1):
            for j in range(i+1,user_num):
                num = 0
                for k in range(num_hash):
                    if user_minhash_matrix[k][i] == user_minhash_matrix[k][j]:
                        num += 1
                user_similarity_matrix[i][j] = user_similarity_matrix[j][i] = num/num_hash

        np.save('temp/user_similar_2.npy', user_similarity_matrix)
        return user_similarity_matrix


def get_content_similar():
    df = pd.read_csv('anime.csv')
    anime_ids = df['Anime_id'].tolist()
    try:
        content_similarity_matrix = np.load('temp/content_similar_2.npy'), anime_ids
        return content_similarity_matrix
    except:
        # 读取数据集
        df = pd.read_csv('anime.csv')
        # 提取动漫类别作为特征值
        genres = df['Genres'].str.split(', ').tolist()
        all_genres = list(set([genre for genre_list in genres for genre in genre_list]))
        num_genres = len(all_genres)
        num_anime = len(df)
        genre_map = dict()
        for i in range(num_genres):
            genre_map[all_genres[i]] = i

        feature_matrix_01= np.zeros((num_anime,num_genres))
        index = 0
        for genre_list in genres:
            for genre in genre_list:
                feature_matrix_01[index][genre_map[genre]]=1
            index += 1

        all_genres_count = [0] * len(all_genres)
        for genre_list in genres:
            for genre in genre_list:
                all_genres_count[genre_map[genre]] += 1

        num_hash = 20
        hash_matrix = np.zeros((num_genres,num_hash),dtype=int)
        hash_functions = generate_affine_hash_functions(num_genres,num_hash)
        for i in range(num_genres):
            for j in range(num_hash):
                hash_matrix[i][j] = hash_functions[j](i)

        content_minhash_matrix = np.zeros((num_hash,num_anime))
        for i in range(num_anime):
            temp_min_array = [random.randint(100001, sys.maxsize)]*num_hash
            for j in range(num_genres):
                if feature_matrix_01[i][j] == 1:
                    for k in range(num_hash):
                        temp_min_array[k] = min(hash_matrix[j][k], temp_min_array[k])
            for j in range(num_hash):
                content_minhash_matrix[j][i] = temp_min_array[j]

        # 计算动漫之间的相似度矩阵
        content_similarity_matrix = np.eye(num_anime, dtype=float)
        for i in range(num_anime):
            for j in range(i+1,num_anime):
                num = 0
                for k in range(num_hash):
                    if content_minhash_matrix[k][i] == content_minhash_matrix[k][j]:
                        num += 1
                content_similarity_matrix[i][j] = content_similarity_matrix[j][i] = num/num_hash
            if i % 1000 == 0:
                print(f"已完成{i / num_anime * 100:.2f}%")
        # 存储相似度矩阵
        np.save('temp/content_similar_2.npy', content_similarity_matrix)
        return content_similarity_matrix, anime_ids


def get_user_predict_test(user_similarity_matrix: np.ndarray, k=150):
    # 读取测试集数据
    user_id = []
    anime_id = []
    rating = []
    with open('test_set.csv', 'r') as f:
        next(f)  # 跳过标题行
        for line in f:
            user, anime, rate = line.strip().split(',')
            user_id.append(int(user))
            anime_id.append(int(anime))
            rating.append(int(rate))

    predictions = {}
    SSE = 0
    for user_id, anime_id, rating in zip(user_id, anime_id, rating):
        # 找到与当前用户最相似的 k 个用户
        similar_users = list(np.sort(user_similarity_matrix[user_id], kind='quicksort')[::-1][:k])
        sorted_indices = list(np.argsort(user_similarity_matrix[user_id])[::-1][:k])
        # 计算预测评分
        weighted_sum = 0
        similarity_sum = 0
        for similar_user, similarity in zip(sorted_indices, similar_users):
            if anime_id in user_anime_matrix[similar_user]:
                weighted_sum += similarity * user_anime_matrix[similar_user][anime_id]
                similarity_sum += similarity
        if similarity_sum > 0:
            predictions[(user_id, anime_id)] = round(weighted_sum / similarity_sum)
        else:
            predictions[(user_id, anime_id)] = 0
        SSE += (predictions[(user_id, anime_id)] - rating) ** 2

    return predictions, SSE


def get_user_predict(user_id, user_similarity_matrix: np.ndarray, anime_id_all, n=20, k=150):
    # 找到与当前用户最相似的 k 个用户
    a = user_similarity_matrix[user_id]
    similar_users = list(np.sort(user_similarity_matrix[user_id], kind='quicksort')[::-1][:k])
    sorted_indices = list(np.argsort(user_similarity_matrix[user_id])[::-1][:k])
    predictions = {}
    for anime_id in anime_id_all:
        # 计算预测评分
        weighted_sum = 0
        similarity_sum = 0
        for similar_user, similarity in zip(sorted_indices, similar_users):
            if anime_id in user_anime_matrix[similar_user]:
                weighted_sum += similarity * user_anime_matrix[similar_user][anime_id]
                similarity_sum += similarity
        if similarity_sum > 0:
            predictions[anime_id] = weighted_sum / similarity_sum
        else:
            predictions[anime_id] = 0
    return dict(sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:n])


def get_content_predict_test(content_similarity_matrix: np.ndarray, user_anime_matrix: np.ndarray, anime_ids):
    # 读取测试集数据
    user_id = []
    anime_id = []
    rating = []
    with open('test_set.csv', 'r') as f:
        next(f)  # 跳过标题行
        for line in f:
            user, anime, rate = line.strip().split(',')
            user_id.append(int(user))
            anime_id.append(int(anime))
            rating.append(int(rate))

    predictions = {}
    SSE = 0
    for user_id, anime_id, rating in zip(user_id, anime_id, rating):
        # 获取当前用户已打分的动漫的集合
        anime_has_rated = user_anime_matrix[user_id]
        anime_sims = list(content_similarity_matrix[anime_ids.index(anime_id)])
        weighted_sum = 0
        similarity_sum = 0
        for key, value in anime_has_rated.items():
            anime_sim = anime_sims[anime_ids.index(key)]
            if anime_sim > 0:
                similarity_sum += anime_sim
                weighted_sum += anime_sim * value
        if similarity_sum > 0:
            predictions[(user_id, anime_id)] = round(weighted_sum / similarity_sum)
        else:
            predictions[(user_id, anime_id)] = 0
        SSE += (predictions[(user_id, anime_id)] - rating) ** 2
    return predictions, SSE


def get_content_predict(user_id, content_similarity_matrix: np.ndarray, user_anime_matrix: np.ndarray, anime_ids,
                        n=20):
    predictions = {}
    # 获取当前用户已打分的动漫的集合
    anime_has_rated = user_anime_matrix[user_id].items()

    # 获取这些已打分动漫的相似度集合
    anime_sims = dict()
    for anime_id, rating in anime_has_rated:
        anime_sims[anime_id] = list(content_similarity_matrix[anime_ids.index(anime_id)])
    index = 0
    total_num = len(anime_ids)
    for anime_id in list(anime_ids):
        if anime_id in anime_has_rated:
            continue
        weighted_sum = 0
        similarity_sum = 0
        for key, value in anime_has_rated:
            anime_sim = anime_sims[key][index]
            if anime_sim > 0:
                similarity_sum += anime_sim
                weighted_sum += anime_sim * value
        if similarity_sum > 0:
            predictions[anime_id] = weighted_sum / similarity_sum
        else:
            predictions[anime_id] = 0
        index += 1
        if index % 1000 == 0:
            print(f"已完成{index / total_num * 100:.2f}%")
    return dict(sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:n])


def get_train_set():
    # 读取训练集数据
    user_id = []
    anime_id = []
    rating = []
    with open('train_set.csv', 'r') as f:
        next(f)  # 跳过标题行
        for line in f:
            user, anime, rate = line.strip().split(',')
            user_id.append(int(user))
            anime_id.append(int(anime))
            rating.append(int(rate))
    anime_id_all = set(anime_id)
    user_id_all = set(user_id)
    # 构建用户-动画效用矩阵
    user_anime_matrix = defaultdict(dict)
    for u, a, r in zip(user_id, anime_id, rating):
        user_anime_matrix[u][a] = r
    return user_anime_matrix, anime_id_all, user_id_all


if __name__ == '__main__':

    command = 0
    user_anime_matrix, anime_id_all, user_id_all = get_train_set()

    while command != -1:

        print("请选择功能：\n1：计算协同过滤误差\n2：基于协同过滤推荐\n3：计算内容推荐误差\n4：基于内容推荐\n-1：退出")
        command = int(input())
        if command == 1:
            user_similarity_matrix = get_user_similar(user_anime_matrix, anime_id_all, user_id_all)
            pre, SSE = get_user_predict_test(user_similarity_matrix)
            for k, v in pre.items():
                print(f"{k} -> {v}")
            print(f"SSE误差：{SSE}")
        elif command == 2:
            user_id = int(input("请输入用户id"))
            user_similarity_matrix = get_user_similar(user_anime_matrix, anime_id_all, user_id_all)
            predictions = get_user_predict(user_id, user_similarity_matrix, anime_id_all)
            for k, v in predictions.items():
                print(f"{k} -> {v:.2f}")
        elif command == 3:
            content_similarity_matrix, anime_ids = get_content_similar()
            pre, SSE = get_content_predict_test(content_similarity_matrix, user_anime_matrix, anime_ids)
            for k, v in pre.items():
                print(f"{k} -> {v}")
            print(f"SSE误差：{SSE}")
        elif command == 4:
            user_id = int(input("请输入用户id"))
            content_similarity_matrix, anime_ids = get_content_similar()
            predictions = get_content_predict(user_id, content_similarity_matrix, user_anime_matrix, anime_ids)
            for k, v in predictions.items():
                print(f"{k} -> {v:.2f}")
