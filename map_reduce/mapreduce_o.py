import json
import os
import re
import time
from collections import defaultdict
import threading


class MapNode(threading.Thread):
    def __init__(self, folder_path, use_words: set, num, shuffler: defaultdict,
                 shuffler_lock: threading.Lock):
        super(MapNode, self).__init__()
        self.shuffler = shuffler
        self.shuffler_lock = shuffler_lock
        self.folder_path = folder_path
        self.num = num
        self.use_words = use_words

    def run(self):
        start_time = time.time()
        tuples = MapNode.mapper(self.folder_path, self.use_words)
        end_time = time.time()
        running_time = end_time - start_time
        print(f'mapper{self.num}运行时间：{running_time}秒')

        now_time = time.time()
        result = MapNode.combiner(tuples)
        end_time = time.time()
        running_time = end_time - now_time
        print(f'combine{self.num}运行时间：{running_time}秒')
        now_time = time.time()
        self.shuffle(result)
        end_time = time.time()
        running_time = end_time - now_time
        print(f'shuffle{self.num}运行时间：{running_time}秒')
        output_to_file(f'map_reduce/result/map{self.num}.json', result)
        end_time = time.time()
        running_time = end_time - start_time
        print(f'map{self.num}运行时间：{running_time}秒')

    @staticmethod
    def read_words_from_file(file_path):
        words = []
        pattern = r"\b[a-zA-Z]+\b"
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                words += re.findall(pattern, line)
        return words

    def shuffle(self, result: dict):
        with self.shuffler_lock:
            for key, value in result.items():
                if key in self.shuffler:
                    now_value = self.shuffler[key]
                    set1 = now_value[0]
                    set2 = value[0]
                    merged_set = set1.union(set2)
                    self.shuffler[key] = (merged_set, now_value[1] + value[1])
                else:
                    self.shuffler[key] = value

    @staticmethod
    def combiner(tuples: list):
        result = {}
        for m_tuple in tuples:
            key = m_tuple[0]
            if key in result:
                now_value = result[key]
                now_value[0].add(m_tuple[1])
                result[key] = (now_value[0], now_value[1] + 1)
            else:
                m_set = set()
                m_set.add(m_tuple[1])
                result[key] = (m_set, 1)
        return result

    @staticmethod
    def mapper(folder_path, use_words: set):
        file_list = []
        tuples = []
        for root, directories, files in os.walk(folder_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                file_list.append(file_path)
        for file_path in file_list:
            # 获取文件名
            file_name_with_extension = os.path.basename(file_path)
            # 去掉文件后缀
            file_name, file_extension = os.path.splitext(file_name_with_extension)
            words = MapNode.read_words_from_file(file_path)
            for word in words:
                if use_words.__contains__(word):
                    tuples.append((word, file_name))
        return tuples


class ReduceNode(threading.Thread):
    def __init__(self, num):
        super(ReduceNode, self).__init__()
        self.num = num

    def run(self):
        start_time = time.time()
        with open(f'map_reduce/result/shuffler{self.num}.json', 'r') as file_r:
            with open(f'map_reduce/result/reduce{self.num}.json', 'w') as file_w:
                shuffler = json.load(file_r)
                sorted_result = dict(sorted(shuffler.items(), key=lambda value: value[1][1], reverse=True))
                top_1000 = dict(list(sorted_result.items())[:1000])
                json.dump(top_1000, file_w)
                file_r.close()
                file_w.close()
        end_time = time.time()
        running_time = end_time - start_time
        print(f'reduce{self.num}运行时间：{running_time}秒')


# 输出dict(string , tuple( set<string>, int))到文件
def output_to_file(file_path, result):
    output = {}
    for key, value in result.items():
        output[key] = (tuple(value[0]), value[1])
    with open(file_path, 'w') as file:
        json.dump(output, file)
        file.close()


def map_reduce_function(use_words: set):
    threads = []
    shuffler1 = defaultdict()
    shuffler1_lock = threading.Lock()
    shuffler2 = defaultdict()
    shuffler2_lock = threading.Lock()
    shuffler3 = defaultdict()
    shuffler3_lock = threading.Lock()
    for i in range(1, 4):
        thread = MapNode(folder_path=f'map_reduce/source_data/folder_{i}', num=i, use_words=use_words,
                         shuffler=shuffler1, shuffler_lock=shuffler1_lock)
        thread.start()
        threads.append(thread)

    for i in range(4, 7):
        thread = MapNode(folder_path=f'map_reduce/source_data/folder_{i}', num=i, use_words=use_words,
                         shuffler=shuffler2, shuffler_lock=shuffler2_lock)
        thread.start()
        threads.append(thread)

    for i in range(7, 10):
        thread = MapNode(folder_path=f'map_reduce/source_data/folder_{i}', num=i, use_words=use_words,
                         shuffler=shuffler3, shuffler_lock=shuffler3_lock)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    output_to_file('map_reduce/result/shuffler1.json', shuffler1)
    output_to_file('map_reduce/result/shuffler2.json', shuffler2)
    output_to_file('map_reduce/result/shuffler3.json', shuffler3)

    threads.clear()

    thread = ReduceNode(1)
    thread.start()
    threads.append(thread)

    thread = ReduceNode(2)
    thread.start()
    threads.append(thread)

    thread = ReduceNode(3)
    thread.start()
    threads.append(thread)

    for thread in threads:
        thread.join()


def read_word_set_from_file(file_path):
    words = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            words += line.split()
    return set(words)


def combine_reduce():
    reduce_result = {}
    for i in range(1, 4):
        with open(f'map_reduce/result/reduce{i}.json') as file:
            now_reduce = json.load(file)
            for key, value in now_reduce.items():
                set1 = set(now_reduce[key][0])
                if key in reduce_result:
                    set2 = reduce_result[key][0]
                    merged_set = set1.union(set2)
                    reduce_result[key] = (merged_set, reduce_result[key][1] + value[1])
                else:
                    reduce_result[key] = (set1, value[1])
    sorted_result = dict(sorted(reduce_result.items(), key=lambda value: value[1][1], reverse=True))
    top_1000 = dict(list(sorted_result.items())[:1000])
    output_to_file('map_reduce/result/reduce.json',top_1000)
    with open('map_reduce/result/reduce.txt', 'w') as file:
        for key, value in top_1000.items():
            file.write(f"{key} : {value[1]}\n")


if __name__ == "__main__":
    start_time = time.time()
    use_words = read_word_set_from_file('map_reduce/words.txt')
    map_reduce_function(use_words)
    combine_reduce()
    end_time = time.time()
    running_time = end_time - start_time
    print(f'总运行时间：{running_time}秒')
    # result = list(map_reduce_function(words, num_threads=4))
    # for i in result:
    #     print(i)
