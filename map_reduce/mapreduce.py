import json
import os
import re
import time
from collections import defaultdict
import threading


def input_splitter(file_path):
    words = []
    pattern = r"\b[a-zA-Z]+\b"
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            words += re.findall(pattern, line)
    return words


class MapNode(threading.Thread):
    def __init__(self, folder_path, use_words: set, num, shufflers: [],
                 shufflers_lock: threading.Lock):
        super(MapNode, self).__init__()
        self.shufflers = shufflers
        self.shufflers_lock = shufflers_lock
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
        output_to_file(f'result/map{self.num}.json', result)
        end_time = time.time()
        running_time = end_time - start_time
        print(f'map{self.num}运行时间：{running_time}秒')

    def shuffle(self, result: dict):
        with self.shufflers_lock:
            for key, value in result.items():
                target_shuffler_num = hash(key) % 3
                shuffler = self.shufflers[target_shuffler_num]
                if key in shuffler:
                    now_value = shuffler[key]
                    set1 = now_value[0]
                    set2 = value[0]
                    merged_set = set1.union(set2)
                    shuffler[key] = (merged_set, now_value[1] + value[1])
                else:
                    shuffler[key] = value

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
            words = input_splitter(file_path)
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
        with open(f'result/shuffler{self.num}.json', 'r') as file_r:
            with open(f'result/reduce{self.num}.json', 'w') as file_w:
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
    shufflers = [defaultdict(), defaultdict(), defaultdict()]
    shufflers_lock = threading.Lock()
    for i in range(1, 4):
        thread = MapNode(folder_path=f'source_data/folder_{i}', num=i, use_words=use_words,
                         shufflers=shufflers, shufflers_lock=shufflers_lock)
        thread.start()
        threads.append(thread)

    for i in range(4, 7):
        thread = MapNode(folder_path=f'source_data/folder_{i}', num=i, use_words=use_words,
                         shufflers=shufflers, shufflers_lock=shufflers_lock)
        thread.start()
        threads.append(thread)

    for i in range(7, 10):
        thread = MapNode(folder_path=f'source_data/folder_{i}', num=i, use_words=use_words,
                         shufflers=shufflers, shufflers_lock=shufflers_lock)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    output_to_file('result/shuffler1.json', shufflers[0])
    output_to_file('result/shuffler2.json', shufflers[1])
    output_to_file('result/shuffler3.json', shufflers[2])

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


def reduce_combiner():
    reduce_result = {}
    for i in range(1, 4):
        with open(f'result/reduce{i}.json') as file:
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
    new_titles = {}
    for title, linked_titles in top_1000.items():
        count = linked_titles[1]
        linked_titles = linked_titles[0]
        linked_titles = [key for key in linked_titles if key in top_1000.keys()]
        new_titles[title] = (linked_titles, count)
    top_1000 = new_titles
    output_to_file('result/reduce.json', top_1000)
    with open('result/reduce.txt', 'w') as file:
        for key, value in top_1000.items():
            file.write(f"{key} : {value[1]}\n")


if __name__ == "__main__":
    start_time = time.time()
    use_words = read_word_set_from_file('words.txt')
    map_reduce_function(use_words)
    reduce_combiner()
    end_time = time.time()
    running_time = end_time - start_time
    print(f'总运行时间：{running_time}秒')
    # result = list(map_reduce_function(words, num_threads=4))
    # for i in result:
    #     print(i)
