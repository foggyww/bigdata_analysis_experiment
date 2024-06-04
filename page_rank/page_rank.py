import json
import time

import numpy as np


def build_graph(titles: dict):
    keys = titles.keys()

    new_titles = {}

    for title, linked_titles in titles.items():
        linked_titles = linked_titles[0]
        linked_titles = [key for key in linked_titles if key in keys]
        new_titles[title] = linked_titles

    titles = new_titles
    graph = {}
    for title, linked_titles in titles.items():
        if title not in graph:
            graph[title] = set()
        for linked_title in linked_titles:
            if linked_title not in graph:
                graph[linked_title] = set()
            graph[linked_title].add(title)
    return graph


def pagerank(graph, beta=0.85, epsilon=1e-8):
    nodes = list(graph.keys())
    n = len(nodes)
    M = np.zeros((n, n))

    node_index = {node: i for i, node in enumerate(nodes)}

    for node, out_links in graph.items():
        if len(out_links) == 0:
            continue
        else:
            for out_link in out_links:
                M[node_index[out_link], node_index[node]] = 1.0 / len(out_links)

    R = np.ones(n) / n
    teleport = np.ones(n) / n
    while True:
        R_new = beta * np.dot(M, R) + (1 - beta) * teleport
        if abs(np.sum(R_new)-np.sum(R)) < epsilon:
            break
        R = R_new
    pagerank_values = {nodes[i]: R[i] for i in range(n)}
    return pagerank_values


if __name__ == "__main__":
    with open('../map_reduce/result/reduce.json', 'r') as file:
        start_time = time.time()
        titles = json.load(file)
        graph = build_graph(titles)
        pagerank_values = pagerank(graph)
        pagerank_values = dict(sorted(pagerank_values.items(), key=lambda x: x[1],reverse=True))
        sum = 0
        with open("result/page_rank.txt", "w") as file:
            for key, value in pagerank_values.items():
                print(f"{key} : {value}")
                sum += value
                file.write(f"{key} : {value}\n")
            file.close()
        end_time = time.time()
        print(f"用时：{end_time - start_time}秒,大小：{len(pagerank_values.keys())}")
        print(f"sum：{sum}")
