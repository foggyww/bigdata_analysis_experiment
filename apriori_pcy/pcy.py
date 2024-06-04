import pickle
from itertools import combinations
from collections import defaultdict

min_support_ratio = 0.15
min_confidence = 0.3


def get_support(item_set, transactions):
    return sum(1 for transaction in transactions if set(item_set).issubset(set(transaction))) / len(transactions)


def generate_rules(frequent_item_sets, transactions):
    rules = []
    for item_set in frequent_item_sets:
        if len(item_set) < 2:
            continue
        for subset_length in range(1, len(item_set)):
            for subset in combinations(item_set, subset_length):
                # 前项
                antecedent = set(subset)
                # 后项
                consequent = item_set - antecedent
                support = get_support(item_set, transactions)
                confidence = support / get_support(antecedent, transactions)
                if confidence >= min_confidence:
                    rules.append((set(antecedent), set(consequent), support, confidence))
    return rules


def pcy(transactions):
    single_items = set(item for transaction in transactions for item in transaction)
    current_item_sets = set(frozenset([item]) for item in single_items)
    frequent_item_sets = []
    k = 1
    min_support = min_support_ratio
    min_support_num = min_support * len(single_items)
    while current_item_sets:
        # 扫描筛选
        current_item_sets = \
            set([item_set for item_set in current_item_sets if get_support(item_set, transactions) >= min_support])
        frequent_item_sets.extend(current_item_sets)
        # 生成哈希表
        hash_table = [0] * 2000

        for transaction in transactions:
            result = set()
            for item_set in current_item_sets:
                r = {item for item in item_set if item in transaction}
                result = result.union(r)
            for subset in combinations(result, k + 1):
                subset = frozenset(subset)
                hash_value = hash(subset) % 2000
                hash_table[hash_value] += 1
        new_item_sets = set()
        for i in current_item_sets:
            for j in current_item_sets:
                set1 = i
                set2 = j
                now_items = set1.union(set2)
                if len(now_items) != k + 1:
                    continue
                hash_value = hash(now_items) % 2000
                if hash_table[hash_value] >= min_support_num:
                    new_item_sets.add(now_items)
        current_item_sets = new_item_sets
        k += 1
        if k > 4:
            break

    rules = generate_rules(frequent_item_sets, transactions)

    return frequent_item_sets, rules


if __name__ == '__main__':
    titles_input = []
    with open("top_keywords.pkl", "rb") as file:
        new_input = pickle.load(file)
        for key, values in new_input[1].items():
            values.add(key)
            new_list = values
            titles_input.append(new_list)
        file.close()

    frequent_item_sets, rules = pcy(titles_input)
    print("Frequent Itemsets and their Support:")
    for itemset in frequent_item_sets:
        support = get_support(itemset, titles_input)
        print(f"{itemset}: {support}")

    print("\nAssociation Rules and their Confidence:")
    for antecedent, consequent, support, confidence in rules:
        print(f"{antecedent} => {consequent} (Support: {support}, Confidence: {confidence})")

    print("\nNumber of Frequent Itemsets by size:")
    size_counts = defaultdict(int)
    for itemset in frequent_item_sets:
        size_counts[len(itemset)] += 1
    for size, count in size_counts.items():
        print(f"{size}-itemsets: {count}")

    print(f"\nTotal number of association rules: {len(rules)}")
