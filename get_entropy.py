import time
import numpy as np
from collections import Counter


def extract_data(file_train_path: str, file_test_path: str):
    train = list()
    test = list()

    with open(file_train_path) as file_train:
        while True:
            line_train = file_train.readline()
            if not line_train:
                break
            else:
                line_train = line_train.strip()
                train.append(line_train)

    with open(file_test_path) as file_test:
        while True:
            line_test = file_test.readline()
            if not line_test:
                break
            else:
                line_test = line_test.strip()
                test.append(line_test)

    return train, test


def preprocess_data(data: list, stopwords: list):
    processed_data = list()

    for item in data:
        tokens = item.split()
        processed = list()

        for token in tokens:
            if token not in stopwords and token.isalpha():
                processed.append(token)
        processed_data.append(" ".join(processed))

    return processed_data


def get_probabilities_unigram(data: list):
    data_join = " ".join(data)
    data_splitted = data_join.split()

    counter_names = dict(Counter(data_splitted))
    keys = list(counter_names.keys())
    total = sum(counter_names.values())

    probabilities = list(map(lambda number: number / total, counter_names.values()))
    probabilities_words = list(map(lambda word: probabilities[keys.index(word)], data_splitted))

    return probabilities_words


def get_entropy_unigram(sentence: list, data: list, probabilities: list):
    data_join = " ".join(data)
    data_splitted = data_join.split()

    sentence_join = " ".join(sentence)
    sentence_splitted = sentence_join.split()

    entropy_values = list()

    for token in sentence_splitted:
        count = data_splitted.count(token)
        if count:
            probability = probabilities[data_splitted.index(token)]
            entropy_values.append(probability * np.log2(probability))
        else:
            entropy_values.append(0)

    return -1 * sum(entropy_values)


def get_probabilities_bigram(data: list):
    keys = list()
    probabilities = list()

    for items in data:
        tokens = items.split()
        for i in range(len(tokens) - 1):
            couple = tokens[i] + " " + tokens[i + 1]
            keys.append(couple)

    counter_keys = dict(Counter(keys))

    data_join = " ".join(data)
    data_all_split = data_join.split()
    counter_names = dict(Counter(data_all_split))

    for key, value in counter_keys.items():
        probability = value / counter_names[key.split()[0]]
        probabilities.append(probability)

    return list(counter_keys.keys()), probabilities


def get_indices(keys: list, token: str):
    indices = list()

    for i in range(len(keys)):
        key = keys[i]
        first_half = key.split()[0]
        if first_half == token:
            indices.append(i)
    return indices


def get_entropy_bigram(sentence: list,
                       data: list,
                       keys: list,
                       prob_unigram: list,
                       prob_bigram: list):
    data_join = " ".join(data)
    data_all_split = data_join.split()

    entropy_values = list()

    for items in sentence:
        tokens = items.split()
        for i in range(len(tokens) - 1):
            couple = tokens[i] + " " + tokens[i + 1]
            if couple in keys:
                probability_unigram = prob_unigram[data_all_split.index(tokens[i])]

                indices = get_indices(keys, tokens[i])
                probs_bigram = np.array([prob_bigram[index] for index in indices])

                partial_result = probability_unigram * sum(probs_bigram * np.log2(probs_bigram))
                entropy_values.append(partial_result)
            else:
                entropy_values.append(0)
    return -1 * sum(entropy_values)


def get_probabilities_trigram(data: list):
    keys = list()
    probabilities = list()

    for items in data:
        tokens = items.split()
        for i in range(len(tokens) - 2):
            triple = tokens[i] + " " + tokens[i + 1] + " " + tokens[i + 2]
            keys.append(triple)

    counter_keys = dict(Counter(keys))

    data_join = " ".join(data)
    data_all_split = data_join.split()
    counter_names = dict(Counter(data_all_split))

    for key, value in counter_keys.items():
        probability = value / counter_names[key.split()[0]]
        probabilities.append(probability)

    return list(counter_keys.keys()), probabilities


def get_entropy_trigram(sentence: list,
                        data: list,
                        keys: list,
                        keys_bigram: list,
                        probabilities_unigram: list,
                        probabilities_bigram: list,
                        probabilities_trigram: list):

    data_join = " ".join(data)
    data_all_split = data_join.split()

    entropy_values = list()

    for items in sentence:
        tokens = items.split()
        for i in range(len(tokens) - 2):
            triple = tokens[i] + " " + tokens[i + 1] + " " + tokens[i + 2]
            if triple in keys:
                probability_unigram = probabilities_unigram[data_all_split.index(tokens[i])]
                probability_bigram = probabilities_bigram[keys_bigram.index(tokens[i] + " " + tokens[i + 1])]

                indices = get_indices(keys, tokens[i])
                probs_trigram = np.array([probabilities_trigram[index] for index in indices])

                partial_result = probability_unigram * probability_bigram * sum(probs_trigram * np.log2(probs_trigram))
                entropy_values.append(partial_result)
            else:
                entropy_values.append(0)
    return -1 * sum(entropy_values)


if __name__ == '__main__':
    start = time.time()

    data_train, data_test = extract_data("fold436.train", "fold436.test")

    reserved_words = list()

    with open("java_words.txt") as f:
        while True:
            line = f.readline()
            if not line:
                break
            else:
                line = line.strip()
                reserved_words.append(line)

    data_train = preprocess_data(data_train, reserved_words)
    data_test = preprocess_data(data_test, reserved_words)

    probabilities_unigram = get_probabilities_unigram(data_train)
    entropy_unigram = get_entropy_unigram(data_test, data_train, probabilities_unigram)

    print(f"Entropy value for the unigram model: {entropy_unigram}")

    keys_bigram, probabilities_bigram = get_probabilities_bigram(data_train)
    entropy_bigram = get_entropy_bigram(data_test, data_train, keys_bigram, probabilities_unigram, probabilities_bigram)

    print(f"Entropy value for the bigram model: {entropy_bigram}")

    keys_trigram, probabilities_trigram = get_probabilities_trigram(data_train)
    entropy_trigram = get_entropy_trigram(data_test,
                                          data_train,
                                          keys_trigram,
                                          keys_bigram,
                                          probabilities_unigram,
                                          probabilities_bigram,
                                          probabilities_trigram)

    print(f"Entropy value for the trigram model: {entropy_trigram}")
    end = time.time()

    print(f"\nExecution time: {(end - start)}")
