import os
import pandas as pd
from pydriller import RepositoryMining
from xml.etree import ElementTree
from tokenizer import TokeNizer
from matplotlib import pyplot as plt
from get_entropy import *


REPOSITORIES_FOLDER = "data/"


def extract_dependencies(pom_file: str):
    pom_str = list()

    with open(pom_file) as f:
        while True:
            line = f.readline()
            if not line:
                break
            else:
                pom_str.append(line.strip())

    dependencies = list()

    tree = ElementTree.fromstring("\n".join(pom_str))

    # Analyze properties of the POM file
    for child in tree:
        child_tag = child.tag
        child_tag = child_tag.split('}')[-1]

        if child_tag == "dependencies":
            for dependency in child:
                dependency_sentence = ''

                for attribute in dependency:
                    dependency_sentence += attribute.text + "|"

                dependencies.append(dependency_sentence[:-1])

    return set(dependencies)


def checkout_previous_version(project_path, hash_number):
    os.system(f"cd {project_path} && git checkout {hash_number} > /dev/null 2>&1")


def analyse_java_files(project_path, number):
    project_name = project_path.split("/")[1]
    code_files = list()

    for root, folders, files in os.walk(project_path):
        for file in files:
            if file.endswith('.java'):
                token_procedure = TokeNizer("Java")
                code = code_file_to_str(os.path.join(root, file))

                tokens = ' '.join(token_procedure.get_pure_tokens(code))
                code_files.append(tokens)

    path_folder = f"naturalness-data/java/new_data/{project_name}/"
    if not os.path.exists(path_folder):
        os.mkdir(path_folder)

    with open(f"{path_folder}/fold{number}.train", "w") as f:
        for file_code in code_files:
            f.writelines(file_code + "\n")


def code_file_to_str(path_file: str):
    lines_code = list()

    with open(path_file) as f:
        while True:
            line = f.readline()
            if not line:
                break
            else:
                line = line.strip()
                if not line.count("*"):
                    lines_code.append(line)
    return ' '.join(lines_code)


def modification_to_str(modification: str):
    if not modification:
        return ""

    lines_code = list()
    modification_lines = modification.split("\n")

    for mod in modification_lines:
        line = mod.strip()
        if not line.count("*"):
            lines_code.append(line)

    return ' '.join(lines_code)


def found_match_update(library: str, list_libraries: list):
    updated_libraries = list()
    group_artifact = "|".join(library.split("|")[:2])

    for lib in list_libraries:
        if lib.startswith(group_artifact):
            updated_libraries.append(lib)

    if not len(updated_libraries):
        group = library.split("|")[0]

        for lib in list_libraries:
            if lib.startswith(group):
                updated_libraries.append(lib)

    return updated_libraries


def analyse_dependencies_changes(project_path: str):
    hash_list = list()
    dependencies_history = list()
    commit_counter = 0
    pom_counter = 0
    dependency_counter = 0

    for commit in RepositoryMining(project_path).traverse_commits():
        hash_list.append(commit.hash)
        commit_counter += 1

        added_value = 0
        removed_value = 0

        for modification in commit.modifications:
            if modification.filename == "pom.xml":
                if modification.new_path:
                    path = modification.new_path
                else:
                    path = modification.old_path

                if pom_counter < 1:
                    checkout_previous_version(project_path, commit.hash)
                    dependencies_history.append(extract_dependencies(project_path + "/" + path))
                else:
                    checkout_previous_version(project_path, commit.hash)
                    previous_dependencies = dependencies_history[pom_counter - 1]
                    current_dependencies = extract_dependencies(project_path + "/" + path)

                    removed_libraries = previous_dependencies.difference(current_dependencies)
                    added_libraries = current_dependencies.difference(previous_dependencies)

                    dependencies_history.append(current_dependencies)

                    # Checking for only changes in the dependencies
                    if len(removed_libraries) or len(added_libraries):
                        dependency_counter += 1

                        # Make the trains files with the snapshot before
                        checkout_previous_version(project_path, hash_list[commit_counter - 1])
                        analyse_java_files(project_path, dependency_counter - 1)

                        # Make the test files with the files changed
                        checkout_previous_version(project_path, commit.hash)
                        code_files = list()

                        for mod in commit.modifications:
                            if mod.filename.endswith(".java"):
                                removed_value += mod.removed
                                added_value += mod.added
                                source_preprocessed = preprocess_code(mod.source_code)

                                token_procedure = TokeNizer("Java")
                                code = modification_to_str(source_preprocessed)

                                tokens = ' '.join(token_procedure.get_pure_tokens(code))
                                code_files.append(tokens)

                        with open(f"naturalness-data/java/new_data/{project_path.split('/')[1]}/fold{dependency_counter - 1}.test", "w") as f:
                            for file_code in code_files:
                                f.writelines(file_code + "\n")

                pom_counter += 1
    print(f"Commits: {commit_counter}, POM Changes: {pom_counter}, Dependencies changed: {dependency_counter}")
    restore_to_latest_commit(project_path)
    return dependency_counter


def restore_to_latest_commit(project_path: str):
    os.system(f"cd {project_path} && git checkout master > /dev/null 2>&1")


def preprocess_code(code: str):
    code_split = code.split("\n") if code else " "
    code_filtered = list()

    for line in code_split:
        line = line.strip()

        if not line.startswith("import ") and not line.startswith("package ") and not line.count("@"):
            code_filtered.append(line)

    return "\n".join(code_filtered)


def get_entropy_commit(data_path: str, stopwords: list, number_commits: int):
    print(f"Getting entropy values ...")

    entropy_unigram_list = list()
    entropy_bigram_list = list()
    entropy_trigram_list = list()

    for i in range(number_commits):
        train_file = data_path + "/" + f"fold{i}.train"
        test_file = data_path + "/" + f"fold{i}.test"

        data_train, data_test = extract_data(train_file, test_file)

        # Cleaning input
        data_train = preprocess_data(data_train, stopwords)
        data_test = preprocess_data(data_test, stopwords)

        probabilities_unigram = get_probabilities_unigram(data_train)
        entropy_unigram = get_entropy_unigram(data_test, data_train, probabilities_unigram)
        entropy_unigram_list.append(entropy_unigram)

        keys_bigram, probabilities_bigram = get_probabilities_bigram(data_train)
        entropy_bigram = get_entropy_bigram(data_test,
                                            data_train,
                                            keys_bigram,
                                            probabilities_unigram,
                                            probabilities_bigram)
        entropy_bigram_list.append(entropy_bigram)

        keys_trigram, probabilities_trigram = get_probabilities_trigram(data_train)
        entropy_trigram = get_entropy_trigram(data_test,
                                              data_train,
                                              keys_trigram,
                                              keys_bigram,
                                              probabilities_unigram,
                                              probabilities_bigram,
                                              probabilities_trigram)

        entropy_trigram_list.append(entropy_trigram)

    return entropy_unigram_list, entropy_bigram_list, entropy_trigram_list


def get_reserved_words():
    reserved_words = list()

    with open("java_words.txt") as f:
        while True:
            line = f.readline()
            if not line:
                break
            else:
                line = line.strip()
                reserved_words.append(line)

    return reserved_words


def plot_trigrams(results_path: str, project_name: str):
    dataframe = pd.read_csv(results_path)

    if (dataframe["trigram_values"] > 10).any():
        max_lim = max(dataframe["trigram_values"]) + 1
    else:
        max_lim = 10

    plt.plot(range(len(dataframe)), dataframe["trigram_values"], ".g-", label="Trigram Model")

    plt.xlabel("Commits")
    plt.ylabel("Entropy")
    plt.ylim([0, max_lim])

    plt.legend()
    plt.title(project_name)
    plt.show()


def save_results_csv(project_path: str, unigram_values: list, bigram_values: list, trigram_values: list):
    print("Saving the results ...")
    results_folder = "results/entropy/java/"

    data = {'unigram_values': unigram_values,
            'bigram_values': bigram_values,
            'trigram_values': trigram_values}

    dataframe = pd.DataFrame(data=data)

    dataframe.to_csv(results_folder + f"{project_path}.csv")


if __name__ == '__main__':
    projects_poms = list()

    with open(f'{REPOSITORIES_FOLDER}/remaining.txt') as f:
        while True:
            line = f.readline()
            if not line:
                break
            else:
                line = line.strip()
                projects_poms.append(line)

    reserved = get_reserved_words()

    for project_location in projects_poms:
        project_name = project_location.split("/")[1]
        print(f"Analysing project {project_name} ...")

        commits = analyse_dependencies_changes(project_location)
        unigram_list, bigram_list, trigram_list = get_entropy_commit(f"naturalness-data/java/new_data/{project_name}",
                                                                     reserved,
                                                                     commits)
        save_results_csv(project_name, unigram_list, bigram_list, trigram_list)

    # Plot only the trigram model
    for project_location in projects_poms:
        project_name = project_location.split("/")[1]
        plot_trigrams(f"results/entropy/java/{project_name}.csv", project_name)
