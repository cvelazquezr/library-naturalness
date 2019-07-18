import os
import git
import pandas as pd
from shutil import rmtree
from pydriller import RepositoryMining
from xml.etree import ElementTree
from tokenizer import TokeNizer
from get_entropy import *
from matplotlib import pyplot as plt

REPOSITORIES_FOLDER = "data/"


def get_projects(projects_list: list):
    projects_paths = list()

    for project in projects_list:
        print(f"Cloning repository {project}")

        path = REPOSITORIES_FOLDER + project.split("/")[-1]
        if not os.path.exists(path):
            git.Git(REPOSITORIES_FOLDER).clone(f"{project}")

        print("Checking if the project contains pom file ...")

        if not contains_pom_file(path):
            print(f"Removing {path}")
            rmtree(path)
        else:
            projects_paths.append(path)

    return projects_paths


def contains_pom_file(project_path: str):
    for root, folders, files in os.walk(project_path):
        if files.count('pom.xml'):
            return True
    return False


def extract_dependencies(source_code: str):
    dependencies = list()
    tree = ElementTree.fromstring(source_code)

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


def analyze_code(source_code: str):
    if source_code:
        source = source_code.split('\n')
        cleaned_lines = [line.strip() for line in source]
        source = '\n'.join(cleaned_lines)

        return extract_dependencies(source)


def checkout_previous_version(project_path, hash_number):
    os.system(f"cd {project_path} && git checkout {hash_number} > /dev/null 2>&1")


def restore_to_latest_commit(project_path):
    os.system(f"cd {project_path} && git checkout master > /dev/null 2>&1")


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


def make_train_files(project_path: str):
    print(f"Making train files ...")
    hash_list = list()
    k = 0

    for commit in RepositoryMining(project_path).traverse_commits():
        if k < 1:
            hash_list.append(commit.hash)
        else:
            checkout_previous_version(project_path, hash_list[k - 1])
            analyse_java_files(project_path, k - 1)
            hash_list.append(commit.hash)

        k += 1

    restore_to_latest_commit(project_path)


def make_test_files(project_path: str):
    print(f"Making test files ...")

    project_name = project_path.split("/")[1]
    k = 0
    for commit in RepositoryMining(project_path).traverse_commits():
        if k >= 1:
            checkout_previous_version(project_path, commit.hash)
            code_files = list()

            for modification in commit.modifications:
                if modification.filename.endswith(".java"):
                    token_procedure = TokeNizer("Java")
                    code = modification_to_str(modification.source_code)

                    tokens = ' '.join(token_procedure.get_pure_tokens(code))
                    code_files.append(tokens)

            with open(f"naturalness-data/java/new_data/{project_name}/fold{k - 1}.test", "w") as f:
                for file_code in code_files:
                    f.writelines(file_code + "\n")
        k += 1

    restore_to_latest_commit(project_path)


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


def number_commits(project_path: str):
    commits = 0
    for commit in RepositoryMining(project_path).traverse_commits():
        commits += 1
    return commits


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


def save_results_csv(project_path: str, unigram_values: list, bigram_values: list, trigram_values: list):
    print("Saving the results ...")
    RESULTS_FOLDER = "results/entropy/java/"

    data = {'unigram_values': unigram_values,
            'bigram_values': bigram_values,
            'trigram_values': trigram_values}

    dataframe = pd.DataFrame(data=data)

    dataframe.to_csv(RESULTS_FOLDER + f"{project_path}.csv")


def plot_results(results_path: str, project_name: str):
    dataframe = pd.read_csv(results_path)

    plt.plot(range(len(dataframe)), dataframe["unigram_values"], ".b-", label="Unigram Model")
    plt.plot(range(len(dataframe)), dataframe["bigram_values"], ".r-", label="Bigram Model")
    plt.plot(range(len(dataframe)), dataframe["trigram_values"], ".g-", label="Trigram Model")

    plt.xlabel("Commits")
    plt.ylabel("Entropy")

    plt.legend()
    plt.title(project_name)
    plt.show()


def plot_trigrams(results_path: str, project_name: str):
    dataframe = pd.read_csv(results_path)

    plt.plot(range(len(dataframe)), dataframe["trigram_values"], ".g-", label="Trigram Model")

    plt.xlabel("Commits")
    plt.ylabel("Entropy")

    plt.legend()
    plt.title(project_name)
    plt.show()


if __name__ == '__main__':
    projects = list()

    with open(f'{REPOSITORIES_FOLDER}/repositories.txt') as f:
        while True:
            line = f.readline()
            if not line:
                break
            else:
                line = line.strip()
                projects.append(line)

    projects_paths = get_projects(projects)

    with open(f'{REPOSITORIES_FOLDER}/remaining.txt', 'w') as f:
        for path in projects_paths:
            f.writelines(path + "\n")

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

        make_train_files(project_location)
        make_test_files(project_location)

        commits = number_commits(project_location) - 1
        unigram_list, bigram_list, trigram_list = get_entropy_commit(f"naturalness-data/java/new_data/{project_name}",
                                                                     reserved,
                                                                     commits)
        save_results_csv(project_name, unigram_list, bigram_list, trigram_list)

    # Plot all models
    for project_location in projects_poms:
        project_name = project_location.split("/")[1]
        plot_results(f"results/entropy/java/{project_name}.csv", project_name)

    # Plot only the trigram model
    for project_location in projects_poms:
        project_name = project_location.split("/")[1]
        plot_trigrams(f"results/entropy/java/{project_name}.csv", project_name)
