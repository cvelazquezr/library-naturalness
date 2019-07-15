import os
import git
from shutil import rmtree
from pydriller import RepositoryMining
from xml.etree import ElementTree
from tokenizer import TokeNizer
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
    # Make train files
    code_files = list()

    for root, folders, files in os.walk(project_path):
        for file in files:
            if file.endswith('.java'):
                token_procedure = TokeNizer("Java")
                code = code_file_to_str(os.path.join(root, file))

                tokens = ' '.join(token_procedure.get_pure_tokens(code))
                code_files.append(tokens)

    with open(f"naturalness-data/java/new_data/fold{number}.train", "w") as f:
        for file_code in code_files:
            f.writelines(file_code + "\n")


def analyse_set_files(project_location: str, files: list, number):
    code_files = list()

    for file in files:
        if file.endswith('.java'):
            token_procedure = TokeNizer("Java")
            code = code_file_to_str(project_location + "/" + file)

            tokens = ' '.join(token_procedure.get_pure_tokens(code))
            code_files.append(tokens)

    with open(f"naturalness-data/java/new_data/fold{number}.test", "w") as f:
        for file_code in code_files:
            f.writelines(file_code + "\n")


def make_train_files(project_path: str):
    hash_list = list()
    k = 0

    for commit in RepositoryMining(project_path).traverse_commits():
        if k < 1:
            hash_list.append(commit.hash)
        else:
            print(f"Making train file {k - 1} ...")

            checkout_previous_version(project_path, hash_list[k - 1])
            analyse_java_files(project_path, k - 1)
            hash_list.append(commit.hash)

        k += 1

    restore_to_latest_commit(project_path)

        # for modification in commit.modifications:
        #     if modification.filename == 'pom.xml':
        #         dependencies = analyze_code(modification.source_code)
        #         print(dependencies, end=" ")
        # print()


def make_test_files(project_path: str):

    k = 0
    for commit in RepositoryMining(project_path).traverse_commits():
        print(f"Making test file {k} ...")
        checkout_previous_version(project_path, commit.hash)
        code_files = list()

        for modification in commit.modifications:
            if modification.filename.endswith(".java"):
                token_procedure = TokeNizer("Java")
                code = modification_to_str(modification.source_code)

                tokens = ' '.join(token_procedure.get_pure_tokens(code))
                code_files.append(tokens)

        with open(f"naturalness-data/java/new_data/test_tokens/{k}.java.tokens", "w") as f:
            for file_code in code_files:
                f.writelines(file_code + "\n")

        with open(f"naturalness-data/java/new_data/fold{k}.test", "w") as f:
            f.writelines(f"naturalness-data/java/new_data/test_tokens/{k}.java.tokens" + "\n")
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


if __name__ == '__main__':
    # projects = list()
    #
    # with open(f'{REPOSITORIES_FOLDER}/repositories.txt') as f:
    #     while True:
    #         line = f.readline()
    #         if not line:
    #             break
    #         else:
    #             line = line.strip()
    #             projects.append(line)
    #
    # projects_paths = get_projects(projects)
    #
    # with open(f'{REPOSITORIES_FOLDER}/remaining.txt', 'w') as f:
    #     for path in projects_paths:
    #         f.writelines(path + "\n")

    projects_poms = list()

    with open(f'{REPOSITORIES_FOLDER}/remaining.txt') as f:
        while True:
            line = f.readline()
            if not line:
                break
            else:
                line = line.strip()
                projects_poms.append(line)

    # make_train_files(projects_poms[0])
    make_test_files(projects_poms[0])
    # plt.show()
