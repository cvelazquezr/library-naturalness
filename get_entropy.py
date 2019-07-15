import os
import sys
import time
import random
from random import shuffle
from matplotlib import pyplot as plt


def random_split(input_file_dir, fold_num, suffix):
    files_list = []
    file_dir = os.path.join(input_file_dir, 'files')
    for file in os.listdir(file_dir):
        if file.endswith(suffix):
            files_list.append((int(file[:file.find('.')]), os.path.join(file_dir, file)))

    files_list.sort()
    files_list = [file for id, file in files_list]
    shuffle(files_list)

    files_number = len(files_list)

    part_files_number = files_number / fold_num
    left_files_number = files_number % fold_num

    part_files_list = []

    # multi-process
    last_end = 0
    for i in range(fold_num):
        start = last_end
        last_end = start + part_files_number
        if i < left_files_number:
            last_end += 1

        start = int(start)
        last_end = int(last_end)

        part_files_list.append(files_list[start:last_end])

    return part_files_list


def write_train_file(input_files, output_file):
    # print("Writing to " + output_file)
    with open(output_file, "w") as o:
        for t in input_files:
            with open(t, "r") as i:
                temp = i.read()
                o.write(temp)


def select_random_subsample_int(baseList, sampleSize):
    # print("Sample Size:" + str(sampleSize))
    # print("To Sample Size: " + str(len(baseList)))
    #Get random sample of a list for python: credit to answer on this question:
    #http://stackoverflow.com/questions/6482889/get-random-sample-from-list-while-maintaining-ordering-of-items
    subSample = [baseList[i] for i in sorted(random.sample(range(len(baseList)), sampleSize)) ]
    return subSample


def create_fold(input_file_dir, fold_num, downSample, splitSelection):
    part_files_list = []
    #     if(splitSelection == 0):
    #         part_files_list = balancedProjectSplit(input_file_dir, fold_num, '.tokens') #Make test sets a balanced set of projects.
    #     elif(splitSelection == 1):
    #         part_files_list = randomProjectSplit(input_file_dir, fold_num, '.tokens') #Make test sets a random set of projects.
    if (splitSelection == 2):
        part_files_list = random_split(input_file_dir, fold_num, '.tokens')  # Use for english corpus
    #     elif(splitSelection == 3):
    #         part_files_list =  part_files_list = balancedProjectSplit(input_file_dir, fold_num, '.tokens', False) #Make test sets a balanced set of projects, determined from a non-pickle style format.

    # Not used in entropy calculations
    method_files_list = random_split(input_file_dir, fold_num, '.scope')

    # Create a version that aims to create a balanced split of files (i.e. it downsamples more aggressively
    # in the big projects)
    if downSample != 1.0:

        # 0.Get the total number of files we have.
        # 1.Get the number of files we need to reduce to.
        # 2.Sort by size of fold from smallest to biggest, get the smallest.
        # 3.If 10*This size is bigger than raw reductions across the board, then first reduce all folds to
        # this size.  Then downsample the rest evenly...
        # If it is smaller, check next biggest and do #2 if bigger, repeat until we find one that satisfies
        # Then, we only want to sample down on the bigger ones if 10* the smallest is too small

        totalFiles = sum(len(item) for item in part_files_list)
        targetSize = int(totalFiles * downSample)
        # print("Total Files: " + str(totalFiles))
        # print("Target Size: " + str(targetSize))
        part_files_list = sorted(part_files_list, key=lambda x: len(x))
        # print("Start sizes:")
        # for p in part_files_list:
        #     print(str(len(p)))

        part_files_sampled = []
        too_Small_Count = 0
        reducedSize = 0
        for i in range(0, len(part_files_list)):
            smallestSplit = len(part_files_list[i]) * len(part_files_list)  # Add modifier for lost space before...

            # print("Smallest Split:" + str(smallestSplit))

            if (smallestSplit > targetSize):
                part_files_sampled += part_files_list[
                                      :i + 1]  # This seems to be needed in some cases and not in others???

                # This can can remove too much?  (It's okay as long as its in the same general size.)
                part_files_sampled += [select_random_subsample_int(nextFold, len(part_files_list[i])) for nextFold in
                                       part_files_list[i + 1:]]
                reducedSize = len(part_files_list[i])
                break
            else:
                too_Small_Count += 1

        # print("Mid sizes:")
        # for p in part_files_sampled:
        #     print(str(len(p)))

        newSize = sum(len(item) for item in part_files_sampled)
        leftToRemove = sum(len(item) for item in part_files_sampled) - targetSize
        # print("New Total Size:" + str(newSize))
        # print("Left to remove:" + str(leftToRemove))
        if (leftToRemove > 0):
            # Now, we remove an equal amount of files from each sample that is too big until we reach the % target.
            foldReduce = (leftToRemove) / (len(part_files_list) - too_Small_Count)

            # Do we need to add a check to make sure we don't reduce the big ones below our smallest ones? (Only small problem in Ruby....)
            part_files_list = part_files_sampled[:too_Small_Count] + [
                select_random_subsample_int(nextFold, len(nextFold) - foldReduce) for nextFold in
                part_files_sampled[too_Small_Count:]]
        else:
            part_files_list = part_files_sampled

        # print("End sizes:")
        # for p in part_files_list:
        #     print(str(len(p)))

    for i in range(fold_num):
        train_files = []
        for j in range(fold_num):
            if j != i:
                train_files.extend(part_files_list[j])

        # Writing Test Files
        with open(f"{input_file_dir}/fold{i}.test", "w") as f:
            f.writelines('\n'.join(part_files_list[i]))

        write_train_file(train_files, input_file_dir + "/fold" + str(i) + ".train")


def train(input_file_dir, fold_num, order, downSample, splitSelection, lm_conf):
    # create_fold(input_file_dir, fold_num, downSample, splitSelection)

    pipes = [os.pipe() for i in range(fold_num)]

    for i in range(fold_num):
        pid = os.fork()
        if pid == 0:
            os.close(pipes[i][0])

            train_file = '%s/fold%d.train' % (input_file_dir, i)

            if lm_conf[0] == "MITLM":
                # Get vocab...
                print("%s -t %s -write-vocab %s.vocab" % (lm_conf[1], train_file, train_file))
                os.system("%s -t %s -write-vocab %s.vocab" % (lm_conf[1], train_file, train_file))
                # Alternative uses mitlm instead...
                print('%s -order %d -v %s.vocab -unk -smoothing ModKN -t %s -write-lm %s.%dgrams' % (
                lm_conf[1], order, train_file, train_file, train_file, order))
                os.system('%s -order %d -v %s.vocab -unk -smoothing ModKN -t %s -write-lm %s.%dgrams' % (
                lm_conf[1], order, train_file, train_file, train_file, order))
            elif lm_conf[0] == "SRILM":
                # Original Srilm
                print('%s -text %s -lm %s.kn.lm.gz -order %d -unk -kndiscount -interpolate' % (
                lm_conf[1], train_file, train_file, order))
                os.system('%s -text %s -lm %s.kn.lm.gz -order %d -unk -kndiscount -interpolate' % (
                lm_conf[1], train_file, train_file, order))
                print('%s -lm %s.kn.lm.gz -unk -order %d -write-lm %s.%dgrams' % (
                lm_conf[1], train_file, order, train_file, order))
                os.system('%s -lm %s.kn.lm.gz -unk -order %d -write-lm %s.%dgrams' % (
                lm_conf[1], train_file, order, train_file, order))
                os.system('rm %s.kn.lm.gz' % train_file)
            elif lm_conf[0] == "KENLM":
                # Kenlm
                # print('%s -o %d -S %s --interpolate_unigrams 0 <%s >%s.%dgrams' % (
                # lm_conf[1], order, "5%", train_file, train_file, order))
                os.system('%s -o %d -S %s --interpolate_unigrams 0 <%s >%s.%dgrams' % (
                lm_conf[1], order, "5%", train_file, train_file, order))
            else:
                print("This is not a recognized language model.")

            sys.exit()
        else:
            os.close(pipes[i][1])

    for p in pipes:
        os.wait()


def entropy_values(path_file: str):
    entropies = list()

    with open(path_file) as f:
        while True:
            line = f.readline()
            if not line: break
            else:
                if line.startswith('Entropy: '):
                    line = line.strip()
                    value = float(line.split()[1])
                    entropies.append(value)

    return entropies


if __name__ == '__main__':
    # start = time.time()

    # Train process
    # lm_conf = ("KENLM", "/Users/kmilo/Dev/extra/kenlm/build/bin/lmplz")
    # train("naturalness-data/java/new_data", 452, 3, 1.0, 2, lm_conf)

    # Write the entropy raw output to file
    # os.system('python3 cross.py > results/entropy/java/3gramsCache.txt')

    # Getting entropies values
    entropy_list = entropy_values("results/entropy/java/3gramsCache.txt")
    plt.bar(range(len(entropy_list)), entropy_list)
    plt.show()

    # TODO: Replace Nan values for 0
    # TODO: Make visualization tool to check commits contents vs entropy values
    # TODO: Check the entropy values to another project
    # TODO: Calculate the cosine similarity for the projects

    # end = time.time()
    #
    # print(f"Time consumed: {end - start}")
