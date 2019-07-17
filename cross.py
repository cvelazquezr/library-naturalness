import os
import sys


def cross(input_file_dir, fold_num, options, order, split_file=""):
    pipes = [os.pipe() for i in range(fold_num)]

    for i in range(fold_num):
        pid = os.fork()
        if pid == 0:
            os.close(pipes[i][0])

            train_file = '%s/fold%d.train' % (input_file_dir, i)
            test_file = '%s/fold%d.test' % (input_file_dir, i)
            scope_file = '%s/fold%d.scope' % (input_file_dir, i)
            if split_file == "":
                os.system(
                    './completion %s -NGRAM_FILE %s.%dgrams -NGRAM_ORDER %d -SCOPE_FILE %s -INPUT_FILE %s -OUTPUT_FILE %s.output | tee %s.log' % (
                        options, train_file, order, order, scope_file, test_file, test_file, test_file))
            else:
                os.system(
                    './completion %s -NGRAM_FILE %s.%dgrams -NGRAM_ORDER %d -SCOPE_FILE %s -INPUT_FILE %s -OUTPUT_FILE %s.output -SPLIT_FILE %s | tee %s.log' % (
                        options, train_file, order, order, scope_file, test_file, test_file, split_file, test_file))
            sys.exit()
        else:
            os.close(pipes[i][1])

    for p in pipes:
        os.wait()


if __name__ == '__main__':
    cross("naturalness-data/java/new_data/", 451,
          "-ENTROPY -BACKOFF -DEBUG -CACHE -CACHE_ORDER 3 -CACHE_DYNAMIC_LAMBDA -WINDOW_CACHE -WINDOW_SIZE 5000 -FILES",
          3)
