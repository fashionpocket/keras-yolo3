import argparse
import os
import numpy as np

def make_toydata(filepath, num_data, outpath):
    with open(filepath) as f:
        big_lines = f.readlines()
    with open(outpath, "w") as f_new:
        sampled = np.random.choice(len(big_lines), num_data, replace=False)

        for i in sampled:
            f_new.write(big_lines[i])

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='make_toydata.py',
        description='Generate toi data from ssss_train.txt (annotation txt file)',
        add_help=True,
    )
    parser.add_argument('-p', '--file_path', type=str, required=True, help="path to ssss_train.txt")
    parser.add_argument('-n', '--num_data', type=int, required=True, help="path to ssss_train.txt")
    parser.add_argument('-o', '--out_path', type=str, required=True)

    args = parser.parse_args()
    file_path = args.file_path
    num_data = args.num_data
    out_path = args.out_path

    if not os.path.exists(file_path):
        raise Exception(file_path + ' does not exist.')

    print('- ANNOTATION FILE PATH : ' + file_path)
    print('- OUTPUT FILE PATH     : ' + out_path)
    print('- NUMBER OF DATA       : ' + str(num_data))
    make_toydata(file_path, num_data, out_path)
    print("Making toydata file correctly")