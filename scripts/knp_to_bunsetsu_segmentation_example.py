import sys
import re


def main(knp_dir, filelist_file):
    files = read(filelist_file)
    for file in files:
        write_example(knp_dir + '/' + file)

def read(filelist_file):
    files = []
    with open(filelist_file) as f:
        for line in f:
            files.append(line.rstrip())
    return files

def write_example(knp_file):
    with open(knp_file) as f:
        results = []
        for line in f:
            if re.match(r'[#+]', line):
                pass
            elif re.match(r'\*', line):
                if len(results) == 0:
                    continue
                results[-1] = re.sub(r'/N$', r'/Y', results[-1])
            elif line == 'EOS\n':
                if len(results) == 0:
                    continue
                results[-1] = re.sub(r'/N$', r'/Y', results[-1])
                print(' '.join(results))
                results = []
            else:
                array = line.rstrip().split()
                results.append(array[0] + '/' + array[3] + '/N')


if __name__ == '__main__':
    args = sys.argv
    main(args[1], args[2])
