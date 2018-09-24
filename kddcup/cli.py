

"""kddcup program

Usage:
    kddcup train FILE [--nrows --batch]
    kddcup test 
    kddcup <operation> [--file] <filename>
    kddcup -h | --help

Options:
    -f --file           data to train the model on
    -h --help           Show this screen
"""

from docopt import docopt


def cli():
    args = docopt(__doc__)
    print(args)


if __name__ == '__main__':
    cli()
