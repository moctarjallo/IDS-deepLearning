
"""kddcup program

Usage:
    kddcup train FILE [--nrows=<int> --batch --batch_train --epochs --verbose] 
    kddcup -h | --help

Options:
    --nrows=<int>             number of rows to read from FILE [default: 10000]
    --batch=<None>             iteration size over data being read from FILE [default: None]
    --bach_train=<int>        training bacth [default: 128]
    --epochs=<int>            epochs [default: 5]
    --verbose=<int>           show the training [default: 1]
    -h --help                 Show this screen
"""

from docopt import docopt

from kddcup.main import train


def cli():
    args = docopt(__doc__)
    if args['train']:
        model = train(args['FILE'], int(args['--nrows']))
        print(model)


if __name__ == '__main__':
    cli()
