
import pickle


def load(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def save(scores, out_file):
    with open(out_file, 'wb') as f:
        pickle.dump(scores, f)


def num2str(n, K=4):
    n_str = str(n)
    for k in range(0, K - len(n_str)):
        n_str = '0' + n_str
    return n_str