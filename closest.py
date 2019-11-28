from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("vectors", help="Vectors generated")
parser.add_argument("lookup", help="Word you want to look up for")

if __name__ == "__main__":
    import sys
    args = parser.parse_args(sys.argv[1:])
    vecs: WordEmbeddingsKeyedVectors = WordEmbeddingsKeyedVectors.load(args.vectors)
    print(vecs.most_similar(positive=(args.lookup, )))
