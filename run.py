from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("dir", help="Directory where text files can be found")
parser.add_argument("out", help="Output file to generate")
parser.add_argument("--window", type=int, help="Window", default=5)
parser.add_argument("--size", type=int, help="Size of the vector", default=100)

if __name__ == "__main__":
    import sys
    args = parser.parse_args(sys.argv[1:])

    from gensim.corpora.textcorpus import TextDirectoryCorpus
    from gensim_svd_wppmi import SVD_WPPMI

    corpus = TextDirectoryCorpus(args.dir)
    vecs = SVD_WPPMI(sentences=corpus, window=args.window, vector_size=args.size)
    vectors, cooc_matrix, ppmi = vecs.train(sentences=corpus)

    vecs.wv.save("vectors.keyed")
