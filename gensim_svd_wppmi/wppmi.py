from gensim.models.base_any2vec import BaseWordEmbeddingsModel
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors, Vocab
from gensim.corpora.textcorpus import TextCorpus
from gensim.utils import SaveLoad
from logging import getLogger
from collections import Counter
from tqdm import tqdm
from typing import Tuple
from types import GeneratorType
import scipy.sparse
import numpy as np

logger = getLogger(__name__)


class SvdWppmiVocabulary(SaveLoad):
    def __init__(self, min_count: int = 0):
        self.min_count = min_count
        self.raw_vocab = Counter()

    def _scan_vocab(self, sentences: TextCorpus) -> Tuple[int, int]:
        total_words, corpus_count = 0, 0
        for sentence in tqdm(sentences.get_texts()):
            self.raw_vocab.update(Counter(sentence))
            corpus_count += 1
            total_words += len(sentence)
        return total_words, corpus_count

    def scan_vocab(self, sentences=None, *args, **kwargs):
        logger.info("collecting all words and their counts")

        total_words, corpus_count = self._scan_vocab(sentences)

        logger.info(
            "collected %i word types from a corpus of %i raw words and %i sentences",
            len(self.raw_vocab), total_words, corpus_count
        )

        return total_words, corpus_count

    def prepare_vocab(self, wv: WordEmbeddingsKeyedVectors, dry_run: bool = False):
        retain_words, retain_total = [], 0
        drop_unique, drop_total = 0, 0
        for word, freq in self.raw_vocab.items():
            if freq > self.min_count:
                retain_words.append(word)
                retain_total += freq
                if not dry_run:
                    wv.vocab[word] = Vocab(count=freq, index=len(wv.index2word))
                    wv.index2word.append(word)
            else:
                drop_unique += 1
                drop_total += freq

        original_unique_total = len(retain_words) + drop_unique
        retain_unique_pct = len(retain_words) * 100 / max(original_unique_total, 1)
        logger.info(
            "min_count=%d retains %i unique words (%i%% of original %i, drops %i)",
            self.min_count, len(retain_words), retain_unique_pct, original_unique_total, drop_unique
        )
        original_total = retain_total + drop_total
        retain_pct = retain_total * 100 / max(original_total, 1)
        logger.info(
            "min_count=%d leaves %i word corpus (%i%% of original %i, drops %i)",
            self.min_count, retain_total, retain_pct, original_total, drop_total
        )

        return wv


class SVD_WPPMI(BaseWordEmbeddingsModel):
    def __init__(
            self, sentences=None, workers=3, vector_size=100, window=5):

        self.vocabulary = SvdWppmiVocabulary()
        self.window = window
        self.vector_size = vector_size
        self.wv = WordEmbeddingsKeyedVectors(vector_size=self.vector_size)

        if sentences is not None:
            self._check_input_data_sanity(data_iterable=sentences)
            if isinstance(sentences, GeneratorType):
                raise TypeError("You can't pass a generator as the sentences argument. Try a sequence.")

            self.build_vocab(sentences=sentences)
            # self.train(sentences=sentences)
            #    sentences=sentences, corpus_file=corpus_file, total_examples=self.corpus_count,
            #    total_words=self.corpus_total_words, epochs=self.epochs, start_alpha=self.alpha,
            #    end_alpha=self.min_alpha, compute_loss=compute_loss)

    def tokens_2_index(self, tokens):
        return tuple(map(self.wv.index2word.index, tokens))

    def build_vocab(self, sentences: TextCorpus, dry_run: bool = False):
        total_words, corpus_count = self.vocabulary.scan_vocab(sentences)
        self.corpus_count = corpus_count
        self.corpus_total_words = total_words
        print(self.vocabulary.prepare_vocab(self.wv, dry_run=dry_run))

    def _coocurence_matrix(self, sentences: TextCorpus, threshold=0.7):
        # https://github.com/johndpope/summerschool/blob/master/assignment/classifier/3-Embeddings_SVD_Viz.ipynb
        matrix_size = len(self.wv.index2word)
        C = scipy.sparse.csc_matrix((matrix_size, matrix_size), dtype=np.float32)

        print("Scanning for pairs")
        for sentence in tqdm(sentences.get_texts(), total=self.corpus_count):
            tokens = self.tokens_2_index(sentence)
            for k in range(1, self.window + 1):
                logger.info(u"Counting pairs (i, i \u00B1 {:d}) ...".format(k))
                i = tokens[:-k]  # current word
                j = tokens[k:]  # k words ahead
                data = (np.ones_like(i), (i, j))  # values, indices

                Ck_plus = scipy.sparse.csc_matrix(data, shape=C.shape, dtype=np.float32)
                Ck_minus = Ck_plus.T  # Consider k words behind
                C += Ck_plus + Ck_minus

        print()
        print("Co-occurrence matrix: {:,} words x {:,} words".format(*C.shape))
        print("  {:.02g} nonzero elements".format(C.nnz))
        return C

    def _ppmi(self, cooc_matrix: scipy.sparse.csc_matrix):
        """Tranform a counts matrix to PPMI.

        Args:
          cooc_matrix: scipy.sparse.csc_matrix of counts C_ij

        Returns:
          (scipy.sparse.csc_matrix) PPMI(C) as defined above
        """
        total = float(cooc_matrix.sum())  # total counts
        # sum each column (along rows)
        total_column = np.array(cooc_matrix.sum(axis=0), dtype=np.float64).flatten()
        # sum each row (along columns)
        total_row = np.array(cooc_matrix.sum(axis=1), dtype=np.float64).flatten()

        # Get indices of relevant elements
        ii, jj = cooc_matrix.nonzero()  # row, column indices
        print(ii)
        Cij = np.array(cooc_matrix[ii, jj], dtype=np.float64).flatten()
        print(Cij)
        ##
        # PMI equation
        pmi = np.log(Cij * total / (total_row[ii] * total_column[jj]))
        ##
        # Truncate to positive only
        ppmi = np.maximum(0, pmi)  # take positive only

        # Re-format as sparse matrix
        ret = scipy.sparse.csc_matrix((ppmi, (ii, jj)), shape=cooc_matrix.shape,
                                      dtype=np.float64)
        ret.eliminate_zeros()  # remove zeros
        return ret

    def train(self, sentences: TextCorpus) -> Tuple[scipy.sparse.csc_matrix]:
        # Build co-occurence matrix
        cooc_matrix = self._coocurence_matrix(sentences)
        ppmi = self._ppmi(cooc_matrix)
        return cooc_matrix,


if __name__ == "__main__":
    from gensim.corpora.textcorpus import TextDirectoryCorpus
    logger.setLevel("INFO")

    corpus = TextDirectoryCorpus("../data")
    vecs = SVD_WPPMI(sentences=corpus)
    cooc_matrix, = vecs.train(sentences=corpus)

    lasciv, = vecs.tokens_2_index(["lascivvs"])
    assert cooc_matrix.getrow(lasciv).todense().tolist() == \
           cooc_matrix.getcol(lasciv).transpose().todense().tolist()
