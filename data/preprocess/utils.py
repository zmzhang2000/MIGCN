from stanfordcorenlp import StanfordCoreNLP
import numpy as np
import pickle
import string


class WordEmbedding:
    """
    Example:
        word2vec = WordEmbedding()
        word2vec.get("hello")
    Return:
        300d numpy array
    """

    def __init__(self):
        self.word2vec = dict()
        for line in open("data/raw_data/glove.840B.300d.txt",'r'):
            line = line[:-1].split(" ")
            word = " ".join(line[:-300])
            vec = np.array(line[-300:],dtype=np.float)
            self.word2vec[word] = vec

    def get(self, word):
        return self.word2vec.get(word, np.random.uniform(-0.05, 0.05))


class SyntacticDependencyParser:
    """
    Example:
        parser = SyntacticDependencyParser()
        parser.parse("He chops the cucumber into pieces with the knife.")
    Return:
        tokens: list
        adj_mat: numpy array
    # https://github.com/Lynten/stanford-corenlp
    # https://universaldependencies.org/u/dep/index.html
    """

    def __init__(self):
        self.nlp = StanfordCoreNLP(r'data/raw_data/stanford-corenlp-full-2018-10-05')
        with open('relation', 'r') as f:
            self.relation = eval(f.read())

    def parse(self, sentence):
        sentence = sentence.lower()

        remove = str.maketrans('', '', string.punctuation)
        sentence = sentence.translate(remove)

        tokens = self.nlp.word_tokenize(sentence)

        dependency = self.nlp.dependency_parse(sentence)
        dependency = [d for d in dependency if d[0] in self.relation]

        adj_mat = np.eye(len(tokens))
        for d in dependency:
            w = self.relation[d[0]]
            adj_mat[d[1] - 1, d[2] - 1] = w

        return tokens, adj_mat
