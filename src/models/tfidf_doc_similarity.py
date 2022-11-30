import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity


def get_stopwords():
    pubmed_stopwords = 'a, about, again, all, almost, also, although, always, among, an, and, another, any, are, as, ' \
                       'at, be, because, been, before, being, between, both, but, by,can, could,did, do, does, done, ' \
                       'due, during,each, either, enough, especially, etc, for, found, from, further, had, has, ' \
                       'have, having, here, how, however, i, if, in, into, is, it, its, itself, just, kg, km, made, ' \
                       'mainly, make, may, mg, might, ml, mm, most, mostly, must, nearly, neither, no, nor, ' \
                       'obtained, of, often, on, our, overall, perhaps, pmid, quite, rather, really, regarding, ' \
                       'seem, seen, several, should, show, showed, shown, shows, significantly, since, so, some, ' \
                       'such, than, that, the, their, theirs, them, then, there, therefore, these, they, this, ' \
                       'those, through, thus, to, upon, various, very, was, we, were, what, when, which, while, ' \
                       'with, within, without, would'.replace(' ', '').split(',')
    # nlp = spacy.load("en_core_web_sm")
    stopwords = ['nothing', 'between', 'how', 'top', 'due', 'whereafter', 'do', 'he', 'at', 'while', 'anyhow', 'besides', 'only', 'ours', 'may', 'third', 'much', 'several', 'anyway', 'together', 'also', 'fifteen', 'any', 'even', 'becomes', 'noone', 'their', 'her', 'it', 'whenever', 'whereupon', 'see', 'though', 'please', 'used', 'what', 'an', 'that', 'doing', 'towards', 'per', 'never', 'else', 'moreover', 'did', 'we', 'am', 'a', 'show', "'ll", 'these', 'for', 'hers', 'us', 'would', 'elsewhere', 'wherein', 'each', 'beyond', 'therein', 'which', 'whatever', 'within', 'yours', 'himself', 'thereafter', 'n’t', 'up', 'sixty', 'they', 'latterly', 'toward', 'among', 'seem', 'until', 'more', 'during', 'is', 'either', 'being', 'something', 'themselves', 'before', 'neither', 'herein', 'made', 'many', 'go', 'take', 'indeed', 'whether', 'ourselves', 'move', 'nine', '‘ll', 'your', 'serious', 'quite', 'name', 'whence', 'other', 'rather', 'sometimes', 'seemed', 'others', 'down', '‘re', 'enough', 'somehow', '’ll', 'various', 'his', 'former', 'after', 'over', 'none', 'now', 'mostly', 'really', 'might', 'own', 'all', 'herself', 'onto', 'was', 'had', 'amount', 'through', 'nevertheless', 'by', 'beside', 'around', 'but', '‘s', 'does', 'this', 'them', 'against', 'six', 'on', 'into', 'last', 'to', 'most', 'well', 'often', 'ten', 'who', 'yourselves', 'every', 'everywhere', 'although', 'anyone', 'became', 'always', 'without', 'just', 'except', 'with', 'nowhere', 'those', 'wherever', 'n‘t', 'full', 'hereby', "'ve", 'sometime', 'done', 'someone', 'using', 'than', 'off', 'then', 'forty', '’m', 'eleven', 'have', 'some', '’s', 'in', 'must', 'should', 'empty', 'will', 'otherwise', 'give', 'alone', 'or', 'latter', 'as', 'and', 'no', 'out', 'hence', 'i', 'five', 'hundred', 'when', 'say', 'everything', 'could', 'front', 'again', 'meanwhile', 'unless', 'yourself', 'its', '’re', 'further', 'already', '’d', 'along', 'hereupon', 'under', 'here', "'re", 'same', 'get', 'whose', 'whereby', 'seems', 'has', 'once', 'formerly', 'my', 'anywhere', 'regarding', 'nor', '‘m', 'still', 'namely', 'put', 'cannot', 'our', 'nobody', 'however', 'both', 'been', 'itself', 'thereupon', 'can', 'you', 'such', 'why', 'yet', 'thereby', 'thus', 'another', 'thence', 'four', "'d", 'one', 'twelve', 'whole', 'afterwards', 'least', 'less', 'from', 'becoming', 'few', "'m", 'she', 'be', 'beforehand', '‘ve', 'seeming', 'part', 'are', 'make', 'next', 'mine', 'first', 'where', 'since', 'thru', 'about', 'almost', 'below', 'across', 'become', 'there', 'him', 'whom', 'behind', 'three', 'ever', 'so', 'myself', 'via', 'me', '’ve', 'amongst', 'not', 'hereafter', 'above', 'whoever', 'were', 'of', 'throughout', 'call', 'whither', 'eight', 'twenty', 'bottom', 'the', 'because', '‘d', 're', 'very', 'therefore', 'keep', 'somewhere', 'side', 'everyone', 'back', 'if', 'upon', 'perhaps', 'whereas', 'too', "n't", 'two', 'fifty', 'ca', 'anything', "'s"]
    #
    # list(nlp.Defaults.stop_words)
    punctuations = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
    # list(string.punctuation)
    single_chars = ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
                    'w', 'x', 'y', '‘', '’']
    allstopwords = stopwords + punctuations + single_chars + pubmed_stopwords
    allstopwords = list(set(allstopwords))
    return allstopwords


class TFIDF_Doc_Sim:
    def __init__(self, corpus):
        self.corpus = corpus

        self.search_tfidf_vectorizer = TfidfVectorizer(stop_words=get_stopwords(),
                                                       use_idf=True, dtype=np.float32)
        self.search_embeddings = self.search_tfidf_vectorizer.fit_transform(corpus)

    def build_query_tfidf_vectorizer(self, corpus):
        self.query_tfidf_vectorizer = TfidfVectorizer(stop_words=get_stopwords(),
                                                      use_idf=True, dtype=np.float32)
        self.query_embeddings = self.query_tfidf_vectorizer.fit_transform(corpus)

    def generate_tfidf_representation(self, document):
        tf_transformer = TfidfVectorizer(use_idf=False)
        normalised_tf_vector = tf_transformer.fit_transform([document])
        # query sequence tf_feature->idf_indice
        idf_indices = [self.search_tfidf_vectorizer.vocabulary_[feature_name] for feature_name in tf_transformer.get_feature_names()
                       if feature_name in self.search_tfidf_vectorizer.vocabulary_.keys()]
        # search sequence tfidf_feature -> tf_trans_indice
        tf_indices = [tf_transformer.vocabulary_[feature_name] for feature_name in self.search_tfidf_vectorizer.get_feature_names()
                      if feature_name in tf_transformer.vocabulary_.keys()]
        final_idf = self.search_tfidf_vectorizer.idf_[np.array(idf_indices)]
        final_tf = np.array(normalised_tf_vector.toarray()[0])[np.array(tf_indices)]
        document_tfidf = np.asmatrix(final_tf * final_idf)
        return document_tfidf, idf_indices

    def one_to_all_cosine_similarity(self, query_embedding, search_embeddings=None):
        if search_embeddings is None:
            search_embeddings = self.search_embeddings
        cosine_similarities = 1 - pairwise_distances(query_embedding, search_embeddings,
                                                     metric='cosine', n_jobs=-1).flatten()
        ranked_index = cosine_similarities.argsort()[::-1]
        return cosine_similarities.tolist(), ranked_index

    def one_to_all_cosine_similarity_infer(self, doc, search_embeddings=None):
        if search_embeddings is None:
            search_embeddings = self.search_embeddings
        query_embedding, idf_indices = self.generate_tfidf_representation(doc)
        search_embeddings = search_embeddings[:, np.array(idf_indices)]
        cosine_similarities = 1 - pairwise_distances(query_embedding, search_embeddings,
                                                     metric='cosine', n_jobs=-1).flatten()
        related_docs_rank_indices = cosine_similarities.argsort()[::-1]
        return cosine_similarities.tolist(), related_docs_rank_indices
