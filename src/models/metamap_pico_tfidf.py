import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances


class MetaMap_PICO_TFIDF:
    def __init__(self, clinical_trial_pico_path=None, mapping_keys=None):

        assert isinstance(mapping_keys, list)
        assert len(mapping_keys) > 0
        assert isinstance(mapping_keys[0], list)
        self.mapping_keys = mapping_keys

        if clinical_trial_pico_path:
            if isinstance(clinical_trial_pico_path, str):
                self.clinical_trial_pico_path = clinical_trial_pico_path
                self.clinical_trial_pico = []
                with open(self.clinical_trial_pico_path) as fin:
                    for line in fin:
                        self.clinical_trial_pico.append(json.loads(line))
            elif isinstance(clinical_trial_pico_path, list):
                self.clinical_trial_pico_path = clinical_trial_pico_path
                self.clinical_trial_pico = []
                for p in clinical_trial_pico_path:
                    with open(p) as fin:
                        for line in fin:
                            self.clinical_trial_pico.append(json.loads(line))

            tmp_clinical_trial_pico = []

            for clinical_trial_info in self.clinical_trial_pico:
                cpico_info = []
                for sk, ck in self.mapping_keys:
                    tmp = []
                    for k in ck:
                        tmp.extend([kd.get('name', '') for kd in clinical_trial_info.get(k, [])])
                    cpico_info.append(tmp)
                tmp_clinical_trial_pico.append(cpico_info)

            self.clinical_trial_pico = tmp_clinical_trial_pico
            del tmp_clinical_trial_pico


        self.search_tfidf_vectorizer_dict = {
            'pico': TfidfVectorizer(tokenizer=lambda x: [t for t in x], use_idf=True, dtype=np.float32,
                                    lowercase=False),
        }

        self.corpus_dict = {'pico': [],
                            }
        for clinical_trial_info in self.clinical_trial_pico:
            self.corpus_dict['pico'].append(clinical_trial_info[0])

        self.search_embedding_dict = {
            'pico': self.search_tfidf_vectorizer_dict['pico'].fit_transform(self.corpus_dict['pico']),
        }

    def generate_tfidf_representation(self, document, search_tfidf_vectorizer):
        tf_transformer = TfidfVectorizer(tokenizer=lambda x: [t for t in x], use_idf=True,
                                         dtype=np.float32, lowercase=False)
        normalised_tf_vector = tf_transformer.fit_transform([document])
        # query sequence tf_feature->idf_indice
        idf_indices = [search_tfidf_vectorizer.vocabulary_[feature_name] for feature_name in
                       tf_transformer.get_feature_names()
                       if feature_name in search_tfidf_vectorizer.vocabulary_.keys()]
        # search sequence tfidf_feature -> tf_trans_indice
        tf_indices = [tf_transformer.vocabulary_[feature_name] for feature_name in
                      search_tfidf_vectorizer.get_feature_names()
                      if feature_name in tf_transformer.vocabulary_.keys()]
        if len(idf_indices) == 0:
            print('no idf indices')
            return None, None
        final_idf = search_tfidf_vectorizer.idf_[np.array(idf_indices)]
        final_tf = np.array(normalised_tf_vector.toarray()[0])[np.array(tf_indices)]
        document_tfidf = np.asmatrix(final_tf * final_idf)
        return document_tfidf, idf_indices

    def one_to_all_cosine_similarity_infer(self, doc):
        query_embedding_dict = {
            'pico': self.generate_tfidf_representation(doc['pico'], self.search_tfidf_vectorizer_dict['pico']),
        }
        if query_embedding_dict['pico'] is not None:
            cosine_similiarity_dict = {'pico': 1 - pairwise_distances(query_embedding_dict['pico'][0],
                                                                      self.search_embedding_dict['pico'][:,
                                                                      np.array(query_embedding_dict['pico'][1])],
                                                                      metric='cosine', n_jobs=-1).flatten(),
                                       }
            ans = {'pico': (cosine_similiarity_dict['pico'].tolist(), cosine_similiarity_dict['pico'].argsort()[::-1]),
                   }

        else:
            ans = {'pico': ([0. for _ in range(len(self.corpus_dict['pico']))],
                            np.array([0. for _ in range(len(self.corpus_dict['pico']))]).argsort()[::-1]),
                   }
        return ans
