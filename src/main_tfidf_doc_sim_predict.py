import csv
import numpy as np
from models import TFIDF_Doc_Sim
import argparse
import json
import os
import datetime


def main():
    parser = argparse.ArgumentParser(description='evaluate the tf-idf cosine document similarity method')
    parser.add_argument('--systematic_review_path', default='../data/systematic_reviews.csv', help='the path to the systematic review file')
    parser.add_argument('--clinical_trial_path', default='../data/clinical_trials.csv', help='the path to the clinical trials file')
    parser.add_argument('--output_path', default='../exps/tfidf/', help='the path to the ouput folder.')
    parser.add_argument('--over_ride', action='store_true', help='over write the output folder')
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    print('Starting...', datetime.datetime.now())
    with open(args.systematic_review_path, newline='') as systematic_review_csvfile, \
            open(args.clinical_trial_path, newline='') as clinical_trial_csvfile:
        # pmid,title,abstract
        systematic_review_reader = csv.DictReader(systematic_review_csvfile)#, fieldnames=['pmid', 'title', 'abstract'])
        clnical_trial_reader = csv.DictReader(clinical_trial_csvfile)#, fieldnames=['nctid', 'title', 'pub_pmid',
                                                                      #            'pub_title', 'pub_abstract'])

        clinical_trial_corpus = []
        clinical_trial_id2nctid = []
        clinical_trial_id2pmid = []
        for row in clnical_trial_reader:
            clinical_trial_id2nctid.append(row['nctid'])
            clinical_trial_id2pmid.append(row['pub_pmid'])
            clinical_trial_corpus.append('{} {}'.format(row['pub_title'], row['pub_abstract']))

        clinical_trial_nctid2id = {}
        for i, nctid in enumerate(clinical_trial_id2nctid):
            if nctid in clinical_trial_nctid2id:
                clinical_trial_nctid2id[nctid].append(i)
            else:
                clinical_trial_nctid2id[nctid] = [i]
        clinical_trial_pmid2id = {}
        for i, pmid in enumerate(clinical_trial_id2pmid):
            if pmid in clinical_trial_pmid2id:
                clinical_trial_pmid2id[pmid].append(i)
            else:
                clinical_trial_pmid2id[pmid] = [i]

        print(len(clinical_trial_corpus), len(clinical_trial_id2nctid), len(clinical_trial_id2pmid),
              len(clinical_trial_nctid2id), len(clinical_trial_pmid2id))

        with open(os.path.join(args.output_path, 'clinical_trial_nctid2id.json'), 'w') as fout:
            json.dump(clinical_trial_nctid2id, fout)
        with open(os.path.join(args.output_path, 'clinical_trial_pmid2id.json'), 'w') as fout:
            json.dump(clinical_trial_pmid2id, fout)
        with open(os.path.join(args.output_path, 'clinical_trial_id2nctid.json'), 'w') as fout:
            json.dump(clinical_trial_id2nctid, fout)
        with open(os.path.join(args.output_path, 'clinical_trial_id2pmid.json'), 'w') as fout:
            json.dump(clinical_trial_id2pmid, fout)

        systematic_review_corpus = []
        systematic_review_id2pmid = []
        for row in systematic_review_reader:
            systematic_review_id2pmid.append(row['pmid'])
            systematic_review_corpus.append('{} {}'.format(row['title'], row['abstract']))
        systematic_review_pmid2id = {pmid: i for i, pmid in enumerate(systematic_review_id2pmid)}

        print(len(systematic_review_corpus), len(systematic_review_id2pmid), len(systematic_review_pmid2id))

        with open(os.path.join(args.output_path, 'systematic_review_pmid2id.json'), 'w') as fout:
            json.dump(systematic_review_pmid2id, fout)
        with open(os.path.join(args.output_path, 'systematic_review_id2pmid.json'), 'w') as fout:
            json.dump(systematic_review_id2pmid, fout)
        print('Data Processing Done, Start building model...', datetime.datetime.now())
        model = TFIDF_Doc_Sim(corpus=clinical_trial_corpus)
        print('Model build Done, Start evaluation...', datetime.datetime.now())

    with open(os.path.join(args.output_path, 'evaluation.json'), 'w') as fout:
        systematic_review_trial_conenctions = []
        for i in range(len(systematic_review_corpus)):
            systematic_review_text = systematic_review_corpus[i]
            systematic_review_pmid = systematic_review_id2pmid[i]

            systematic_review_trial_conenctions.append({
                'text': systematic_review_text,
                'pmid': systematic_review_pmid,
                'id': i,
                'connections': []
            })

        for systematic_review_instance in systematic_review_trial_conenctions:
            cosine_similarities, rank_idx = model.one_to_all_cosine_similarity_infer(
                systematic_review_instance['text'])
            rank_idx = rank_idx.tolist()
            systematic_review_instance['rank_idx'] = rank_idx
            systematic_review_instance['cosine_similarities'] = cosine_similarities
            systematic_review_instance.pop('text')
            json.dump(systematic_review_instance, fout)
            fout.write('\n')

    print('Evaluation done...', datetime.datetime.now())
if __name__ == "__main__":
    main()
