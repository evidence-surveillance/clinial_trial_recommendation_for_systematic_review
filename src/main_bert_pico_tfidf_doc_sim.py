import csv
import numpy as np
from models import Bert_PICO_TFIDF
import argparse
import json
import os
import datetime


def main():
    parser = argparse.ArgumentParser(description='evaluate the tf-idf cosine document similarity method')
    parser.add_argument('--systematic_review_path', default='../data/systematic_reviews.csv',
                        help='the path to the systematic review file')
    parser.add_argument('--clinical_trial_path', default='../data/clinical_trials.csv',
                        help='the path to the clinical trials file')
    parser.add_argument('--systematic_review_pico_path', default='../data/systematic_reviews.json',
                        help='the path to the systematic review pico file')
    parser.add_argument('--clinical_trial_pico_path', default='../data/clinical_trials.json',
                        help='the path to the clinical trials pico file')
    parser.add_argument('--connections_path', default='../data/connections.csv',
                        help='the path to the connections file')
    parser.add_argument('--score_func', type=str,
                        help='the score calculation methods of a systematic review and a clinical trial in pico')
    parser.add_argument('--output_path', default='../exps/pico/', help='the path to the ouput folder.')
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if not os.path.exists(os.path.join(args.output_path, args.score_func)):
        os.makedirs(os.path.join(args.output_path, args.score_func))
    output_path = os.path.join(args.output_path, args.score_func)
    print('Starting...', datetime.datetime.now())
    with open(args.systematic_review_path, newline='') as systematic_review_csvfile, \
            open(args.clinical_trial_path, newline='') as clinical_trial_csvfile:
        systematic_review_reader = csv.DictReader(systematic_review_csvfile)
        clnical_trial_reader = csv.DictReader(clinical_trial_csvfile)

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

        with open(os.path.join(output_path, 'clinical_trial_nctid2id.json'), 'w') as fout:
            json.dump(clinical_trial_nctid2id, fout)
        with open(os.path.join(output_path, 'clinical_trial_pmid2id.json'), 'w') as fout:
            json.dump(clinical_trial_pmid2id, fout)
        with open(os.path.join(output_path, 'clinical_trial_id2nctid.json'), 'w') as fout:
            json.dump(clinical_trial_id2nctid, fout)
        with open(os.path.join(output_path, 'clinical_trial_id2pmid.json'), 'w') as fout:
            json.dump(clinical_trial_id2pmid, fout)

        systematic_review_corpus = []
        systematic_review_id2pmid = []
        for row in systematic_review_reader:
            systematic_review_id2pmid.append(row['pmid'])
            systematic_review_corpus.append('{} {}'.format(row['title'], row['abstract']))
        systematic_review_pmid2id = {pmid: i for i, pmid in enumerate(systematic_review_id2pmid)}

        print(len(systematic_review_corpus), len(systematic_review_id2pmid), len(systematic_review_pmid2id))

        with open(os.path.join(output_path, 'systematic_review_pmid2id.json'), 'w') as fout:
            json.dump(systematic_review_pmid2id, fout)
        with open(os.path.join(output_path, 'systematic_review_id2pmid.json'), 'w') as fout:
            json.dump(systematic_review_id2pmid, fout)
        print('Data Processing Done, Start building model...', datetime.datetime.now())

    model = Bert_PICO_TFIDF(clinical_trial_pico_path=args.clinical_trial_pico_path, score_func=args.score_func)
    print('Model build Done, Start evaluation...', datetime.datetime.now())

    if os.path.exists(os.path.join(output_path, 'evaluation.json')):
        with open(os.path.join(output_path, 'evaluation.json'), 'r') as f:
            line_count = sum(1 for _ in f)
    else:
        line_count = 0

    with open(os.path.join(output_path, 'evaluation.json'), 'a', buffering=1000) as fout, \
            open(args.connections_path, newline='') as connections_csvfile:
        connections_reader = csv.DictReader(connections_csvfile)

        systematic_review_trial_conenctions = []
        for i in range(len(systematic_review_corpus)):
            systematic_review_text = systematic_review_corpus[i]
            systematic_review_pmid = systematic_review_id2pmid[i]

            systematic_review_trial_conenctions.append({
                'pmid': systematic_review_pmid,
                'id': i,
                'connections': []
            })

        for row in connections_reader:
            pmid = row['pmid']  # -> systematic review
            nctid = row['nctid']  # -> clinical trial
            if nctid not in clinical_trial_nctid2id:
                nctid_ids = [-1]
            else:
                nctid_ids = clinical_trial_nctid2id[nctid]
            verified = row['verified']
            systematic_review_id = systematic_review_pmid2id.get(pmid, None)
            if systematic_review_id is None:
                continue
            systematic_review_trial_conenctions[systematic_review_id]['connections'].append({
                'nctid': nctid,
                'nctid_ids': nctid_ids,
                'verified': verified
            })

        systematic_review_pico_path = args.systematic_review_pico_path
        if systematic_review_pico_path:
            if isinstance(systematic_review_pico_path, str):
                systematic_review_pico = []
                with open(systematic_review_pico_path) as fin:
                    for line in fin:
                        systematic_review_pico.append(json.loads(line))
            elif isinstance(systematic_review_pico_path, list):
                systematic_review_pico = []
                for p in systematic_review_pico_path:
                    with open(p) as fin:
                        for line in fin:
                            systematic_review_pico.append(json.loads(line))
        score_func = args.score_func
        query_corpus_dict = {'pico': []}
        if score_func == 'pico':
            for systematic_review_info in systematic_review_pico:
                c_pico_i = [t[-1] for t in systematic_review_info['pico_i']]
                c_pico_p = [t[-1] for t in systematic_review_info['pico_p']]
                c_pico_o = [t[-1] for t in systematic_review_info['pico_o']]
                c_pico = c_pico_i + c_pico_p + c_pico_o
                query_corpus_dict['pico'].append(c_pico)

        if score_func == 'metamap_pico':
            for systematic_review_info in systematic_review_pico:
                c_pico_i = [t[0][0] for t in systematic_review_info['metamap_pico_i'] if t]
                c_pico_p = [t[0][0] for t in systematic_review_info['metamap_pico_p'] if t]
                c_pico_o = [t[0][0] for t in systematic_review_info['metamap_pico_o'] if t]
                c_pico = c_pico_i + c_pico_p + c_pico_o
                query_corpus_dict['pico'].append(c_pico)

        for scount, systematic_review_instance in enumerate(systematic_review_trial_conenctions):
            if scount < line_count: continue
            if len(systematic_review_instance['connections']):
                if len(query_corpus_dict['pico'][systematic_review_instance['id']]) == 0:
                    rank_results = [(-1, -1) for _ in systematic_review_instance['connections']]
                    systematic_review_instance['rank_results'] = {'pico': rank_results}
                    json.dump(systematic_review_instance, fout)
                    fout.write('\n')
                    continue
                cosine_similarities, rank_idx = model.one_to_all_cosine_similarity_infer(
                    {'pico': query_corpus_dict['pico'][systematic_review_instance['id']]})['pico']
                rank_idx = rank_idx.tolist()
                rank_results = []
                for connection in systematic_review_instance['connections']:
                    tmp_result = []
                    for nctid_id in connection['nctid_ids']:
                        if nctid_id != -1:
                            nctid_rank = rank_idx.index(nctid_id)
                            tmp_result.append((nctid_rank, cosine_similarities[nctid_id]))
                        else:
                            tmp_result.append((-1, -1))
                    rank_results.append(tmp_result)
                systematic_review_instance['rank_results'] = {'pico':rank_results}
            else:
                systematic_review_instance['rank_results'] = {'pico':[]}
            json.dump(systematic_review_instance, fout)
            fout.write('\n')
    print('Evaluation done...', datetime.datetime.now())


if __name__ == "__main__":
    main()
