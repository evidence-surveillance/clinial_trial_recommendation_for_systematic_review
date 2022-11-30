import csv
import numpy as np
from models import MetaMap_PICO_TFIDF
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
    parser.add_argument('--mapping_keys', default='../data/clinical_trials.json',
                        help='the path of the keys of matching between systematic review and clinical trial')
    parser.add_argument('--output_path', default='../exps/pico/', help='the path to the ouput folder.')

    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    output_path = os.path.join(args.output_path, os.path.splitext(os.path.basename(args.mapping_keys))[0])
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print('Starting...', datetime.datetime.now())
    with open(args.systematic_review_path, newline='') as systematic_review_csvfile, \
            open(args.clinical_trial_path, newline='') as clinical_trial_csvfile:
        systematic_review_reader = csv.DictReader(systematic_review_csvfile)
        clnical_trial_reader = csv.DictReader(clinical_trial_csvfile)

        clinical_trial_id2nctid = []
        clinical_trial_id2pmid = []
        for row in clnical_trial_reader:
            clinical_trial_id2nctid.append(row['nctid'])
            if 'pub_pmid' in row:
                clinical_trial_id2pmid.append(row['pub_pmid'])
            elif 'pmid' in row:
                clinical_trial_id2pmid.append(row['pmid'])

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

        print(len(clinical_trial_id2nctid), len(clinical_trial_id2pmid),
              len(clinical_trial_nctid2id), len(clinical_trial_pmid2id))

        with open(os.path.join(output_path, 'clinical_trial_nctid2id.json'), 'w') as fout:
            json.dump(clinical_trial_nctid2id, fout)
        with open(os.path.join(output_path, 'clinical_trial_pmid2id.json'), 'w') as fout:
            json.dump(clinical_trial_pmid2id, fout)
        with open(os.path.join(output_path, 'clinical_trial_id2nctid.json'), 'w') as fout:
            json.dump(clinical_trial_id2nctid, fout)
        with open(os.path.join(output_path, 'clinical_trial_id2pmid.json'), 'w') as fout:
            json.dump(clinical_trial_id2pmid, fout)

        systematic_review_id2pmid = []
        for row in systematic_review_reader:
            systematic_review_id2pmid.append(row['pmid'])
        systematic_review_pmid2id = {pmid: i for i, pmid in enumerate(systematic_review_id2pmid)}

        print(len(systematic_review_id2pmid), len(systematic_review_pmid2id))

        with open(os.path.join(output_path, 'systematic_review_pmid2id.json'), 'w') as fout:
            json.dump(systematic_review_pmid2id, fout)
        with open(os.path.join(output_path, 'systematic_review_id2pmid.json'), 'w') as fout:
            json.dump(systematic_review_id2pmid, fout)
        print('Data Processing Done, Start building model...', datetime.datetime.now())

    # pre-process mapping_keys
    with open(args.mapping_keys) as fin:
        mapping_keys = json.load(fin)

    model = MetaMap_PICO_TFIDF(clinical_trial_pico_path=args.clinical_trial_pico_path, mapping_keys=mapping_keys)
    print('Model build Done, Start evaluation...', datetime.datetime.now())

    if os.path.exists(os.path.join(output_path, 'evaluation.json')):
        with open(os.path.join(output_path, 'evaluation.json'), 'r') as f:
            line_count = sum(1 for _ in f)
    else:
        line_count = 0

    with open(os.path.join(output_path, 'evaluation.json'), 'a', buffering=1000) as fout:
        systematic_review_trial_conenctions = []
        for i in range(len(systematic_review_id2pmid)):
            systematic_review_pmid = systematic_review_id2pmid[i]

            systematic_review_trial_conenctions.append({
                'pmid': systematic_review_pmid,
                'id': i,
                'connections': []
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

            tmp_clinical_trial_pico = []

            for systematic_review_info in systematic_review_pico:
                cpico_info = []
                for sk, ck in mapping_keys:
                    tmp = []
                    for k in sk:
                        tmp.extend([kd.get('name', '') for kd in systematic_review_info.get(k, [])])
                    cpico_info.append(tmp)
                tmp_clinical_trial_pico.append(cpico_info)

            systematic_review_pico = tmp_clinical_trial_pico
            del tmp_clinical_trial_pico

        query_corpus_dict = {'pico': []}
        for systematic_review_info in systematic_review_pico:
            query_corpus_dict['pico'].append(systematic_review_info[0])

        for scount, systematic_review_instance in enumerate(systematic_review_trial_conenctions):
            if scount < line_count: continue
            if len(query_corpus_dict['pico'][systematic_review_instance['id']]) == 0:
                systematic_review_instance['rank_idx'] = []
                systematic_review_instance['cosine_similarities'] = []
            else:
                cosine_similarities, rank_idx = model.one_to_all_cosine_similarity_infer(
                    {'pico': query_corpus_dict['pico'][systematic_review_instance['id']]})['pico']
                rank_idx = rank_idx.tolist()
                systematic_review_instance['rank_idx'] = rank_idx
                systematic_review_instance['cosine_similarities'] = cosine_similarities

            json.dump(systematic_review_instance, fout)
            fout.write('\n')
    print('Evaluation done...', datetime.datetime.now())


if __name__ == "__main__":
    main()
