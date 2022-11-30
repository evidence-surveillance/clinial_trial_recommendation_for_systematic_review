# tfidf
python src/main_tfidf_doc_sim_predict.py --systematic_review_path data/systematic_review/manual_systematic_reviews.csv \
                                         --clinical_trial_path data/clinical_trial_article/extended_clinical_trials.csv \
                                         --output_path exps/SRA2CTP/manual/manual_tfidf/ \
                                         >> logs/SRA2CTP_manual_tfidf.log 2>&1 &

python src/main_tfidf_doc_sim_predict.py --systematic_review_path data/systematic_review/manual_systematic_reviews.csv \
                                         --clinical_trial_path data/clinical_trial_registration_pico_text/clinical_trials_pico_text.csv \
                                         --output_path exps/SRA2CTR/manual/manual_tfidf/ \
                                         >> logs/SRA2CTR_pico_text_manual_tfidf.log 2>&1 &

python src/main_tfidf_doc_sim_predict.py --systematic_review_path data/prospero/manual_prospero_systematic_reviews.csv \
                                         --clinical_trial_path data/clinical_trial_article/extended_clinical_trials.csv \
                                         --output_path exps/SRR2CTP/manual/manual_tfidf/ \
                                         >> logs/SRR2CTP_manual_tfidf.log 2>&1 &

python src/main_tfidf_doc_sim_predict.py --systematic_review_path data/prospero/manual_prospero_systematic_reviews.csv \
                                         --clinical_trial_path data/clinical_trial_registration_pico_text/clinical_trials_pico_text.csv \
                                         --output_path exps/SRR2CTR/manual/manual_tfidf/ \
                                         >> logs/SRR2CTR_pico_text_manual_tfidf.log 2>&1 &

# bert pico tfidf main_bert_pico_tfidf_doc_sim_predict
python src/main_bert_pico_tfidf_doc_sim_predict.py --systematic_review_path data/systematic_review/manual_systematic_reviews.csv \
                                --clinical_trial_path data/clinical_trial_article/extended_clinical_trials.csv \
                                --systematic_review_pico_path data/systematic_review/manual_systematic_reviews.json \
                                --clinical_trial_pico_path data/clinical_trial_article/extended_clinical_trials.json \
                                --output_path exps/SRA2CTP/manual/manual_pico_tfidf/ \
                                --score_func pico \
                                >> logs/SRA2CTP_manual_pico_pico_tfidf.log 2>&1 &
python src/main_bert_pico_tfidf_doc_sim_predict.py --systematic_review_path data/systematic_review/manual_systematic_reviews.csv \
                                --clinical_trial_path data/clinical_trial_registration_pico_text/clinical_trials_pico_text.csv \
                                --systematic_review_pico_path data/systematic_review/manual_systematic_reviews.json \
                                --clinical_trial_pico_path data/clinical_trial_registration_pico_text/clinical_trials_pico_text.json \
                                --output_path exps/SRA2CTR/manual/manual_pico_tfidf/ \
                                --score_func pico \
                                >> logs/SRA2CTR_manual_pico_pico_tfidf.log 2>&1 &
python src/main_bert_pico_tfidf_doc_sim_predict.py --systematic_review_path data/prospero/manual_prospero_systematic_reviews.csv \
                                --clinical_trial_path data/clinical_trial_article/extended_clinical_trials.csv \
                                --systematic_review_pico_path data/prospero/manual_prospero_systematic_reviews.json \
                                --clinical_trial_pico_path data/clinical_trial_article/extended_clinical_trials.json \
                                --output_path exps/SRR2CTP/manual/manual_pico_tfidf/ \
                                --score_func pico \
                                >> logs/SRR2CTP_manual_pico_pico_tfidf.log 2>&1 &
python src/main_bert_pico_tfidf_doc_sim_predict.py --systematic_review_path data/prospero/manual_prospero_systematic_reviews.csv \
                                --clinical_trial_path data/clinical_trial_registration_pico_text/clinical_trials_pico_text.csv \
                                --systematic_review_pico_path data/prospero/manual_prospero_systematic_reviews.json \
                                --clinical_trial_pico_path data/clinical_trial_registration_pico_text/clinical_trials_pico_text.json \
                                --output_path exps/SRR2CTR/manual/manual_pico_tfidf/ \
                                --score_func pico \
                                >> logs/SRR2CTR_manual_pico_pico_tfidf.log 2>&1 &

# bert pico metamap tfidf main_bert_pico_tfidf_doc_sim_predict
python src/main_bert_pico_tfidf_doc_sim_predict.py --systematic_review_path data/systematic_review/manual_systematic_reviews.csv \
                                --clinical_trial_path data/clinical_trial_article/extended_clinical_trials.csv \
                                --systematic_review_pico_path data/systematic_review/manual_systematic_reviews.json \
                                --clinical_trial_pico_path data/clinical_trial_article/extended_clinical_trials.json \
                                --output_path exps/SRA2CTP/manual/manual_pico_tfidf/ \
                                --score_func metamap_pico \
                                >> logs/SRA2CTP_manual_pico_meta_pico_tfidf.log 2>&1 &

python src/main_bert_pico_tfidf_doc_sim_predict.py --systematic_review_path data/systematic_review/manual_systematic_reviews.csv \
                                --clinical_trial_path data/clinical_trial_registration_pico_text/clinical_trials_pico_text.csv \
                                --systematic_review_pico_path data/systematic_review/manual_systematic_reviews.json \
                                --clinical_trial_pico_path data/clinical_trial_registration_pico_text/clinical_trials_pico_text.json \
                                --output_path exps/SRA2CTR/manual/manual_pico_tfidf/ \
                                --score_func metamap_pico \
                                >> logs/SRA2CTR_manual_pico_meta_pico_tfidf.log 2>&1 &

python src/main_bert_pico_tfidf_doc_sim_predict.py --systematic_review_path data/prospero/manual_prospero_systematic_reviews.csv \
                                --clinical_trial_path data/clinical_trial_article/extended_clinical_trials.csv \
                                --systematic_review_pico_path data/prospero/manual_prospero_systematic_reviews.json \
                                --clinical_trial_pico_path data/clinical_trial_article/extended_clinical_trials.json \
                                --output_path exps/SRR2CTP/manual/manual_pico_tfidf/ \
                                --score_func metamap_pico \
                                >> logs/SRR2CTP_manual_pico_meta_pico_tfidf.log 2>&1 &

python src/main_bert_pico_tfidf_doc_sim_predict.py --systematic_review_path data/prospero/manual_prospero_systematic_reviews.csv \
                                --clinical_trial_path data/clinical_trial_registration_pico_text/clinical_trials_pico_text.csv \
                                --systematic_review_pico_path data/prospero/manual_prospero_systematic_reviews.json \
                                --clinical_trial_pico_path data/clinical_trial_registration_pico_text/clinical_trials_pico_text.json \
                                --output_path exps/SRR2CTR/manual/manual_pico_tfidf/ \
                                --score_func metamap_pico \
                                >> logs/SRR2CTR_manual_pico_meta_pico_tfidf.log 2>&1 &
# metamap tfidf main_metamap_pico_tfidf_doc_sim_predict
python src/main_metamap_pico_tfidf_doc_sim_predict.py --systematic_review_path data/systematic_review/manual_systematic_reviews.csv \
                                --clinical_trial_path data/clinical_trial_article/extended_clinical_trials.csv \
                                --systematic_review_pico_path data/systematic_review/manual_systematic_reviews_metamap.json \
                                --clinical_trial_pico_path data/clinical_trial_article/extended_clinical_trials_metamap.json \
                                --output_path exps/SRA2CTP/manual/manual_metamap_tfidf/ \
                                --mapping_keys mapping_keys/sra2ctp.json \
                                >> logs/SRA2CTP_manual_metamap_tfidf.log 2>&1 &

python src/main_metamap_pico_tfidf_doc_sim_predict.py --systematic_review_path data/systematic_review/manual_systematic_reviews.csv \
                                --clinical_trial_path data/clinical_trial_registration_pico_text/clinical_trials_pico_text.csv \
                                --systematic_review_pico_path data/systematic_review/manual_systematic_reviews_metamap.json \
                                --clinical_trial_pico_path data/clinical_trial_registration_pico_text/clinical_trials_pico_metamap.json \
                                --output_path exps/SRA2CTR/manual/manual_metamap_tfidf/ \
                                --mapping_keys mapping_keys/sra2ctr.json \
                                >> logs/SRA2CTR_manual_metamap_tfidf.log 2>&1 &

python src/main_metamap_pico_tfidf_doc_sim_predict.py --systematic_review_path data/prospero/manual_prospero_systematic_reviews.csv \
                                --clinical_trial_path data/clinical_trial_article/extended_clinical_trials.csv \
                                --systematic_review_pico_path data/prospero/manual_prospero_systematic_reviews_metamap.json \
                                --clinical_trial_pico_path data/clinical_trial_article/extended_clinical_trials_metamap.json \
                                --output_path exps/SRR2CTP/manual/manual_metamap_tfidf/ \
                                --mapping_keys mapping_keys/srr2ctp.json \
                                >> logs/SRR2CTP_manual_metamap_tfidf.log 2>&1 &

python src/main_metamap_pico_tfidf_doc_sim_predict.py --systematic_review_path data/prospero/manual_prospero_systematic_reviews.csv \
                                --clinical_trial_path data/clinical_trial_registration_pico_text/clinical_trials_pico_text.csv \
                                --systematic_review_pico_path data/prospero/manual_prospero_systematic_reviews_metamap.json \
                                --clinical_trial_pico_path data/clinical_trial_registration_pico_text/clinical_trials_pico_metamap.json \
                                --output_path exps/SRR2CTR/manual/manual_metamap_tfidf/ \
                                --mapping_keys mapping_keys/srr2ctr.json \
                                >> logs/SRR2CTR_manual_metamap_tfidf.log 2>&1 &
