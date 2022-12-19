# NATbase
NATbase is a an implementation of vanilla non-autoregressive neural manchine translation. In the code implementation, 
I refer to fairseq that is a sequence modeling toolkit developped by facebook.


# Preprocess
Use preprocess.py to generate vocabulary(dictionary) files and binary token index files from language pair corpus.


# Train
Use train.py with appropriate command line arguments to train your model.

eg: python train.py ./data-bin/distilled_data/iwslt14de-en
./check_points/distilled_data/iwslt14de-en_tgt_emb_baseline
--task
translation
--criterion
label_smoothed_cross_entropy
length_criterion
--lr-scheduler
linear_lr_scheduler
--src_lang
de
--tgt_lang
en
--label_smoothing
0.1
--save-period
100
--dev-chunk-size
1
--log-interval
200
--encoder_embed_dim
256
--encoder_ffn_embed_dim
512
--encoder_layers
5
--encoder_attention_heads
4
--decoder_embed_dim
256
--decoder_ffn_embed_dim
512
--decoder_layers
5
--decoder_attention_heads
4
--dropout
0.3
--apply_bert_init
--decoder_input_how
interpolate
--pred_length
--save-after-epoch
20
--max-tokens
4096
--performance-indicator
BLEU
--max-update
250000


# Inference 
Use generate.py to generate results with trained model from test dataset.
