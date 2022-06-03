# language-identification

![Language identification inference demo](demo/language-identification.mp4)

Tfidf feature based machine learning model for language identification. The model is trained using the huggingface dataset "[papluca/language-identification](https://huggingface.co/datasets/papluca/language-identification)". 

The dataset was created by collecting data from 3 sources: [Multilingual Amazon Reviews Corpus](https://huggingface.co/datasets/amazon_reviews_multi), [XNLI](https://huggingface.co/datasets/xnli), and [STSb Multi MT](https://huggingface.co/datasets/stsb_multi_mt).

The dataset contains text in 20 languages, which are:

`
arabic (ar), bulgarian (bg), german (de), modern greek (el), english (en), spanish (es), french (fr), hindi (hi), italian (it), japanese (ja), dutch (nl), polish (pl), portuguese (pt), russian (ru), swahili (sw), thai (th), turkish (tr), urdu (ur), vietnamese (vi), and chinese (zh)
`

To train the model, execute:
```
python3 train.py --save_dir train_out
```

To infer using the trained model, execute:
```
python3 infer.py --load_dir train_dir --proba_th 0.3
```