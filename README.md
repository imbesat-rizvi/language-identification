# language-identification

![Language identification prediction demo](demo/language-identification.gif)

A machine learning model for language identification. The model is trained using the huggingface dataset "[papluca/language-identification](https://huggingface.co/datasets/papluca/language-identification)". 

The dataset was created by collecting data from 3 sources: [Multilingual Amazon Reviews Corpus](https://huggingface.co/datasets/amazon_reviews_multi), [XNLI](https://huggingface.co/datasets/xnli), and [STSb Multi MT](https://huggingface.co/datasets/stsb_multi_mt).

The dataset contains text in the following 20 languages:

`
arabic (ar), bulgarian (bg), german (de), modern greek (el), english (en), spanish (es), french (fr), hindi (hi), italian (it), japanese (ja), dutch (nl), polish (pl), portuguese (pt), russian (ru), swahili (sw), thai (th), turkish (tr), urdu (ur), vietnamese (vi), and chinese (zh)
`

##### Model description and training:

The language text is converted into a feature vector using a [TFIDF vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) which can be fit on either word, character ('char') or character within word boundaries ('char\_wb') level. The vectorizer can also be fit on various n-gram ranges. The best performance was achieved with 'char\_wb' and an n-gram range of (3,4).

A Multinomial Naive Bayes ([MultinomialNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)) model is trained for multi-class (20-language) classification with a smoothening factor (alpha) of 1. Other models were tried too but they were comparatively slow and didn't lead to any significant increase in the accuracy.

Even with such a simple setting, the performance achieved was 99.22% accuracy on the test set of the [papluca/language-identification](https://huggingface.co/datasets/papluca/language-identification) dataset. The achieved accuracy is competitive to the reported accuracy of 99.60% on the test set using their [papluca/xlm-roberta-base-language-detection](https://huggingface.co/papluca/xlm-roberta-base-language-detection) model which is a finetuned version of the [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) model.

To train and save the TFIDF featurizer and MultinomialNB model, execute:
```
python3 train.py --save_dir train_out
```

The python version used is 3.8

##### Prediction and accounting for unseen languages during training:

The trained model will only return prediction and probabilities for the 20 languages seen during training. Thus, using the model as is will incorrectly classify to one of those seen language categories even when the new text belongs to an unseen language. To address this, we use probability thresholding on the class with highest probability and report that class only when its probability is more than the threshold, otherwise report the class to be 'Other' i.e. such a language was not seen during training. 

This probability threshold can be played around with for a desired precision or recall value, as increasing this threshold will increase precision but some seen language examples can be classified as 'Other', while decreasing this threshold is going to increase the recall but more and more 'unseen' languages will get incorrectly classified to one of the 20 seen languages. A probability threshold of 0.3 seemed to work well for the given dataset. In the [demo](demo/language-identification.gif) shown at the top, the last example was a gibberish input which was correctly classified as 'Other' i.e. none of the seen languages.

To continually predict languages for input texts on the command line using the trained model, execute:
```
python3 predict.py --load_dir train_out --proba_th 0.3
```