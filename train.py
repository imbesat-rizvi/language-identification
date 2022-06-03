import pickle
from pathlib import Path
from argparse import ArgumentParser

from datasets import load_dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from utils import preprocess_text


def train(
    analyzer="char_wb", 
    ngram_range=(3,4), 
    save_dir="train_out",
    alpha=1,
):

    dataset_name = "papluca/language-identification"
    lang_dataset = load_dataset(dataset_name)
    lang_dataset = lang_dataset.map(lambda x: {"text": preprocess_text(x["text"])})

    # initialize and fit vectorizer
    vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range)
    X_train = vectorizer.fit_transform(lang_dataset["train"]["text"])

    # initialize and fit label encoder
    le = LabelEncoder()
    y_train = le.fit_transform(lang_dataset["train"]["labels"])

    # save vectorizer and label encoder
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir/"tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(save_dir/"label_encoder.pkl", "wb") as f:
        pickle.dump(le, f, protocol=pickle.HIGHEST_PROTOCOL)

    # initalize, train and save the language detection model
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train, y_train)
    with open(save_dir/"lang_detect_model.pkl", "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    for split in ("validation", "test"):
        X = vectorizer.transform(lang_dataset[split]["text"])
        y = le.transform(lang_dataset[split]["labels"])
        y_pred = model.predict(X)

        accuracy = accuracy_score(y, y_pred)
        print(f"Accuracy on {split} set: {accuracy}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Language Identification model training on the "\
        "papluca/language-identification huggingface dataset."
    )

    parser.add_argument(
        "--analyzer",
        default="char_wb",
        choices=["word", "char", "char_wb"],
        help="analyzer to be used for initializing Tfidf vectorizer",
    )
    parser.add_argument(
        "--min_ngram",
        type=int,
        default=3,
        help="min ngram to be used for Tfidf vectorizer ngram_range",
    )
    parser.add_argument(
        "--max_ngram",
        type=int,
        default=4,
        help="max ngram to be used for Tfidf vectorizer ngram_range",
    )
    parser.add_argument(
        "--save_dir",
        default="train_out",
        help="Directory to save vectorizer, label encoder and model files",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1,
        help="value of the alpha argument for the MultinomialNB model",
    )

    args = parser.parse_args()

    train(
        analyzer=args.analyzer,
        ngram_range=(args.min_ngram, args.max_ngram),
        save_dir=args.save_dir,
        alpha=args.alpha,
    )