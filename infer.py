import pickle
import numpy as np
from pathlib import Path
from pycountry import languages
from argparse import ArgumentParser

from utils import preprocess_text

def predict(texts, vectorizer, label_encoder, model, proba_th=0.5):
    if isinstance(texts, str):
        texts = [texts]
    
    texts = [preprocess_text(i) for i in texts]
    X = vectorizer.transform(texts)
    probas = model.predict_proba(X)

    # Unseen langauge during training will be reported as Other
    y = np.full(shape=len(texts), fill_value="Other")
    seen_lang_mask = probas.max(axis=1) >= proba_th
    y[seen_lang_mask] = label_encoder.inverse_transform(
        probas[seen_lang_mask].argmax(axis=1)
    )

    y = [(i if i == "Other" else languages.get(alpha_2=i).name) for i in y]
    
    return y


def infer(
    load_dir="train_out",
    proba_th=0.5,
):

    load_dir = Path(load_dir)
    with open(load_dir/"tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(load_dir/"label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    with open(load_dir/"lang_detect_model.pkl", "rb") as f:
        model = pickle.load(f)

    print("\nEnter texts for language identification. Unseen languages during "\
        "training will be reported as 'Other'."
    )

    while True:
        text = input("\nText: ")
        lang = predict(text, vectorizer, label_encoder, model, proba_th)[0]
        print("\nLanguage:", lang)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Continuous inference for language identification for languages"\
        " in the papluca/language-identification huggingface dataset. Unseen "\
        " languages will be reported as Other.",
    )

    parser.add_argument(
        "--load_dir",
        default="train_out",
        help="Directory from where vectorizer and model will be loaded",
    )
    parser.add_argument(
        "--proba_th",
        type=float,
        default=0.5,
        help="Probability threshold below which 'Other' will be reported"\
        " to avoid spurious identification in case of gibberish input",
    )

    args = parser.parse_args()

    infer(load_dir=args.load_dir, proba_th=args.proba_th)