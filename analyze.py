import ast
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

OUTPUT_FOLDER = "results"

# the names the speaker can be listed under in the transcripts
BIDEN_NAMES = ["joe biden", "president biden", "president joe biden"]
TRUMP_NAMES = ["donald trump", "president trump", "president donald trump", "president donald j. trump"]


def get_speaker_text(speech_list, politician):
    # speech_list is a string that looks like a python list of dicts
    segments = ast.literal_eval(speech_list)

    if politician == "Biden":
        names = BIDEN_NAMES
    else:
        names = TRUMP_NAMES

    speeches = []
    for segment in segments:
        # the speaker is stored in the "Time" field, fx "Joe Biden: (00:01)"
        label = str(segment.get("Time", ""))
        label = label.split(": (")[0].lower().strip()

        # some labels have an extra timestamp after the name
        for name in names:
            if label == name or label.startswith(name + " ("):
                speeches.append(str(segment.get("Speech", "")))
                break

    return " ".join(speeches)


def load_data():
    all_data = []

    for politician in ["Biden", "Trump"]:
        df = pd.read_csv(politician.lower() + ".csv")

        # only keep speeches from 2022 to 2024
        df["date"] = pd.to_datetime(df["Date Published"], errors="coerce")
        df = df[df["date"].dt.year.between(2022, 2024)].copy()

        # get only the words said by the politician
        df["text"] = df["Speech List"].apply(lambda x: get_speaker_text(x, politician))

        # remove very short transcripts (less than 50 words)
        df = df[df["text"].str.split().str.len() >= 50]

        df["politician"] = politician
        all_data.append(df[["politician", "date", "Title", "text"]])

    data = pd.concat(all_data, ignore_index=True)
    print("Number of transcripts:", data.groupby("politician").size().to_dict())
    return data


def make_tfidf(data):
    # each transcript is one document, english stopwords are removed
    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True, min_df=3, max_df=0.9)
    tfidf_matrix = vectorizer.fit_transform(data["text"])
    words = vectorizer.get_feature_names_out()
    return tfidf_matrix, words


def find_top_words(data, tfidf_matrix, words):
    labels = data["politician"].to_numpy()

    # average tfidf score of every word for each politician
    biden_mean = np.asarray(tfidf_matrix[labels == "Biden"].mean(axis=0)).ravel()
    trump_mean = np.asarray(tfidf_matrix[labels == "Trump"].mean(axis=0)).ravel()

    # words with the biggest difference are the most distinctive
    biden_diff = biden_mean - trump_mean
    trump_diff = trump_mean - biden_mean

    results = []
    for politician, diff in [("Biden", biden_diff), ("Trump", trump_diff)]:
        # indexes of the 10 highest values, highest first
        top_indexes = np.argsort(diff)[-10:][::-1]
        for i in top_indexes:
            results.append({
                "politician": politician,
                "term": words[i],
                "mean_tfidf_difference": float(diff[i]),
            })

    return pd.DataFrame(results)


def make_plot(top_words):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.5))

    for ax, politician in zip(axes, ["Biden", "Trump"]):
        subset = top_words[top_words["politician"] == politician]
        # reverse so the biggest bar is at the top
        terms = list(subset["term"])[::-1]
        values = list(subset["mean_tfidf_difference"])[::-1]

        ax.barh(terms, values)
        ax.set_title(politician + ": distinctive terms")
        ax.set_xlabel("Difference in mean TF-IDF")

    fig.suptitle("Distinctive vocabulary in speech transcripts (2022–2024)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, "distinctive_terms.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_results(top_words):
    top_words.to_csv(os.path.join(OUTPUT_FOLDER, "top_terms.csv"), index=False)

    for politician in ["Biden", "Trump"]:
        subset = top_words[top_words["politician"] == politician]
        print(politician, list(subset["term"]))


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    data = load_data()
    tfidf_matrix, words = make_tfidf(data)
    top_words = find_top_words(data, tfidf_matrix, words)
    make_plot(top_words)
    save_results(top_words)


if __name__ == "__main__":
    main()