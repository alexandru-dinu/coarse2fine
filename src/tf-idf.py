import os
from sklearn.feature_extraction.text import TfidfVectorizer


def load_files(directory: str, vocabulary_size: int):
	files = os.listdir(directory)
	files = [os.path.join(directory, file) for file in files]

	tfidf = TfidfVectorizer(
		input="filename",
		decode_error="ignore",
		analyzer="word",
		stop_words="english",
		token_pattern='[a-zA-Z]+',
		max_features=vocabulary_size,
		min_df=0.01
	)

	tfidf.fit(files)
	return tfidf.get_feature_names()


if __name__ == "__main__":
	top_words = load_files("computer science", 20000)
	print(top_words)
