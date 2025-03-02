from setuptools import setup, find_packages

setup(
    name='pubmed_pipeline',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "pandas",
		"scispacy",
        "spacy",
        "tqdm",
        "transformers",
        "joblib",
		"biopython",
        "pandas",
        "numpy",
        "spacy",
        "torch",
        "transformers",
        "tqdm",
        "scikit-learn"
        # ajoutez d'autres dépendances si nécessaire
    ],
)
