packages = ['torch', 'transformers', 'datasets', 'evaluate', 'accelerate', 'sklearn', 'pandas', 'numpy', 'emoji', 'streamlit', 'plotly', 'tqdm', 'joblib', 'spacy', 'nltk']

for pkg in packages:
    try:
        m = __import__(pkg)
        print(f"{pkg:15s}: OK ({getattr(m, '__version__', 'installed')})")
    except ImportError:
        print(f"{pkg:15s}: MISSING")
