# Sources and dependencies

SALlog's LogCMA detector, training/threshold logic, calling rule, reference selection, knowledge resources and acceptance logic were migrated from this research project's implementations into an independent package. The coverage-selection and normalization implementations retain their algorithms; paths and entry points were decoupled from the previous research workspace.

External dependencies are installed separately and retain their respective licenses: PyTorch, Hugging Face Transformers, NumPy, SciPy, scikit-learn, PyYAML, Joblib and Threadpoolctl. No dependency source distribution is vendored.

- The template encoder is [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5). Model weights are not distributed in this repository; consult its model card and license when obtaining them.
- Dataset sources include [Loghub](https://github.com/logpai/loghub). Datasets and dataset labels are not distributed here.
- Original parsing used [LogPAI Logparser / Drain](https://github.com/logpai/logparser). No parser implementation is distributed here.
- Public mechanism documents used for knowledge-card curation are linked in the individual cards. Their citations describe factual sources and do not relicense those documents.

This repository does not contain implementations of comparison methods or their result files.

