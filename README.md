# POS-tagger
POS tagger for the English language made with datasets from Universal Dependencies.

# 🚀 Getting started

It is good practice to run all commands and installations inside a python environment. Check out [this page](https://docs.python.org/3/library/venv.html) to know how.

To install the dependencies to reproduce this project, just run:

````
    pip install -r dependencies.txt
````

The analysis and findings are recommended in the following order:

1. **Exploratory Data Analysis** - To check out the data we have used to train and test our model. It is found inside the folder `eda/`. It is not recommended to re-run the execution of the notebooks since some of the plots and analysis might take a while to load. However, the results are already available and visible in the notebook itself. It is also recommended to check them with some notebook visualizator, since github does not show some of the interactive plots generated.

2. **The algorithm** - A class has been implemented to use the pos-tagger with the viterbi implementation. The code is inside the folder `src/`. There, you can also find some visualization and preprocessing files with different functionalities.

3. **Evaluation** - To check out the results provided by our pos-tagger implementation. We analyze some metrics using the test data explored in section 1. It is found inside the folder `evaluation/`.


# 📝 The data 

* The data used has been extracted from the resources provided by the [Universal Dependencies Project](https://universaldependencies.org/).
* For the analysis, two datasets from different languages have been used: English and Catalan.

## English Dataset
* The corpus is referred as GUM.
* Github repository available [here](https://github.com/UniversalDependencies/UD_English-GUM/tree/master)
* Train dataset available [here](https://github.com/UniversalDependencies/UD_English-GUM/blob/master/en_gum-ud-train.conllu) (8548)
* Test dataset available [here](https://github.com/UniversalDependencies/UD_English-GUM/blob/master/en_gum-ud-test.conllu) (1096)

## Catalan Dataset
* Sentences from the corpus [Ancora](https://clic.ub.edu/corpus/)
* Github repository available [here](https://github.com/UniversalDependencies/UD_Catalan-AnCora/tree/master)
* Train dataset available [here](https://github.com/UniversalDependencies/UD_Catalan-AnCora/blob/master/ca_ancora-ud-train.conllu) (13123 sentences)
* Test dataset available [here](https://github.com/UniversalDependencies/UD_Catalan-AnCora/blob/master/ca_ancora-ud-test.conllu) (1846 sentences)


# To delete :


# 💡 Ideas for the analysis
Putting it here atm to have it in mind:

* Analyze results with the datasets, plot matrixes with the happy path and probabilities
    * Compare probabilities when a result is correct
    * Compare probabilities when a result is incorrect
* Analyze complexity, compare with the nltk HMM method.
* Compare with multiplication and addition maybe?
* Compare using matrices vs dicts

# 📝 Tasks
- [ ] Performance analysis (computationally-wise)
- [ ] Performance analysis (algorithm quality-wise: e.g. precision, recall, etc -> 2 languages datasets, how making an error propagates errors)
- [ ] EDA (Exploratory data analysis)