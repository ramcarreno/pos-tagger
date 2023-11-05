# POS-tagger
POS tagger for the English language made with datasets from Universal Dependencies.

# 🚀 Getting started

It is good practice to run all commands and installations inside a python environment. Check out [this page](https://docs.python.org/3/library/venv.html) to know how.

To install the dependencies to reproduce this project, just run:

````
    pip install -r requirements.txt
````

To use this implementation of the pos tagger, you just have to use our class!

```python
>>> from src.tagger import HiddenMarkovModel
>>> from src.scrapper import parse_conllu_file

>>> train = parse_conllu_file(filepath="datasets/en_gum-ud-train.conllu")
>>> test = [[('hello', ) , ('world', )]] 

>>> tagger.predict(corpus=test)
[[('hello', 'intj'), ('world', 'noun')]]

```

Take a look at our analysis to check how to play around with all the functionalities: `viterbi_best_path`, `get_confusion_matrix`, etc!

# The project

In this repo we don't only provide the code to use your own POS-Tagger but also a a set of analysis performed in two datasets: exploratory data analysis, performance evaluation and a bit of algorithm profiling.  The analysis and findings are recommended in the following order:

1. **Exploratory Data Analysis** - To check out the data we have used to train and test our model. It is found inside the folder `eda/`. It is not recommended to re-run the execution of the notebooks since some of the plots and analysis might take a while to load. However, the results are already available and visible in the notebook itself. It is also recommended to check them with some notebook visualizator, since github does not show some of the interactive plots generated.

2. **The algorithm** - A class has been implemented to use the pos-tagger with the viterbi implementation. The code is inside the folder `src/`. There, you can also find some visualization and preprocessing files with different functionalities.

3. **Evaluation** - To check out the results provided by our pos-tagger implementation. We analyze some metrics using the test data explored in section 1. It is found inside the folder `evaluation/`.


# 📝 The data 

* The data used has been extracted from the resources provided by the [Universal Dependencies Project](https://universaldependencies.org/).
* For the analysis, two datasets from different languages have been used: English and Catalan.

## English Dataset
* The corpus is referred as GUM, Georgetown University Multilayer corpus. [NICT JLE](https://gucorpling.org/gum/index.html). Its purpose is to research on discourse models and therefore it contains mutliple text types and might include code-switching.
* Github repository available [here](https://github.com/UniversalDependencies/UD_English-GUM/blob/master)
* Train dataset available [here](https://github.com/UniversalDependencies/UD_English-GUM/blob/master/en_gum-ud-train.conllu) (8548 sentences)
* Test dataset available [here](https://github.com/UniversalDependencies/UD_English-GUM/blob/master/en_gum-ud-test.conllu) (1096 sentences)

## Catalan Dataset
* Sentences from the corpus [Ancora](https://clic.ub.edu/corpus/)
* Github repository available [here](https://github.com/UniversalDependencies/UD_Catalan-AnCora/tree/master)
* Train dataset available [here](https://github.com/UniversalDependencies/UD_Catalan-AnCora/blob/master/ca_ancora-ud-train.conllu) (13123 sentences)
* Test dataset available [here](https://github.com/UniversalDependencies/UD_Catalan-AnCora/blob/master/ca_ancora-ud-test.conllu) (1846 sentences)
