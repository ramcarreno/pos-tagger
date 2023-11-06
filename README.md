# POS-tagger
POS tagger for any language made with datasets from Universal Dependencies.

# 🚀 Getting started

It is good practice to run all commands and installations inside a python environment. Check out [this page](https://docs.python.org/3/library/venv.html) to know how.

To install the dependencies to reproduce this project, just run in the terminal:

````
    pip install -r requirements.txt
````

To use this implementation of the pos tagger, you just have to use our class! Inside a python terminal run:

```python
>>> from src.tagger import HiddenMarkovModel
>>> from src.scrapper import parse_conllu_file

>>> train = parse_conllu_file(filepath="datasets/en_gum-ud-train.conllu")
>>> tagger = HiddenMarkovModel(corpus=train).train()

>>> test = [[('hello', ) , ('world', )]] 
>>> tagger.predict(corpus=test)
[[('hello', 'intj'), ('world', 'noun')]]

```
Read the documentation of the methods present in src/tagger.py to understand formatting, printing and how the input arguments work. 
You can also play around with several methods present in the classes HiddenMarkovModel and HiddenMarkovModelTagger such as `viterbi_best_path`, `get_confusion_matrix`, et cetera.

# 📌 The project

In this repo we don't only provide the code to use your own POS-Tagger but also a a set of analyses performed in two datasets: exploratory data analysis, performance evaluation and a bit of algorithm profiling and cost assessment. The analyses and findings are recommended in the following order:

1. **Exploratory Data Analysis** - To check out the data we have used to train and test our model. It is found inside the folder `eda/`. It is not recommended to re-run the execution of the notebooks since some of the plots and analysis might take a while to load. However, the results are already available and visible in the notebook itself. It is also recommended to check them with some notebook visualizator, since GitHub does not show some of the interactive plots generated. 

2. **The algorithm** - The class HiddenMarkovModelTagger has been implemented to use the pos-tagger with the viterbi implementation. It is wrapped by the class HiddenMarkovModel, which takes corpus data and returns a HiddenMarkovModelTagger instance intended for general use. The code is inside the folder `src/`. There, you can also find some scrapping and visualization/plotting modules with different functionalities.

3. **Evaluation** - Notebooks where you can check out the results provided by our tagger implementation, validate the code, etc. It is found inside the folder `evaluation/`. Below it is listed a guide for all of the notebooks present in it.
   * _model_testing.ipynb_ -> Very simple, lightweight analysis demonstrating the tagger works for simple sentences, validating transition & emission matrices consistency, etc.
   * _english_model_evaluation.ipynb_ -> Performance analysis of the tagger model trained using the english language dataset, along with some relevant metrics
   * _catalan_model_evaluation.ipynb_ -> Performance analysis of the tagger model trained using the catalan language dataset, along with some relevant metrics
   * _cost_analysis.ipynb_ -> Computational cost of training analysis & trivia.
  
⚠️ It is recommended to visualize the notebooks in a local environment, since github does not display all the plots we include.

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
