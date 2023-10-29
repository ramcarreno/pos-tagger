# POS-tagger
POS tagger for the English language made with datasets from Universal Dependencies.

# Getting started
Once we finish state how to get started (install dependencies, etc)

# The data 

* The data used has been extracted from the resources provided by the [Universal Dependencies Project](https://universaldependencies.org/).
* For the analysis, two datasets from different languages have been used: English and Catalan.

## English Dataset
* The corpus is referred as ESLSpok in the project, but its data is from [NICT JLE](https://alaginrc.nict.go.jp/nict_jle/index_E.html), a corpus of spoken second language English
* Github repository available [here](https://github.com/UniversalDependencies/UD_English-ESLSpok/tree/master)
* Train dataset available [here](https://github.com/UniversalDependencies/UD_English-ESLSpok/blob/master/en_eslspok-ud-train.conllu) (1856 sentences)
* Test dataset available [here](https://github.com/UniversalDependencies/UD_English-ESLSpok/blob/master/en_eslspok-ud-test.conllu) (232 sentences)

## Catalan Dataset
* Sentences from the corpus [Ancora](https://clic.ub.edu/corpus/)
* Github repository available [here](https://github.com/UniversalDependencies/UD_Catalan-AnCora/tree/master)
* Train dataset available [here](https://github.com/UniversalDependencies/UD_Catalan-AnCora/blob/master/ca_ancora-ud-train.conllu) (13123 sentences)
* Test dataset available [here](https://github.com/UniversalDependencies/UD_Catalan-AnCora/blob/master/ca_ancora-ud-test.conllu) (1846 sentences)

# Ideas for the analysis
Putting it here atm to have it in mind:

* Analyze results with the datasets, plot matrixes with the happy path and probabilities
    * Compare probabilities when a result is correct
    * Compare probabilities when a result is incorrect
* Analyze complexity, compare with the nltk HMM method.
* Compare with multiplication and addition maybe?
* Compare using matrices vs dicts

# Tasks
- [ ] Build "predict" -> build the class
- [ ] Refactor code and convert to class
- [ ] Performance analysis (computationally-wise)
- [ ] Performance analysis (algorithm quality-wise: e.g. precision, recall, etc -> 2 languages datasets, how making an error propagates errors)
- [ ] EDA (Exploratory data analysis)
- [ ] ⚠️ States -> convert to list !! ⚠️

