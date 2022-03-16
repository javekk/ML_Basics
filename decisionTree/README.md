# Decision Trees and Random Forests

## Model

> DecisionTree 

Basic decision tree implementation, using `Numpy` and `Pandas`. It works both for regression and classification.

> Node

Binary tree implementation which is used as data structure for the deciosion tree.



## Use cases


> decisionTreeClassification.py

Predict breast cancer being *benign* or *malignant*, from UCi breast-cancer-wisconsin dataset. Basic usage:

```bash
$ python decisionTreeClassification.py
```


> decisionTreeClassificationExtLib.py

Similar to the previous one, predicts breast cancer being *benign* or *malignant*. But this one is implemented using `Sklearn` implementation. Usage:

```bash
$ python decisionTreeClassificationExtLib.py
```


> decisionTreeRegression.py

Predict fuel consumption expressed in *mpg* based on the auto specifics using the UCI auto dataset. Basic usage:

```bash
$ python decisionTreeRegression.py
```


> decisionTreeRegressionExtLib.py

Similar to the previous one, fuel consumption expressed in *mpg* based on the auto specifics. But this one is implemented using `Sklearn` implementation. Usage:

```bash
$ python decisionTreeRegressionExtLib.py
```


# Bibliography 

* [ How to program a decision tree in Python from 0 - Ander Fernandez Jauregui](https://anderfernandez.com/en/blog/code-decision-tree-python-from-scratch/)

* [ StatQuest: Decision Trees - StatQuest with Josh Starmer ](https://www.youtube.com/watch?v=7VeUPuFGJHk)

* [ UCI Machine Learning Repository - Breast Cancer Wisconsin (Original) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29)

* [ UCI Machine Learning Repository - Auto MPG Dataset](https://archive-beta.ics.uci.edu/ml/datasets/auto+mpg)
