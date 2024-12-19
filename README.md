# kaggle-depression

This project aims to predict mental health outcomes based on a collection of patient features. This uses the [Depression Dataset from kaggle.com](https://www.kaggle.com/datasets/anthonytherrien/depression-dataset/data). The project is built and tested using Python 3.12.7 and MacOS Sequoia.

## Repo Structure
```bash
.
├── resources
│   ├── data_inference.csv
│   ├── data_train.csv
│   ├── depression_data.csv
│   └── model.pkl
├── src 
│   ├── evaluate_classifier.py 
│   ├── inference.py 
│   └── train_classifier.py 
├── tests 
│   ├── data_prep.py 
│   ├── train_classifier.py 
│   └── evaluate_classifier.py 
├── depression-model.ipynb 
├── requirements.txt 
└── README.md
```

**resources** contains input and output files utilised or created by the modelling. <br /> 
**src** contains python scripts to train the model, evaluate its performance and infer on unseen data. This folder also contains **src/depression-model.ipynb** which hosts exploratory data analysis and modelling, used to inform the model and features chosen as an initial solution to this project <br /> 
**tests** contains unit testing for the custom functions used within the deployable code. <br /> 
**requirements.txt** contains python package requirements to run the solution.


## EDA and Feature Engineering
Manual data exploration and analysis is undertaken in `depression-model.ipynb`. Here I have aimed to understand the distribution and characteristics of each features and how they correlate to each other and the target.

For many of the features provided, there is a little correlation to the target, indicating model performance is unlikely to be strong. Unemployment and Income are the most obvious predictive features, with other factors that may be reasonably expected to be predictive such as family history and lifestyle indicators not highly correlated. 


## Modelling Approach
I have chosen to tackle this problem with a Logistic Regression, as initial testing of alternative more complex approaches including tree-based models and a basic neural network did not show any improvement in accuracy (see `depression-model.ipynb`) so maintaining a simpler model is beneficial for explainability and processing speed.

Features for the Logistic Regression are chosen based on correlation to the target, and low cross-correlation with other features.


## Deployable Code and Testing
Whilst data exploration, visualisation and testing of modelling approaches is visible in the notebook, I have refactored the code that is ultimately required for the solution into python scripts in `src`. These are split into two scipts:
- **Training and Evaluation**: Based on the learnings of the exploratory analysis and modelling, this script trains and saves a scikit-learn pipeline along with an evaluation of performance. If this solution was deployed in a mature cloud computing platform like Microsoft Azure, it may be preferable to save this model by registering it within that platform rather than storing the model object locally.
- **Inference**: Using the saved model pipeline from the training step, this script applies the model to unseen inference data and saves its predictions. 

These are seperated to allow for more different scheduling on re-training and inference in a production setting. Data preparation required on the raw input files is done within the Python scripts for now, but this would ideally be hosted within a data warehouse using dbt, PySpark or similar tools.

Unit tests are also set up to check that the essential model training and inference custom functions within these codes provide expected outputs. These use the native Python `unittest` package, and can be triggered from the root folder of the repo by running `python -m unittest`.

To fully utilise the solution presented within this repo:

1. Set up a virtual environment and install the package requirements in `requirements.txt`
2. Run `src/train_eval.py` to train and review performance of the model
3. Run `src/inference.py` to showcase inference on a small test dataset


## Model Performance, Considerations and Improvements
Model discrimination is currently fairly poor, with key metrics indicating performance that is better than baseline (an uninformed guess) but not strong. Whether this can translate to usable insight would depend on the use case - we would not be able to use this model in its current iteration to confidently state whether a person has experienced a history of mental illness, but it may be able to be used in less deterministic contexts such as a factor in pricing for health and life insurance products.

![alt text](https://github.com/oli-tailby/kaggle-depression/blob/main//resources/confusion_matrix.png?raw=true)

We also need to consider the following when reviewing model performance or using outputs:
- Whilst this data is synthetic, real-world comparable data may contain biases which would be picked up by the modelling. In this scenario, I anticipate the validity of a "History of Mental Illness" outcome could be affected by factors such as age and gender impacting a person's likelihood to self-report past mental illness or seek support for a condition.
- In a realistic dataset, we would expect to see some missing data, perhaps from opt-outs to optional fields or other data quality issues. Imputation or removal hasn't been needed here, but may mean a dataset isn't fully representative of the required population.
- We have no visibility on how this data has been collected, so cannot be certain it is representative of the wider population or any particular customer base. For example, there is no training data on individuals under 18 or over 80, which may not remain the case for the inference population and therefore could limit prediction accuracy.

I believe the most significant improvement in a model like this would be collection of further features. Additional demographic data including home/work location, occupation etc. could help to build further useful predictive features on environment, wealth and lifestyle. I anticipate the most predictive features for a model like this would be based around each patient's medical background, though that might not be possible in the context of this question.

More complex modelling methods (eg. neural networks) could also help to extract further accuracy, although given the relatively small number of features and dataset size I wouldn't expect performance improvements to be substantial. Further parameter tuning and feature engineering on the existing data could also be helpful.
