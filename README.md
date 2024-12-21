# kaggle-depression

This project aims to predict mental health outcomes based on a collection of patient features. This uses the [Depression Dataset from kaggle.com](https://www.kaggle.com/datasets/anthonytherrien/depression-dataset/data). The project is built and tested using Python 3.12.7 and MacOS Sequoia.

## Repo Structure
```bash
.
├── resources
│   ├── depression_data.csv
│   ├── inference_data.csv
│   ├── output_inference.csv
│   └── ...
├── src 
│   ├── depression-model.py 
│   ├── feature_prep.py 
│   ├── helper_functions.py 
│   ├── evaluate_classifier.py 
│   ├── inference.py 
│   └── train_eval.py 
├── tests 
│   ├── test_helpers.py 
│   ├── test_feature_prep.py 
├── requirements.txt 
└── README.md
```

**resources** contains input and output files utilised or created by the modelling. <br /> 
**src** contains python scripts to train/tune the model, evaluate its performance and infer on unseen data. This folder also contains `src/depression-model.ipynb` which hosts exploratory data analysis and modelling, used to inform the model and features chosen as an initial solution to this project <br /> 
**test** contains unit testing for the custom functions used within the deployable code. <br /> 
**requirements.txt** contains python package requirements to run the solution.


## EDA and Feature Engineering
Manual data exploration and analysis is undertaken in `depression-model.ipynb`. Here I have aimed to understand the distribution and characteristics of each features and how they correlate to each other and the target.

For many of the features provided, there is a little correlation to the target, indicating model performance is unlikely to be strong. Unemployment and Income are the most obvious predictive features, with other factors that may be reasonably expected to be predictive such as family history and lifestyle indicators not highly correlated. 

There is no missing or notable outlier data to draw attention to in this set, so I have not needed to apply any removals or imputation. 


## Modelling Approach and Feature Selection
I have chosen to tackle this problem with a Logistic Regression, as initial testing of alternative more complex approaches including tree-based models and a basic neural network did not show any improvement in accuracy (see `depression-model.ipynb`) so maintaining a simpler model is beneficial for explainability and processing speed.

Features for the Logistic Regression are chosen based on correlation to the target, and low cross-correlation with other features to avoid problems with multicolinearity. I have removed features with especially low correlation to the target, as this set of reduced features makes the model simpler and does not appear to negatively impact performance.

I have also developed a few additional features beyond those immediately available in the source data. These include transformations of existing features and interaction features combining multiple other features, based on speculation as to possible important factors. The new interaction-based features show some correlation to the target and be included in the modelling, but are not as strong as some of the most relevant pre-existing variables such as Income and Employment. 


## Deployable Code and Testing
Whilst data exploration, visualisation and testing of modelling approaches is visible in the notebook, I have refactored the code that is ultimately required for the solution into python scripts in `src`. These are split into two top-level scripts:
- **Training and Evaluation**: Based on the learnings of the exploratory analysis and modelling, this script trains and saves a scikit-learn pipeline along with an evaluation of performance. If this solution was deployed in a mature cloud computing platform like Microsoft Azure, it may be preferable to save this model by registering it within that platform rather than storing the model object locally.
- **Inference**: Using the saved model pipeline from the training step, this script applies the model to unseen inference data and saves its predictions. 

These are seperated to allow for different scheduling frequency of training and inference in a production setting. Data preparation required on the raw input files is done within the Python scripts for now, but this would ideally be hosted within a data warehouse using dbt, PySpark or similar tools.

Some early unit tests of individual functions and integration tests on a small data sample are set up in `./test`. They focus on the most critical custom functionality within the solution and could certainly be expanded both to be more robust and cover some of the less critical evaluation functions. These use the native Python `unittest` package, and can be triggered from the root folder of the repo by running `python -m unittest`.

To fully utilise the solution presented within this repo:

1. Set up a virtual environment and install the package requirements in `requirements.txt`
2. Run `src/train_eval.py` from the repository root folder to train and review performance of the model from the outputs in `resources`
3. Run `src/inference.py` from the repository root folder to showcase inference on a small test dataset. This script is applicable to any new inference data provided in the same format as the training data.


## Model Performance, Considerations and Improvements

#### Performance
Model discrimination is currently fairly poor, with key metrics indicating performance that is better than baseline (an uninformed guess) but not strong. Threshold choice is particularly relevant here and would need to be selected based on the context of the application to balance recall and precision.

![Confusion matrix showing reasonably strong recall but poor precision](https://github.com/oli-tailby/kaggle-depression/blob/main//resources/evaluation/confusion_matrix.png?raw=true)

![ROC Curve showing AUC of approx 0.6](https://github.com/oli-tailby/kaggle-depression/blob/main//resources/evaluation/roc_plot.png?raw=true)


Whether this can translate to usable insight would depend on the use case - we would not be able to use this model in its current iteration to confidently state whether a person has experienced a history of mental illness, but it may be able to be used in less deterministic contexts such as a factor in pricing for health and life insurance products.

#### Considerations and Biases
We also need to consider the following risks and potential biases when reviewing model performance or using outputs:
- Whilst this data is synthetic, real-world comparable data may contain biases which would be picked up by the modelling. In this scenario, I anticipate the validity of a "History of Mental Illness" outcome could be affected by factors such as age and gender impacting a person's likelihood to self-report past mental illness or seek support for a condition.
- We have no visibility on how this data has been collected, so cannot be certain it is representative of the wider population or any particular customer base. For example, there is no training data on individuals under 18 or over 80, which may not remain the case for the inference population and therefore could limit prediction accuracy.


#### Possible Improvements
I believe the most significant improvement to this model would be collection of further features. Additional demographic data including home/work location, occupation etc. could help to build further useful predictive features on environment, wealth and lifestyle. I anticipate the most predictive features for a model like this would be based around each patient's medical background, though that might not be possible in the context of this question. Further feature engineering including more complex interaction terms may also help to identify additional signals to the target.

More complex modelling methods (eg. neural networks)  and parameter tuning could extract further accuracy improvements, although given the relatively small number of features and dataset size I wouldn't expect performance improvements to be substantial. 
