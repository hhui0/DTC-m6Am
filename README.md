## Requirements
To use this project, you need to install the following libraries:

- torch                     2.0.0+cu118
- pandas                    2.1.1                   
- pytorch-lightning         2.0.8                    
- pytorch-metric-learning   1.7.3 
- pandas                    2.1.1
- multimethod               1.9.1

You can install them by running the following command:

```bash
pip install -r requirements.txt
```

## Usage

To use the STM-ac4C model for prediction, you need to follow these steps:

1. Import the classifier and the utils from the project folder.

```python
from models.DTC_m6Am import classifier
from utils.one_trial import LitModel
from utils.prepare_data import get_m6am
```

2. Get the m6Am data as trainset and testset.

```python
trainset, testset = get_m6am()
```

3. Get the hyperparameters and the model parameters of the classifier.

```python
hparams = classifier.get_hparams()
model_params = classifier.get_model_params()
```

4. Create a LitModel instance with the classifier, the hyperparameters, and the model parameters.

```python
model = LitModel(classifier, hparams , model_params)
```

5. Use the `test` function to perform prediction and evaluation on the testset. You need to specify the checkpoint path of the trained model.

```python
model.test(testset, ckpt_path="ckpt/mcc=0.411.ckpt")
```

6. Use the `predict_proba` function to get the probability of each sample containing an ac4C site. You need to specify the checkpoint path of the trained model.

```python
proba = model.predict_proba(testset, ckpt_path="ckpt/mcc=0.411.ckpt")
```

7. Use the `fit` function to train the model on the trainset.

```python
model.fit(trainset)
```
