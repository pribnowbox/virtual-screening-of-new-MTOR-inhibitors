# Virtual screening of new MTOR inhibitors
The graph convolutional deep learning model predicts the probability of finding an individual compound within the 'active' class for mTOR inhibition. Due to the stochastic nature of the learning algorithm, three independent deep-learning models were trained by the same dataset. Therefore, potential compounds are those that are in consensus among the three deep learning models.

The models were trained by the DeepChem package (Deepchem version 2.4.0 and Tensorflow version 2.4.1). Run 'predict.py' to make the prediction. The three models are provided as 'saved_model_01', 'saved_model_02', and 'saved_model_03'. 'mushroom_dataset.csv' contains compounds with the SMILES format, which is the input of the models.
