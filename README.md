# **Predicting protein functions from protein sequences with an axial-attention image feature encoder**

MulAxialGO is a protein function prediction model built by the [AxialNet](https://github.com/Worldseer/axial-deeplab) backbone network using only protein features. The model structure is shown in the figure below.
![MulAxialGO](https://github.com/Worldseer/MulAxialGO/blob/main/images/mulaxialgo.jpg)


## Dependencies

* The code was developed and tested using python 3.8.3.
* We provide the dependencies to install the conda environment, first you need to install [ananconda](https://docs.anaconda.com/anaconda/install/index.html) on your computer, and then install the dependencies use:

  ```conda create --name <env> --file requirements.txt```
  or
  ```conda env create -f environment.yml```
* For the integration of model and DiamondScore to obtain MulAxialGO, the [diamond](https://github.com/bbuchfink/diamond) package needs to be installed.



## Data&Models

* data_2016：The 2016 dataset we used, includes training, validation and test sets and  the go.obo file. [download](https://drive.google.com/drive/folders/1QAt3MEqFETPCxhYlBLdC8cAj3QmP6TUu?usp=sharing)
* data_netgo：The CAFA3 dataset we used, includes training, validation and test sets and  the go.obo file. [download](https://drive.google.com/drive/folders/1QAt3MEqFETPCxhYlBLdC8cAj3QmP6TUu?usp=sharing)
* trained_models.zip: This zip file contains our trained models. [download](https://drive.google.com/drive/folders/1QAt3MEqFETPCxhYlBLdC8cAj3QmP6TUu?usp=sharing)
* predict_files: This zip file contains the predictions of the trained model on the test dataset. [download](https://drive.google.com/drive/folders/1QAt3MEqFETPCxhYlBLdC8cAj3QmP6TUu?usp=sharing)
* evaluate_results: Contains the results of the evaluation

## Scripts
- train.py：used to train MulAxialGO and output prediction files

  ```
  python evaluate_plus.py --train-data-file ./data_2016/train_data.pkl --test-data-file ./predict/prediction_2016.pkl --terms-file ./data_2016/terms.pkl --go-file ./data_2016/go.obo --diamond-scores-file ./data_2016/test_diamond.res --ont mf
  ```

- script/generator_loader.py：to generate an iterable DataLoader. 

- script/axialnet.py: contains AxialNet backbone network code, used to build MulAxialGO

- script/create_model.py: contains  code for building MulAxialGO and code for using other backbone network models

- script/utils.py: codes for Gene Ontology terms

  

## Training model
- Training the model with default parameters:
You can train the model directly with the default parameters by running `python train.py`. Line 65 in the train.py file will print the loss values to test if the model is working properly. We recommend commenting out this line if everything works
- Training models with custom parameters,
Please use:
```
python train.py --data-root ./data_2016 --epochs 100 --batch-size 16 --epochs 30 --emb-dim 16 --winding-size 40 --learning-rate 0.003
```

## Evaluate prediction.pkl
- Use the following command to evaluate the resulting prediction.pkl
```
python evaluate_plus.py --train-data-file ./data_2016/train_data.pkl --test-data-file ./predict/prediction_2016.pkl --terms-file ./data_2016/terms.pkl --go-file ./data_2016/go.obo --diamond-scores-file ./data_2016/test_diamond.res --ont mf
```
