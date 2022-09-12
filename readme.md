# Detecting Irregular Network Activity with Adversarial Learning and Expert Feedback

# Required packages

tqdm==4.57.0<br>
torch==1.3.1<br>
numpy==1.19.2<br>
scipy==1.5.2<br>
imageio==2.9.0<br>
scikit_image==0.17.2<br>
pandas==1.1.3<br>
matplotlib==3.3.1<br>
scikit_learn==1.0.2<br>
skimage==0.0<br>

# Instructions

1. Extract dataset of choice and place in `~/data`.
2. To evaluate trained models, follow the evaluation step.

# Datasets

Please download the datasets from https://drive.google.com/drive/folders/1YXxv-OwIms0olIOxDI0530U8_Tu1FHT5?usp=sharing and unzip them inside `~/data`.

# Trained Models

The following trained models are availables in `~/trainedmodels`. 
1. CAAD
2. CAAD-UQ
3. CAAD-EF

# Evaluation

```
python test.py {dataset} {model}
```

where
- dataset can take values 'ltw1', 'ltw2', 'stw1', 'mnist' 
- model can take values 
    * 'caad' for all datasets
    * 'caad-uq' for datasets 'ltw1' and 'stw1'
    * 'caad-ef' for datasets 'ltw1' and 'stw1'

Example
```
python test.py stw1 caad
```
By default evaluation script is set to run on the trained models. If that needs to be change, place affix your model path at line 66 in test.py.


# Training

```
python train_caad.py {dataset}
```
where dataset can take values 'ltw1', 'ltw2', 'stw1', 'mnist' 
```
python train_caad-uq.py {dataset}
```
where dataset can take values 'ltw1', 'stw1'
```
python train_caad-ef.py {dataset} {modelpath}
```
where dataset can take values 'ltw1', 'stw1' and modelpath is where you find the output of the caad-uq model.

Example
```
python train_caad-ef.py ltw1 /home/rgopikrishna/code/logs/caad-uq/
```
