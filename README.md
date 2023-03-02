# LADS
Official Implementation of [LADS (Latent Augmentation using Domain descriptionS)](https://lisadunlap.github.io/LADS-website)

![LADS method overview.](figs/lads-method-2-1.png "LADS method overview")

*WARNING: this is still WIP, please raise an issue if you run into any bugs.*

## Getting started

1. Install the dependencies for our code using Conda. You may need to adjust the environment YAML file depending on your setup.

    ```
    conda env create -f environment.yaml
    ```

    - There are some library conflicts when installing tensorflow by `pip`, so LUO recommends to install `pip` dependencies one by one in the terminal.

        ```
        pip install tensorflow
        pip install tensorflow_hub
        ...
        ```


2. Launch your environment with `conda activate LADS` or `source activate LADS`

3. Compute and store CLIP embeddings for each dataset (see below)

4. Run one of the config files and be amazed (or midly impressed) by what LADS can do

## Code Structure 
The configurations for each method are in the `configs` folder. To try say the baseline of doing normal LR on the CLIP embeddings:
```
python main.py --config configs/Waterbirds/base.yaml
```

Datasets supported are in the [helpers folder](./helpers/data_helpers.py). Currently they are:
* Waterbirds (100% and 95%)
* ColoredMNIST (LNTL version and simplified version)
* DomainNet
* CUB Paintings
* OfficeHome

You can download the CLIP embeddings of these datasets [here](https://drive.google.com/drive/folders/1ItjhX7RPfQ6fQQk6_bEYJPewnkVdcfOC?usp=sharing)

Since computing the CLIP embeddings for each train/val/test set is time consuming, you can store the embeddings by setting `DATA.LOAD_CACHED=False` and `DATA.SAVE_PATH=[path you want to save to]`

Then, add the path to the saved embeddings to DATASET_PATHS in [data_helpers](./helpers/data_helpers.py) and set `DATA.LOAD_CACHED=Tue` in your yaml file

More description of each method and the config files in the config folder. 

## Running LADS
In LADS we train an augmentation network, augment the training data, then train a linear probe with the original and augmented data. Thus we use the same ADVICE_METHOD class and change the `EXP.AUGMENTATION` parameter to `LADS`. 

To make sure everything is working, run:
`python main.py --config configs/CUB/lads.yaml`
and check your results with https://wandb.ai/clipinvariance/LADS_CUBPainting_Replication/runs/ok37oz5h. 

For the bias datasets, the augmentation class is called `BiasLADS`, and you can run the `lads.yaml` configs as well :)

## Running CLIP Zero-Shot
In order to run the CLIP zero-shot baseline, set `EXP.ADVICE_METHOD=CLIPZS` and run the `clip_zs.py` file instead of `main.py` file. 

For example
```
python clip_zs.py --config configs/Waterbirds/ZS.yaml
```

CLIP text templates are located in `helpers/text_templates.py`, and you can specify which template you want with the `EXP.TEMPLATES` parameter. 

Also note that we use the classes given in `EXP.PROMPTS` instead of the dataset classes in the dataset object itself so make sure to set those correctly.

## Running LR

If you want to simply run logistic regression on the embeddings, run the `mlp.yaml` file in any of the config folders. Some of the methods we have dont require any training (e.g. HardDebias), so all those do is perform a transformation on the embeddings before we do the logistic regression. 

Note: you do need to save the embeddings for each model in the `helpers/dataset_helpers.py` folder.

For example, to run LR on CLIP with a resnet50 backbone on ColoredMNIST, run
```
python main.py --config configs/ColoredMNIST/mlp.yaml
```

**LR Initialized with the CLIP ZS Language Weights** For a small bump in OOD performance, you can run the `mlpzs.yaml` config to initalize the linear layer with the text embeddings of the classes. The prompts used are dictated by `EXP.TEMPLATES`, similar to running zero-shot.

## Some important parameters
<details><summary>EXP.TEXT_PROMPTS</summary>

This is the domains/biases that you want to be invariant to. You can either have them be class specific (e.g. `["a painting of a {}.", "clipart of a {}."]`) or generic (e.g. `[["painting"], ["clipart"]]`). The default is class specific so if you want to use generic prompts instead set `AUGMENTATION.GENERIC=True`. For generic prompts, if you want to average the text embeddings of several phrases of a domain, simply add them to the list (e.g. `[["painting", "a photo of a painting", "an image of a painting"], ["clipart", "clipart of an object"]]`).
</details>


<details><summary>EXP.NEUTRAL_PROMPTS</summary>

If you want to take the difference in text embeddings (for things like the directional loss, most of the augmentations, and the embedding debiasing methods). you can set a neutral prompt (e.g. `["a sketch of a {}."]` or `[["a photo of a sketch]]`). Like TEXT_PROMPTS you can have it be class specific or generic, but if TEXT_PROMPTS is class specific so is NEUTRAL_PROMPTS and vice versa.
</details>


<details><summary>EXP.ADVICE_METHOD</summary>

This sets the type of linear probing you are doing. Set to `LR` if you want to use the scikit learn LR (what is in the CLIP repo) or `ClipMLP` for pytorch MLP (if `METHOD.MODEL.NUM_LAYERS=1` this is LR). Typically `CLIPMLP` runs a lot faster than `LR`.

You can also set the advice method to one of the debiasing methods (different from augmentations in that we augment the training data and dont add in the original training data), but we don't use them anymore and I'm too lazy to explain it so if you care to try them out check the configs file (WARNING these are old so high chance of bugs).
</details>


## Checkpoints
The main results and checkpoints of LADS and other baselines can be accessed on wandb.  
* Waterbirds: https://wandb.ai/clipinvariance/LADS_Waterbirds_Replication
* ColoredMNIST: https://wandb.ai/clipinvariance/LADS_ColoredMNIST_Replication
* CUB: https://wandb.ai/clipinvariance/LADS_CUBPainting_Replication 
* miniDomainNet: https://wandb.ai/clipinvariance/LADS_miniDomainNet_Replication 
