# wfield_notebooks
Sample Notebooks for wfield mapping, phase and sign maps.

## Getting set up ##

### 1. Installations ###
Install [git](https://git-scm.com/downloads/) and [Anaconda](https://docs.conda.io/en/latest/miniconda.html).

### 2. Clone repository ###

Open Git Bash in the repository you want to store the code in and **right-click -> 'Git Bash here'**

Clone this repository on your computer(click clone and copy/paste string in Git Bash)

### 3. Environment creation ###

Open Git Bash in your stimpy folder.

In Git Bash enter(*<> symbols signify that you should use your own names,values etc.*):

> `conda create -n <your_env_name> python=3.8`

> `conda activate <your_env_name>`

> `conda install numpy scipy matplotlib pandas jupyterlab tifffile tqdm scikit-learn -c conda-forge -y`

Wait for installations to finish without an error, after that you can type `jupyter notebook`(or `jupyter lab`) in Git Bash and start using the notebooks.
