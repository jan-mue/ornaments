# Symmetry Recognition in Wallpaper Patterns Using Deep Learning

This repository contains the code written for my master's thesis "Symmetry Recognition in Wallpaper Patterns Using Deep Learning"
at the Technical University of Munich.
The two main models for the classification of wallpaper patterns by lattice type and by wallpaper group are defined in
the modules `ornaments.modeling.models` and `main.py`. The latter is trained on artificial patterns generated with
the script `pattern_generator.py`. The code for fine-tuning this model can be found in `fine_tune_group_classifier.py`.

The main dataset used to train the model was downloaded from the website
[Ornament World Exhibition](http://www.science-to-touch.com/en/OWEViewer.html) with the script `ornaments.data_loading.download_data.py`.
Additional preprocessing code can be found in the same package.

The weights for the pre-trained models can be found in the `models/` directory. To reproduce the results in the thesis the
OWE dataset must be downloaded and preprocessed and then the scripts `inference.py` and `lattice_inference.py` can be executed.

The code for the extraction of a lattice basis described in Section 4.4 of the thesis can be found in the script `run_lattice_extraction.py`.

The code in this repository uses Tensorflow 2 as deep learning framework. All dependencies can be installed using Anaconda:
```
conda env create -f environment.yml
```
Many of the plots in the thesis were generated with Sage and its LaTeX integration SageTeX. Installation instructions can
be found [here](https://doc.sagemath.org/html/en/tutorial/sagetex.html). The code in the file `plots.sage` must be imported
from the LaTeX files to compile the thesis.
