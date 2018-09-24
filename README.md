# IDS-deepLearning

Deep Learning for Introduction Detection Systems (IDS)

This repository is about using Deep Learning techniques to help build an introduction detection system.

INSTALLATION:

This repository has been tested with python3.6.5 via anaconda
Install python 3.6 anaconda/miniconda following this link:

  https://conda.io/docs/user-guide/install/index.html

Clone this repository (assuming you have git installed):

    git clone https://github.com/mctrjalloh/IDS-deepLearning

Move into the downloaded repository for the rest of installations

Install dependencies:

After anaconda installation the `conda` command line should be available in the terminal
Use it to install `pip`

    conda install pip

Then use pip to install package dependencies:

<<<<<<< HEAD
But first make a virtual environment:

    conda create kddcup

Activate the virtual env:

    source activate kddcup

Now install the dependencies:

||||||| merged common ancestors
=======
But first make a virtual environment:

    conda create -n kddcup
    
Activate the virtual env:
  
    source activate kddcup
    
Now install the dependencies:

>>>>>>> a711eff4b8c1dfca64c3fdea6f4dd91fe89cface
    pip install -r requirements.txt

(The requirements.txt file lives at the root of the downloaded repository)

Download the training and testing data for this project from this link:

  http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html

click on these links download the training and testing data:
<<<<<<< HEAD

    kddcup.data_10_percent.gz A 10% subset. (2.1M; 75M Uncompressed) (training data)
    corrected.gz (testing data)
    kddcup.names (names file: data column names)

||||||| merged common ancestors
kddcup.data_10_percent.gz A 10% subset. (2.1M; 75M Uncompressed) (training data)
corrected.gz (testing data)
kddcup.names (names file: data column names)
  
=======

    kddcup.data_10_percent.gz A 10% subset. (2.1M; 75M Uncompressed) (training data)
    corrected.gz (testing data)
    kddcup.names (names file: data column names)
  
>>>>>>> a711eff4b8c1dfca64c3fdea6f4dd91fe89cface
Move these files to a folder located in home:
<<<<<<< HEAD

    cd ~
    mkdir .kddcup
    mv <downloaded files> .kddcup/

||||||| merged common ancestors
  
 cd ~
mkdir .kddcup
mv <downloaded files> .kddcup/
  
  
  
=======
  
    cd ~
    mkdir .kddcup
    mv <downloaded files> .kddcup/
   
  
  
>>>>>>> a711eff4b8c1dfca64c3fdea6f4dd91fe89cface
USAGE:
<<<<<<< HEAD

A convenient command line interface is coming soon..
||||||| merged common ancestors
  
 A convenient command line interface is coming soon..
=======
  
Create an alias for more convenience:
    
    alias kddcup="python kddcup/main.py"
>>>>>>> a711eff4b8c1dfca64c3fdea6f4dd91fe89cface

<<<<<<< HEAD
But for a quick run now, run:

    python kddcup/main.py
||||||| merged common ancestors
But for a quick run now, run:
  
 python kddcup/main.py
=======
Now run:
  
    kddcup train    # to train a model
    kddcup test     # to test a model  (not yet implemented)
    kddcup predict  # to classify a packet (not yet implemented)

You can also play around by importing objects and calling their methods

You can modify the config.json file to change some parameters of training and testing
>>>>>>> a711eff4b8c1dfca64c3fdea6f4dd91fe89cface

<<<<<<< HEAD
For now you can play around by importing objects and calling their methods

You can modify the config.json file to achieve intended results

    config.json
||||||| merged common ancestors
For now you can play around by importing objects and calling their methods
=======
    config.json
    
    
>>>>>>> a711eff4b8c1dfca64c3fdea6f4dd91fe89cface
