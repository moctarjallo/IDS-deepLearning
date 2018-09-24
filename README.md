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

But first make a virtual environment:

    conda create -n kddcup
    
Activate the virtual env:
  
    source activate kddcup
    
Now install the dependencies:

    pip install -r requirements.txt

(The requirements.txt file lives at the root of the downloaded repository)

Download the training and testing data for this project from this link:

  http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html

click on these links download the training and testing data:

    kddcup.data_10_percent.gz A 10% subset. (2.1M; 75M Uncompressed) (training data)
    corrected.gz (testing data)
    kddcup.names (names file: data column names)
  
Move these files to a folder located in home:
  
    cd ~
    mkdir .kddcup
    mv <downloaded files> .kddcup/
   
  
  
USAGE:
  
Create an alias for more convenience:
    
    alias kddcup="python kddcup/main.py"

Now run:
  
    kddcup train    # to train a model
    kddcup test     # to test a model  (not yet implemented)
    kddcup predict  # to classify a packet (not yet implemented)

You can also play around by importing objects and calling their methods

You can modify the config.json file to change some parameters of training and testing

    config.json
    
    
