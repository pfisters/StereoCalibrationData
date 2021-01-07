# Stereo Calibration Data
<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about">About</a></li>
    <li><a href="#usage">Usage</a></li>
  </ol>
</details>

<!-- FILE AND FOLDER STRUCTURE -->
## About
This repository is the result of my semester thesis @ ETH Zurich with Supercomputing Systems AG.

It contains classes and functions to analyse the data produced by the repository `pfisters/StereoCameraParameterTracker`, as well as the possibility to create synthetic data sets for the latter in the form of cubes and planes in 3-dimensional space.

For questions and inquriries, you reach me at pfisters@ethz.ch.

<!-- USAGE EXAMPLES -->
## Usage
1. Setup an environment using conda and the specifications in `env/`
```
conda env create -f ./env/environment_win.yml
```
or 
```
conda env create -f ./env/environment_mac.yml
```
respectively

2. OPTIONAL: Create a synthetic data set
``` python
python ./create_synth_data_set.py
```
You can specify the flags either via command line or change them in the beginning of the file. Make sure to specify the factory settings file. The default file can be easily replaced by any factory settings file from a use case in `pfisters/StereoCameraParameterTracker`. The resulting data set is stored in the folder `.\synth_data_sets`.

3. Generate logs with `pfisters/StereoCameraParameterTracker`. They will be generated in the use case directories in the folder `Logs`
4. Analyse the logs 
``` python
python ./analyse_results.py
```

You can specify the log folder to analyse either in the command line or by changing the flags in the beginning of the file.
Make sure to specify, whether the log you analyse is a synthetic data set or a field data set. The class analyses a synthetic data set by default.
