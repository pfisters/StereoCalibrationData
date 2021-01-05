# Stereo Calibration Data

This repository is the result of my semester thesis @ ETH Zurich with Supercomputing Systems AG.

It contains classes and functions to analyse the data produced by the repository `pfisters/StereoCameraParameterTracker`. 

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#file-and-folder-structure">File and Folder Structure</a></li>
    <li><a href="#usage">Usage</a></li>
  </ol>
</details>

<!-- FILE AND FOLDER STRUCTURE -->
## File and Folder Structure



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
