# UTILS

As you may have guessed, you will find all helper function here. The files in this directory includes

## Files in This Directory

- `helper.py`: An assortment of functions.
- `infrasounddataset.py`: The pytorch data set for the infrasound data
- `test_data_prep.py`: Script to prepare test data for each model configuration  

### Plotting scripts: The following are use in creating plots used in the paper
- `relative_plotter.py`: A script for training a CNN using the hyperparameters found.
- `helicorder_plotter.py`:
- `histogram_plotter.py`:
---

### helper.py
The helper file contain a number of functions 
- `get_logger.py`: Handles all logging for the entire project. (A file handler here is you require one)
- `create_slides_np`: The infransound data comes a 24hr format, this function create a ten seconds window with a one-second slide
- `signal_labeller`: Labels the 10-sec windows with respect to the consistenecy measure  and cross correlation measure 
- `create_dataframe`: Creates a datframe from the windowed signals and the labels for each day
- `create_full_dataset`: Creates full dataset for the entire data
- `partition_data`: create train test data 
- `partition_high_signal`: reduce class to two 
- `make_df`:
- `create_temporal_coverage_data`


### infrasounddataset.py

A special dataset object for the infrasound data set, this dataset handle all required preporprocessing of input waveform, as such the following function are implemented in the class
- `_cut_if_necessary`: Trims waveform to a 10-s windows
- `_right_pad_if_necessary`: zero pad if data is less than 10-sec
- `_resample_if_not_target_sample_rate`: Data was sampled at 100Mhz, resample data to default 100Mhz is not 100Mhz
- `_mix_down_if_not_mono`: waveform comes in single channel, this helps to mix down is more than single channel
- `_get_audio_sample_label`


### test_data_prep.py: 
Creates the test set for all model configurations