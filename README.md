# mind-reading-and-control
Deep Learning for Brain‐Computer Interface

Remember to follow the .env.template file to create your own .env file to enable your own environment variables.


# # - - - BCI - Data - - - # # 

https://drive.google.com/drive/folders/106N4BVHHLwRk6HtTb2EMYrbWvSKIkKR1?usp=sharing

The data is structured as follows: 
- Each subject has 23 recorded sessions; All recorded on the same day.
- 10 training sessions for opening 
- 10 training sessions for closing
- 1* dwell session (subject did not move for 1 minute)
- 2 online testing sets were recorded for each subject with hand movements alternating.

* We used 1 closing and 1 opening training set to tune our dwell heuristic.

Data structure: 
channels 0-9 is the EEG data.
channel 12 is the EMG data.
The dataset is not labeled and movements should be inferred from the EMG data.


-- Read more : https://projekter.aau.dk/projekter/files/474331713/cs_22_mi_10_11_thesis.pdf
