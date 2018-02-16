import os

os.system('python prediction_with_weights.py weights-best-submission.hdf5')
os.system('python mask_to_submission.py')
