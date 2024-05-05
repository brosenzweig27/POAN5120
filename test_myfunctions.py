import pandas as pd
import numpy as np

#############################################################################
### Import and inspect the desired datasets
#############################################################################

filt = pd.read_csv("filtered_data.csv")
# filt = pd.read_csv("imputed_filt.csv")
print(filt.head())
print(len(filt))

out = pd.read_csv("targets.csv")
# out = pd.read_csv("imputed_targets.csv")
print(out.head())
print(len(out))

# Get the number of columns in the input dataframe
ncol = len(filt.columns)

#############################################################################
### Randomly select 100,000 rows indices
#############################################################################

#############################################################################
### Use the vocab_dictionary function to create the dictionary mapping
#############################################################################

from code_files.myfunctions import vocab_dictionary

#############################################################################
### Retrieve the vocabulary for the input and output data
#############################################################################

in_vocab = vocab_dictionary(filt)
out_vocab = vocab_dictionary(out)

vocab_size = len(in_vocab)
print(vocab_size)

#############################################################################
### Create data subsets for 2022 and other year subsets we may want
#############################################################################

### Subset 10% of data for testing

np.random.seed(481516)
test_size = int(len(filt) * 0.1)

# create test sets
filt_test = filt.sample(n=test_size)
out_test = out.loc[filt_test.index]

# Create filt_train and out_train by excluding rows in filt_test and out_test
filt_train = filt.drop(index=filt_test.index)
out_train = out.drop(index=out_test.index)

#############################################################################
### Subset out 10,000 rows at a time to improve computation time
#############################################################################
subset_size = 10000

filt_train_subset = filt_train.sample(n=subset_size)
out_train_subset = out_train.loc[filt_train_subset.index]

#############################################################################
### Import encoding functions to get tensors to be used for training
#############################################################################

from code_files.myfunctions import encode_instance, encode_batch, encode_specific

# Get a specific encoding of the elections from 2022, as an example
x, y = encode_batch(df_in=filt_train, df_out=out_train, n_batch=5, in_vocab=in_vocab, out_vocab=out_vocab)
print(x, y)

#############################################################################
### Set globals to be used by the model in the next step
#############################################################################
from code_files.config import globals
globals(vocab_size=vocab_size, out_space=2, ncol=ncol, n_embd=24, n_head=4, n_layer=2, dropout=0.2)

#############################################################################
### Import an instance of the model
#############################################################################
from code_files.model import ElectionModel
model = ElectionModel()

#############################################################################
### Train the model
#############################################################################

from code_files.myfunctions import train

# to test on different data, rerun this code
filt_train_subset = filt_train.sample(n=10000)
out_train_subset = out_train.loc[filt_train_subset.index]

# We should additionally get a small subset of remaining test rows
filt_test_subset = filt_test.sample(n=1000)
out_test_subset = out_test.sample(n=1000)
filt_test_subset = filt_test_subset.reset_index(drop=True).copy()
out_test_subset = out_test_subset.reset_index(drop=True).copy()
out_test_index = [num for num in range(len(out_test_subset))]

print(filt_train_subset.head())

# Run the training algorithm
train(model, epochs=1000, data_in=filt_train_subset, data_out=out_train_subset, target_var='vote_g2022', in_vocab=in_vocab, out_vocab=out_vocab, 
      test_in=filt_test_subset, test_out=out_test_subset, n_batch=100, lr = 1e-5, eval_iters = 50)

#############################################################################
### Evaluate the model on reserved data
## In this case, assume we've subsetted out 2022 house elections
#############################################################################

from code_files.myfunctions import predict

predict(model, data_in=filt_test_subset, data_out=out_test_subset, els=out_test_subset.index, in_vocab=in_vocab, out_vocab=out_vocab)

filt_train_subset.to_csv('training_data.csv', index=False)
out_train_subset.to_csv('training_out.csv', index=False)
filt_test_subset.to_csv('testing_data.csv', index=False)
out_test_subset.to_csv('testing_out.csv', index=False)