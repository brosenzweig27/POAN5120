import pandas as pd
import numpy as np

#############################################################################
### Import and inspect the factorized data
#############################################################################

df = pd.read_csv("factorized_data.csv")
print(df.head())

#############################################################################
### Randomly select 100,000 rows indices
#############################################################################

data = df.sample(n=100000, random_state=2342)
data = data.reset_index(drop=True)

print(data.head())

#############################################################################
### Separate targets and split into test and train (50/50 here)
#############################################################################

out = data[["vote_g2022", "vote_p2022"]]
filt = data.drop(columns = ["vote_g2022", "vote_p2022", 'vote_g2023', 'vote_g2024'])

print(filt.head())
print(out.head())

### Subset 50% of data for testing

np.random.seed(481516)
random_index = np.random.permutation(len(data))

# Calculate the test size (e.g., 50%)
test_size = int(len(data) * 0.5)

# Create test sets
filt_test = filt.iloc[random_index[:test_size]]
out_test = out.iloc[random_index[:test_size]]

# Create train sets by excluding rows in test sets
filt_train = filt.iloc[random_index[test_size:]]
out_train = out.iloc[random_index[test_size:]]

print("filt_train head:")
print(filt_train.head())
print("\nout_train head:")
print(out_train.head())
print("\nfilt_test head:")
print(filt_test.head())
print("\nout_test head:")
print(out_test.head())

filt_test = filt_test.reset_index(drop=True)
out_test = out_test.reset_index(drop=True)
filt_train = filt_test.reset_index(drop=True)
out_train = out_train.reset_index(drop=True)

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
out_vocab_size = len(out_vocab)
print(out_vocab_size)

#############################################################################
### Import encoding functions to get tensors to be used for training
#############################################################################

from code_files.myfunctions import encode_instance, encode_batch, encode_specific

# Get a specific encoding of the elections from 2022, as an example
x, y = encode_specific(df_in=filt_train, df_out=out_train, target_var='vote_g2022', els=[1,2], in_vocab=in_vocab, out_vocab=out_vocab)
print(x, y)

#############################################################################
### Set globals to be used by the model in the next step
#############################################################################

ncol = len(filt.columns)

from code_files.config import globals
globals(vocab_size=vocab_size, out_space=2, ncol=ncol, n_embd=96, n_head=4, n_layer=8, dropout=0.2, width=12)

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
filt_train_subset = filt_train.sample(n=5000)
out_train_subset = out_train.loc[filt_train_subset.index]

filt_train_subset = filt_train_subset.reset_index(drop=True)
out_train_subset = out_train_subset.reset_index(drop=True)

# print(filt_train_subset.head())
# print(out_train_subset.head())

# We should additionally get a small subset of remaining test rows
filt_test_subset = filt_test.sample(n=1000)
out_test_subset = out_test.loc[filt_test_subset.index]

filt_test_subset = filt_test_subset.reset_index(drop=True).copy()
out_test_subset = out_test_subset.reset_index(drop=True).copy()
out_test_index = [num for num in range(len(out_test_subset))]

print(filt_train_subset.head())

# Run the training algorithm on n_batch random rows per epoch
train(model, epochs=100, data_in=filt_train, data_out=out_train, target_var='vote_g2022', in_vocab=in_vocab, out_vocab=out_vocab, 
      test_in=filt_test_subset, test_out=out_test_subset, n_batch=1000, lr = 1e-5, eval_iters = 10)

# Run the training algorithm on 'rows'
rows_list = list(range(len(filt_train_subset)))
train(model, epochs=100, data_in=filt_train_subset, data_out=out_train_subset, target_var='vote_g2022', in_vocab=in_vocab, out_vocab=out_vocab, 
      test_in=filt_test_subset, test_out=out_test_subset, rows=rows_list, n_batch=100, lr = 5e-4, eval_iters = 10)

#############################################################################
### Evaluate the model on reserved data
#############################################################################

from code_files.myfunctions import predict

predict(model, data_in=filt_test_subset, data_out=out_test_subset, target_var='vote_g2022', els=out_test_subset.index, in_vocab=in_vocab, out_vocab=out_vocab)
