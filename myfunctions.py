import pandas as pd
import numpy as np
import random
import torch
import os
import torch.nn as nn
import matplotlib.pyplot as plt

#############################################################################
#############################################################################
# Map unique column entries of a DataFrame to unique tokens
#############################################################################
#############################################################################
def create_mapping(df):
    vocab = {}

    # Iterate over each column
    for col in df.columns:
        # Get unique values for the current column
        unique_values = df[col].unique()
        
        # Update the mapping dictionary with unique values from the current column
        for val in unique_values:
            if val not in vocab and pd.notna(val):
                vocab[val] = len(vocab)  # Assign a unique integer index

    return vocab

#############################################################################
#############################################################################
# Create Vocabulary Dictionary, given a DataFrame
#############################################################################
#############################################################################
def vocab_dictionary(filt: pd.DataFrame):
    nan_counts = filt.isnull().sum(axis=1)

    # Find the index of the row with the maximum number of NaN values
    row_with_most_nans = nan_counts.idxmax()

    # Count the number of non-NaN values in the row with the most NaN values
    num_non_nans = filt.loc[row_with_most_nans].count()

    ncols = len(filt.columns)
    least = num_non_nans

    # Create mapping
    vocab = create_mapping(filt)

    vocab_len = len(vocab)

    return vocab

#############################################################################
#############################################################################
# Get vocabulary mapping of a specific row
#############################################################################
#############################################################################
def encode_row(s: list, vocab: dict):
    return [vocab[c] for c in s]

#############################################################################
#############################################################################
# Encode one specific row, given some vocabulary
#############################################################################
#############################################################################
def encode_instance(df_in: pd.DataFrame, df_out: pd.DataFrame, target_var: str, el_no: int, in_vocab: dict, out_vocab: dict):
        
    # Get the specified row
    row = df_in.loc[el_no].dropna().tolist()

    ncols = len(df_in.columns)

    # Get the integer values of each row entry
    encoding = encode_row(s = row, vocab=in_vocab)

    # Convert to a torch tensor
    tensor = torch.tensor(encoding, dtype=torch.long)

    # Additionally encode the target variable from the out dataframe
    y = df_out.loc[el_no, target_var]
    try:
        out_encoding = out_vocab[y]
    except:
        print(el_no, y)
    
    ytens = torch.tensor([out_encoding] * ncols, dtype=torch.long)

    return tensor, ytens

#############################################################################
#############################################################################
# Encode a random batch of rows
#############################################################################
#############################################################################
def encode_batch(df_in: pd.DataFrame, df_out: pd.DataFrame, target_var: str, n_batch: int, in_vocab: dict, out_vocab: dict):
    X_tensors = []
    y_tensors = []

    df_in_reset = df_in.reset_index(drop=True).copy()
    df_out_reset = df_out.reset_index(drop=True).copy()

    for _ in range(n_batch):
        el_no = random.randint(0, len(df_in) - 1)
        
        # Make sure this election has a result
        # while pd.isnull(df_out.loc[el_no, 'vote_g2022']):
        #     el_no = random.randint(0, len(df_in) - 1)
            
        X_instance, y_instance = encode_instance(df_in=df_in_reset, df_out=df_out_reset, target_var=target_var,
                                                 el_no=el_no, in_vocab=in_vocab, out_vocab=out_vocab)
        X_tensors.append(X_instance)
        y_tensors.append(y_instance)

    # Stack the tensors along the first dimension (batch dimension)
    X_stacked = torch.stack(X_tensors)
    y_stacked = torch.stack(y_tensors)

    return X_stacked, y_stacked

#############################################################################
#############################################################################
# Encode a specific batch of rows (els)
#############################################################################
#############################################################################
def encode_specific(df_in: pd.DataFrame, df_out: pd.DataFrame, target_var: str, els: list, in_vocab: dict, out_vocab: dict):
    X_tensors = []
    y_tensors = []

    for el in els:

        el_no = el
        
        X_instance, y_instance = encode_instance(df_in=df_in, df_out=df_out, target_var=target_var,
                                                el_no=el_no, in_vocab=in_vocab, out_vocab=out_vocab)
        X_tensors.append(X_instance)
        y_tensors.append(y_instance)

    # Stack the tensors along the first dimension (batch dimension)
    X_stacked = torch.stack(X_tensors)
    y_stacked = torch.stack(y_tensors)
    return X_stacked, y_stacked

#############################################################################
#############################################################################
# Train the model
#############################################################################
#############################################################################
def train(model, epochs: int, data_in: pd.DataFrame, data_out: pd.DataFrame, target_var: str, in_vocab: dict, out_vocab: dict,
        test_in: pd.DataFrame, test_out: pd.DataFrame, lr: float = 1e-4, eval_iters: int = 1000, n_batch:int=None, rows:list=None):

    # Create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay=0.0001)

    iters = []
    losses = []
    test_losses = []

    for iter in range(epochs):
        # Get a batch of data
        # xb, yb = encode_batch(pre_2022, out_pre_2022, 4)

        # Encode specific rows if 'rows' is specified
        if rows is not None:
            xb, yb = encode_specific(data_in, data_out, target_var=target_var, els=rows, in_vocab=in_vocab, out_vocab=out_vocab)
        else:
            xb, yb = encode_batch(data_in, data_out, target_var=target_var, n_batch=n_batch, in_vocab=in_vocab, out_vocab=out_vocab)
        # Evaluate the loss
        logits, loss = model.forward_single(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter % eval_iters == 0:

            iters.append(iter)
            losses.append(loss.item())

            # Evaluate test loss as well
            tx, tb = encode_batch(test_in, test_out, target_var=target_var, n_batch=n_batch, in_vocab=in_vocab, out_vocab=out_vocab)
            test_logits, test_loss = model.forward_single(tx, tb)
            test_losses.append(test_loss.item())

            print(f"Losses at iteration {iter} = [{loss.item():.3f}, {test_loss.item():.3f}]")

        
    print(f"Final Loss = {loss.item()}")

    plt.clf()
    plt.plot(iters, losses, label = 'Training Loss')
    plt.plot(iters, test_losses, label = 'Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Training and Test Losses')
    plt.grid(True)
    plt.legend()
    plt.savefig('training_loss_plot.png')
    return

#############################################################################
#############################################################################
# Make predictions on reserved data and evaluate
#############################################################################
#############################################################################

#############################################################################
# To do this, make an individual guess
#############################################################################
def make_guess(model, input):
    
    val, dist = model.generate(input, type="single")

    return val, dist

def predict(model, data_in: pd.DataFrame, data_out: pd.DataFrame, target_var: str, els: list, in_vocab: dict, out_vocab: dict):
    guesses = []
    for i in range(100):
        x, y = encode_specific(data_in, data_out, els=els, target_var=target_var, in_vocab=in_vocab, out_vocab=out_vocab)
        guess, dist = make_guess(model, x)
        guesses.append(guess)

    combined_tensor = torch.stack(guesses, dim=0)
    combined_tensor_float = combined_tensor.float()
    mean_tensor = torch.mean(combined_tensor_float, dim=0)
    mean_vector = mean_tensor.squeeze().tolist()

    rounded_guesses = np.round(mean_vector).astype(int)
    real_outcomes = y[:, :1].numpy().ravel()

    num_same = sum(1 for x, y in zip(rounded_guesses, real_outcomes) if x == y)

    # Calculate the percentage
    percentage_same = (num_same / len(rounded_guesses)) * 100

    print(f"Percentage of 2022 MO turnout predicted correctly = {percentage_same:.2f}%")

    print(f'This is equivalent to {num_same} out of {len(rounded_guesses)} correctly predicted.')

    return