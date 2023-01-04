import wandb
import torch
import librosa

def pad_sequence(batch):
    """Makes all tensor in a batch the same length by padding with zeros."""
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(batch):
    """Collate function for variable-length batches."""
    # Iterate
    tensors, targets = [], []
    for waveform, label in batch:
        tensors += [waveform]
        targets += [torch.tensor(label)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets

def frame_batch(batch,frame_size=5,step=5,sampling_rate=44100):
   
    x=torch.squeeze(batch)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    frames_len=int(frame_size*sampling_rate/1000)
    step_len=int(step*sampling_rate/1000)
    x_=librosa.util.frame(x.cpu().numpy(),frame_length=frames_len,hop_length=step_len,axis=0)
    x_=torch.unsqueeze(torch.from_numpy(x_),1)

    return x_.to(device)

def train_model(model, train_loader, criterion, optimizer, device, nb_epochs):
    """Trains the given model with the provided parameters."""
    loss_list = []
    for e in range(nb_epochs):

        # Accumulated loss
        acc_loss = 0

        wandb.watch(model)
        # Iterate by mini-batches
        for inputs, targets in train_loader:

            # Forward pass
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model.forward(inputs)

            # Compute loss
            loss = criterion(outputs.squeeze(), targets) # Sure about the squeeze ? W/o there is a bug
            acc_loss += loss.item()

            # Backward pass
            optimizer.zero_grad() # Zero grad
            loss.backward()       # Backpropagation
            optimizer.step()      # SGD optimizer

        # Log
        wandb.log({"loss": acc_loss, "epoch": e})


def test_model(model, test_loader, device):
    """Test trained model and prints out test error and accuracy."""

    nb_test_errors, nb_test_samples = 0, 0
    model.eval()

    for input, targets in iter(test_loader):
        
        input, targets = input.to(device), targets.to(device)
        output = model(input)
        
        wta = torch.argmax(output.data, 1).view(-1)

        for i in range(targets.size(0)):
            nb_test_samples += 1
            if wta[i] != targets[i]: 
                nb_test_errors += 1

    test_error = 100 * nb_test_errors / nb_test_samples
    print(f'Test error: {test_error:.02f}% ({nb_test_errors}/{nb_test_samples})')   
    print(f'Accuracy: {round((nb_test_samples-nb_test_errors)*100/nb_test_samples, 2)}% ({nb_test_samples-nb_test_errors}/{nb_test_samples})')
