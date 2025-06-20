#base.py
from base_1 import *
from sklearn.metrics import auc

# Convert data to DataLoader for batch processing
def loader(train_x,train_y,test_x,test_y,batch_size):
    train_data = Data.TensorDataset(train_x,train_y)
    test_data = Data.TensorDataset(test_x,test_y)
    train_loader = Data.DataLoader(dataset = train_data,batch_size = batch_size,shuffle = True,num_workers = 0,worker_init_fn = _init_fn)
    test_loader = Data.DataLoader(dataset = test_data,batch_size = batch_size,shuffle = True,num_workers = 0)
    return train_loader,test_loader

def _init_fn(worker_id,seed):
    np.random.seed(int(seed))

def draw_pr(confidence_scores, data_labels, label):
    precision, recall, thresholds = precision_recall_curve(data_labels, confidence_scores)
    AP = average_precision_score(data_labels, confidence_scores)  # Compute AP
    plt.plot(recall, precision, label='%s(%0.3f)' % (label, AP))


# The list of methods to compare, temporarily excluding some less relevant ones

# Define the architecture of the neural network
class MLP(nn.Module):
    def __init__(self,input_size, hidden1_size,hidden2_size, output_size):
        super(MLP, self).__init__()
        # First hidden layer
        self.hidden1 = nn.Linear(input_size, hidden1_size)
        self.bn1 = nn.BatchNorm1d(hidden1_size, momentum=0.9)
        self.dp1 = nn.Dropout(0.1)
        self.hidden2 = nn.Linear(hidden1_size, hidden2_size)
        self.bn2 = nn.BatchNorm1d(hidden2_size, momentum=0.9)
        self.dp2 = nn.Dropout(0.1)
        # Prediction layer
        self.predict = nn.Linear(hidden2_size, output_size)

    def forward(self,x):
        x = torch.relu(self.hidden1(x))
        x = self.bn1(x)
        x = self.dp1(x)
        x = torch.sigmoid(self.hidden2(x))
        x = self.bn2(x)
        x = self.dp2(x)
        x = self.predict(x)
        return x

def get_MLP():
    # Training set, test set, and feature names for plotting
    x_train, y_train, features_train = train_set()
    x_test, y_test, _ = test_set()
    # Normalization
    x_train, x_test = norm(x_train, x_test)
    # Convert datasets to tensors and format for use in PyTorch
    train_x, train_y, test_x, test_y = data_trans(x_train, y_train, x_test, y_test)

    # Parameters
    d_model = input_size = x_train.shape[1]  # Input layer
    hidden1_size = 128  # First hidden layer
    hidden2_size = 64   # Second hidden layer
    output_size = 2     # Output layer
    batch_size = 30     # Samples per batch
    learning_rate = 0.045  # Learning rate
    epochs = 100        # Number of epochs
    w_d = 0.0049
    seed = 2  # Seed for reproducibility

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    np.random.seed(seed)              # Numpy module
    random.seed(seed)                 # Python random module
    torch.manual_seed(seed)

    # Convert data to DataLoader
    train_loader, test_loader = loader(train_x, train_y, test_x, test_y, batch_size)

    mlp = MLP(input_size, hidden1_size,hidden2_size, output_size)
    # Optimizer for all parameters (weights and biases in neu, using Adam)
    optimizer = optim.Adam(mlp.parameters(), lr=learning_rate, weight_decay=w_d)
    # Loss function: PyTorch’s built-in cross-entropy loss
    Loss = nn.CrossEntropyLoss()
    losses = []
    # Training loop for the neural network
    for epoch in range(epochs):
        train_loss = []
        # Each batch_size of samples is read as a batch in the loop
        for xx, yy in train_loader:
            outputs = mlp(xx)
            loss = Loss(outputs, yy.long())
            optimizer.zero_grad()   # Clear previous gradients
            loss.backward()         # Backpropagate to get new gradients
            optimizer.step()        # Update weights
            # Loss on the training set, reflects model performance
            # Record each batch’s loss to compute the epoch’s mean loss
            train_loss.append(loss.data.numpy())
        # Compute the average loss for this epoch
        losses.append(np.mean(train_loss))
        print('Epoch:  {}  \tTraining Loss: {:.6f}'.format(epoch + 1, losses[epoch]))

    # showlossgraph(losses)

    # Test phase
    pre_y_test = mlp(test_x)
    pre_y_test = pre_y_test[:, -1].tolist()
    a = average_precision_score(test_y, pre_y_test)
    print("\nTest set accuracy of the model: {:.4f}%".format(100. * a))

    pre_y = mlp(train_x)
    pre_y = pre_y[:, -1].tolist()
    a = average_precision_score(train_y, pre_y)
    print("\nTraining set accuracy of the model: {:.4f}%".format(100. * a))

    str1 = 'Saving the trained MLP model...'
    print(str1)
    path = "mlp_model.pth.tar"
    state = {'model': mlp.state_dict(), 'optimizer': optimizer.state_dict(), 'epochs': epochs}
    torch.save(state, path)
    str2 = 'The trained MLP model has been saved'
    print(str2)

    precision, recall, thresholds = precision_recall_curve(y_test, pre_y_test)
    AP = average_precision_score(y_test, pre_y_test)
    plt.plot(recall, precision, label='MLP(AP=%0.3f)' % AP)
    plt.title("PR curves on the CGC dataset")
    plt.legend()
    plt.show()

    fpr, tpr, thresholds1 = roc_curve(y_test, pre_y_test)
    auc1 = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='MLP(AUC=%0.3f)' % auc1)
    plt.title("ROC curves on the CGC dataset")
    plt.legend()
    plt.show()

    return str1,str2

if __name__ == '__main__':
    get_MLP()
