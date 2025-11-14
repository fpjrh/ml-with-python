import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Create class for plotting

def plot_data(data_set, model = None, n = 1, color = False):
    X = data_set[:][0]
    Y = data_set[:][1]
    plt.plot(X[Y == 0, 0].numpy(), Y[Y == 0].numpy(), 'bo', label = 'y = 0')
    plt.plot(X[Y == 1, 0].numpy(), 0 * Y[Y == 1].numpy(), 'ro', label = 'y = 1')
    plt.plot(X[Y == 2, 0].numpy(), 0 * Y[Y == 2].numpy(), 'go', label = 'y = 2')
    plt.ylim((-0.1, 3))
    plt.legend()
    if model != None:
        w = list(model.parameters())[0][0].detach()
        b = list(model.parameters())[1][0].detach()
        y_label = ['yhat=0', 'yhat=1', 'yhat=2']
        y_color = ['b', 'r', 'g']
        Y = []
        for w, b, y_l, y_c in zip(model.state_dict()['0.weight'], model.state_dict()['0.bias'], y_label, y_color):
            Y.append((w * X + b).numpy())
            plt.plot(X.numpy(), (w * X + b).numpy(), y_c, label = y_l)
        if color == True:
            x = X.numpy()
            x = x.reshape(-1)
            top = np.ones(x.shape)
            y0 = Y[0].reshape(-1)
            y1 = Y[1].reshape(-1)
            y2 = Y[2].reshape(-1)
            plt.fill_between(x, y0, where = y0 > y1, interpolate = True, color = 'blue')
            plt.fill_between(x, y0, where = y1 > y2, interpolate = True, color = 'blue')
            plt.fill_between(x, y1, where = y1 > y0, interpolate = True, color = 'red')
            plt.fill_between(x, y1, where = ((y1 > y2) * (y1 > y0)),interpolate = True, color = 'red')
            plt.fill_between(x, y2, where = (y2 > y0) * (y0 > 0),interpolate = True, color = 'green')
            plt.fill_between(x, y2, where = (y2 > y1), interpolate = True, color = 'green')
    plt.legend()
    plt.show()

# Create evaluation function to reduce code duplication
def evaluate_model(model, dataloader, dataset_name="Dataset"):
    """
    Evaluate model on a given dataset and return predictions and accuracy.
    """
    model.eval()
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            z = model(x)
            _, yhat = z.max(1)
            all_predictions.extend(yhat.numpy())
            all_labels.extend(y.numpy())
            correct += (y == yhat).sum().item()
            total += y.size(0)
    
    accuracy = correct / total
    print(f"{dataset_name} Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    return np.array(all_predictions), np.array(all_labels), accuracy

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    Plot confusion matrix using seaborn heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Class 0', 'Class 1', 'Class 2'],
                yticklabels=['Class 0', 'Class 1', 'Class 2'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Function to calculate and display class-wise accuracy
def class_wise_accuracy(y_true, y_pred, num_classes=3):
    """
    Calculate and display accuracy for each class.
    """
    print("\nClass-wise Accuracy:")
    for class_idx in range(num_classes):
        class_mask = y_true == class_idx
        if class_mask.sum() > 0:
            class_acc = (y_pred[class_mask] == y_true[class_mask]).sum() / class_mask.sum()
            print(f"  Class {class_idx}: {class_acc:.4f} ({(y_pred[class_mask] == y_true[class_mask]).sum()}/{class_mask.sum()})")

# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)
# Create the data class

class Data(Dataset):
    
    # Constructor
    def __init__(self):
        self.x = torch.arange(-2, 2, 0.1).view(-1, 1)
        self.y = torch.zeros(self.x.shape[0])
        self.y[(self.x > -1.0)[:, 0] * (self.x < 1.0)[:, 0]] = 1
        self.y[(self.x >= 1.0)[:, 0]] = 2
        self.y = self.y.type(torch.LongTensor)
        self.len = self.x.shape[0]
        
    # Getter
    def __getitem__(self,index):      
        return self.x[index], self.y[index]
    
    # Get Length
    def __len__(self):
        return self.len
# Create the dataset object and plot the dataset object

data_set = Data()
data_set.x
plot_data(data_set)

# Split dataset into train/validation/test sets (70/15/15 split)

train_size = int(0.7 * len(data_set))
val_size = int(0.15 * len(data_set))
test_size = len(data_set) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(data_set, [train_size, val_size, test_size])

print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size: {len(test_dataset)}")

# Build Softmax Classifier technically you only need nn.Linear

model = nn.Sequential(nn.Linear(1, 3))
model.state_dict()

# Create criterion function, optimizer, and dataloaders

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
trainloader = DataLoader(dataset = train_dataset, batch_size = 5, shuffle = True)
valloader = DataLoader(dataset = val_dataset, batch_size = 5)
testloader = DataLoader(dataset = test_dataset, batch_size = 5)

# Train the model with validation

LOSS = []
VAL_LOSS = []
VAL_ACCURACY = []

def train_model(epochs):
    """
    Train model with validation monitoring.
    """
    print("\nStarting training...")
    model.train()
    
    for epoch in range(epochs):
        # Training phase
        epoch_loss = 0
        num_batches = 0
        
        for x, y in trainloader:
            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            LOSS.append(loss.item())
            epoch_loss += loss.item()
            num_batches += 1
            loss.backward()
            optimizer.step()
        
        # Validation phase every 50 epochs
        if epoch % 50 == 0:
            avg_train_loss = epoch_loss / num_batches
            
            # Evaluate on validation set
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for x, y in valloader:
                    yhat = model(x)
                    loss = criterion(yhat, y)
                    val_loss += loss.item()
                    _, predicted = yhat.max(1)
                    val_correct += (predicted == y).sum().item()
                    val_total += y.size(0)
            
            val_loss /= len(valloader)
            val_acc = val_correct / val_total
            VAL_LOSS.append(val_loss)
            VAL_ACCURACY.append(val_acc)
            
            print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Plot decision boundaries
            plot_data(data_set, model)
            
            model.train()
    
    print("\nTraining completed!")

train_model(300)

# Plot training loss
print("\n" + "="*60)
print("TRAINING LOSS VISUALIZATION")
print("="*60)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(LOSS)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Over Iterations')
plt.grid(True)

plt.subplot(1, 2, 2)
if VAL_LOSS:
    epochs_recorded = list(range(0, 300, 50))
    plt.plot(epochs_recorded, VAL_LOSS, 'o-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# Evaluate on all datasets
print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

train_preds, train_labels, train_acc = evaluate_model(model, trainloader, "Training")
val_preds, val_labels, val_acc = evaluate_model(model, valloader, "Validation")
test_preds, test_labels, test_acc = evaluate_model(model, testloader, "Test")

# Display class-wise accuracy for training set
print("\n--- Training Set ---")
class_wise_accuracy(train_labels, train_preds)

# Display class-wise accuracy for validation set
print("\n--- Validation Set ---")
class_wise_accuracy(val_labels, val_preds)

# Display class-wise accuracy for test set
print("\n--- Test Set ---")
class_wise_accuracy(test_labels, test_preds)

# Plot confusion matrices
print("\n" + "="*60)
print("CONFUSION MATRICES")
print("="*60)

plot_confusion_matrix(train_labels, train_preds, "Training Set Confusion Matrix")
plot_confusion_matrix(val_labels, val_preds, "Validation Set Confusion Matrix")
plot_confusion_matrix(test_labels, test_preds, "Test Set Confusion Matrix")

# Print classification report for test set
print("\n" + "="*60)
print("DETAILED CLASSIFICATION REPORT (Test Set)")
print("="*60)
print(classification_report(test_labels, test_preds, 
                          target_names=['Class 0', 'Class 1', 'Class 2']))

# Make prediction on full dataset for visualization
model.eval()
with torch.no_grad():
    z = model(data_set.x)
    _, yhat = z.max(1)
    Softmax_fn = nn.Softmax(dim=-1)
    Probability = Softmax_fn(z)

print("\n" + "="*60)
print("SAMPLE PREDICTIONS AND PROBABILITIES")
print("="*60)
print("Sample predictions (first 10):", yhat[:10])
print("\nProbabilities for first sample:")
for i in range(3):
    print(f"  Class {i}: {Probability[0,i]:.4f}")

# Save the model
print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)
model_path = 'softmax_classifier.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to: {model_path}")

# Plot final decision boundaries with test data highlighted
print("\n" + "="*60)
print("FINAL MODEL VISUALIZATION")
print("="*60)
plot_data(data_set, model, color=True)
    
