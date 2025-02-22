import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import load_data
from torchsummary import summary
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import Models
from datetime import datetime

# add para
parser = argparse.ArgumentParser(description="Process model hyperparameters.")
parser.add_argument('--q', type=int, default=2, help='splitting time')
parser.add_argument('--wavelet', type=str, default='db1', help='Type of wavelet to use.')
parser.add_argument('--epoch', type=int, default=30, help='Number of epochs for training.')
parser.add_argument('--dataset_name', type=str, default = "NTU-Fi-HumanID", help='NTU-Fi-HumanID or NTU-Fi_HAR')
parser.add_argument('--gpu', type=int, default=0, help='0-7')

args = parser.parse_args()

q=args.q
wavelet = args.wavelet
dataset_name = args.dataset_name
num_epochs = args.epoch
gpu_id = args.gpu

train_loader, test_loader, num_classes = load_data.load_data(root='./dataset/', q=q, wavelet=wavelet,
                                                             dataset_name=dataset_name, num_workers=0)
"""
Dataset List:
NTU-Fi-HumanID
q=3, epoch=40: Accuracy: 0.9082, Precision: 0.9206, Recall: 0.9082,
q=2, epoch=30: Accuracy: 0.9116, Precision: 0.9306, Recall: 0.9116,
NTU-Fi_HAR

Widar
UT_HAR_data
"""

for batch in train_loader:
    series_length=batch[0].shape[-1]
    num_features = batch[0].shape[1]# number of input channels
    break

segment_length = int(series_length/(2**(q-1)))

model = Models.TimeSeriesTransformer(num_features=num_features, num_classes=num_classes,
                              segment_length=segment_length, series_length=series_length)
# summary(model, (num_features, series_length))

def get_free_gpu():

    num_gpus = torch.cuda.device_count()
    max_memory = 0
    best_gpu = 0
    for i in range(num_gpus):
        torch.cuda.set_device(i)
        free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
        if free_memory > max_memory:
            max_memory = free_memory
            best_gpu = i
    return best_gpu

"""# Use the GPU with the most free memory
free_gpu = get_free_gpu()
torch.cuda.set_device(free_gpu)

device = torch.device(f'cuda:{free_gpu}' if torch.cuda.is_available() else 'cpu')
"""
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
"""
# Example input tensor
input_tensor = torch.randn(64, 342, 2000).to(device)  # [batch size, channel, length]
output = model(input_tensor)

print(output.shape)  # Expected output shape: [batch size, num_classes]
"""

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_history = []
# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    loss_history.append(average_loss)
    print(f'Epoch {epoch + 1}, Loss: {average_loss:.4f}')


def evaluate_model(model, data_loader, device):
    model.eval()  # Set model to evaluation mode
    true_labels = []
    predictions = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)


            true_labels.extend(labels.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())


    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, fscore, _ = precision_recall_fscore_support(true_labels, predictions, average='macro')

    return accuracy, precision, recall, fscore



accuracy, precision, recall, fscore = evaluate_model(model, test_loader, device)
print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {fscore:.4f}")

########################################Save Res ############################
data_to_save = {
    "Loss History": loss_history,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F-score": fscore
}

# File path to save the data
res_path = os.path.join("./ResultRecord", str(dataset_name), str(wavelet),"q="+str(q))
metric_path = os.path.join(res_path, "model_metrics_"+datetime.now().strftime('%Y_%m_%d_%H_%M_%S')+".txt")
model_path = os.path.join(res_path, "torch_model_"+datetime.now().strftime('%Y_%m_%d_%H_%M_%S')+'.pt')

if not os.path.exists(res_path):
        os.makedirs(res_path)
# Writing to a file
with open(metric_path, 'w') as file:
    for key, value in data_to_save.items():
        file.write(f"{key}: {value}\n")
## write the model
model_scripted = torch.jit.script(model) # Export to TorchScript
model_scripted.save(model_path)