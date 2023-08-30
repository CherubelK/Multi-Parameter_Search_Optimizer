import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.optim import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, precision_recall_fscore_support   # <-- Change

def load_and_process_data(file_name):
    df = pd.read_excel(file_name, sheet_name='building_applications')
    df['building_applications'] = df['building_applications'].str.lower()
    df['keywords'] = df['keywords'].str.lower()
    df['keywords'] += ',' + df['building_applications']

    building_applications = df['building_applications'].tolist()
    keywords = df['keywords'].tolist()
    data = [(kw.strip(), pt) for kw, pt in zip(keywords, building_applications) for kw in kw.split(',')]

    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    return train_data, val_data, df, building_applications

# Initialize the tokenizer and prepare the dataset
def initialize_tokenizer_and_dataset(train_data):
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    train_texts = [t[0] for t in train_data]
    train_labels = [t[1] for t in train_data]

    le = LabelEncoder()
    train_labels = le.fit_transform(train_labels)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    return train_encodings, train_labels, le, tokenizer

# Define PyTorch Dataset class
class MaterialTypeDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Initialize the model and data loaders
def initialize_model_and_data_loaders(train_encodings, train_labels, val_encodings, val_labels, le):
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(le.classes_))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()

    train_dataset = MaterialTypeDataset(train_encodings, train_labels)
    val_dataset = MaterialTypeDataset(val_encodings, val_labels)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

    optim = Adam(model.parameters(), lr=5e-5)
    return model, train_loader, val_loader, device, optim

def training_loop(model, train_loader, val_loader, optim, device, num_epochs=10):
    epoch_nums = []
    training_loss = []
    validation_acc = []
    validation_f1 = []

    for epoch in range(num_epochs):
        total_loss = 0
        all_preds = []
        all_labels = []
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optim.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                _, predicted = torch.max(outputs.logits, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        validation_accuracy = (predicted == labels).sum().item() / labels.size(0)
        validation_f1_score = f1_score(all_labels, all_preds, average='macro')

        print(f"Epoch: {epoch+1}, Loss: {total_loss/len(train_loader)}, Validation Accuracy: {validation_accuracy}, Validation F1-Score: {validation_f1_score}")

        model.train()

        epoch_nums.append(epoch+1)
        training_loss.append(total_loss/len(train_loader))
        validation_acc.append(validation_accuracy)
        validation_f1.append(validation_f1_score)

    return epoch_nums, training_loss, validation_acc, validation_f1

def plot_results(epoch_nums, training_loss, validation_acc, validation_f1):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(epoch_nums, training_loss, 'g-')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.subplot(1, 3, 2)
    plt.plot(epoch_nums, validation_acc, 'b-')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.subplot(1, 3, 3)
    plt.plot(epoch_nums, validation_f1, 'r-')
    plt.title('Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.tight_layout()
    plt.show()


# Define the prediction function
def predict(input_text, model, tokenizer, le, device, df, building_applications):
    input_text = input_text.lower()

    if input_text in building_applications:
        print(f"building_applications: {input_text}, Probability: 1.0")
        return

    model.eval()
    input_text_encoded = tokenizer(input_text, truncation=True, padding=True, return_tensors='pt').to(device)

    with torch.no_grad():
        logits = model(**input_text_encoded).logits

    probs = torch.nn.functional.softmax(logits, dim=-1)
    top_5_probs, top_5_labels = torch.topk(probs, 5)

    top_5_labels = le.inverse_transform(top_5_labels.cpu().detach().numpy()[0])
    top_5_probs = top_5_probs.cpu().detach().numpy()[0]

    predicted_building_applications = list(top_5_labels)
    predicted_probabilities = list(top_5_probs)

    predicted_probabilities = [p / sum(predicted_probabilities) for p in predicted_probabilities]

    for pt, prob in zip(predicted_building_applications, predicted_probabilities):
        print(f"building_applications: {pt}, Probability: {prob}")