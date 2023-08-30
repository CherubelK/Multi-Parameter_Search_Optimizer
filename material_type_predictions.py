import pandas as pd
import torch
from torch.optim import Adam
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, precision_recall_fscore_support

def load_and_process_data(file_name):
    """Loads the Excel file and processes the columns to get training data."""
    
    # Load the dataset from an Excel file
    df = pd.read_excel(file_name, sheet_name='material_type')
    
    # Convert the relevant columns to lowercase
    df['material_type'] = df['material_type'].str.lower()
    df['family'] = df['family'].str.lower()
    df['keywords'] = df['keywords'].str.lower()
    
    # Aggregate keywords, material types, and families
    df['keywords'] += ',' + df['material_type'] + ',' + df['family']

    # Prepare the dataset
    material_types = df['material_type'].tolist()
    families = df['family'].tolist()
    keywords = df['keywords'].tolist()
    data = [(kw.strip(), pt) for kw, pt in zip(keywords, material_types) for kw in kw.split(',')]

    # Split the data into training and validation sets
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    
    return train_data, val_data, df, material_types, families

def initialize_tokenizer_and_dataset(train_data):
    """Initializes the tokenizer and encodes the training data."""
    
    # Use the DistilBert tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # Separate texts and labels from the training data
    train_texts = [t[0] for t in train_data]
    train_labels = [t[1] for t in train_data]

    # Encode the labels into numbers
    le = LabelEncoder()
    train_labels = le.fit_transform(train_labels)

    # Tokenize the training texts
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    
    return train_encodings, train_labels, le, tokenizer

class MaterialTypeDataset(torch.utils.data.Dataset):
    """Defines the custom dataset for material type prediction."""
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Fetch the items and convert them to tensors
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def initialize_model_and_data_loaders(train_encodings, train_labels, val_encodings, val_labels, le):
    """Initializes the model, moves it to the appropriate device, and prepares the data loaders."""
    
    # Initialize the model with the appropriate number of labels
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(le.classes_))
    
    # Check for GPU availability and move the model to the device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()

    # Prepare the datasets and data loaders
    train_dataset = MaterialTypeDataset(train_encodings, train_labels)
    val_dataset = MaterialTypeDataset(val_encodings, val_labels)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Use the Adam optimizer for training
    optim = Adam(model.parameters(), lr=5e-5)
    
    return model, train_loader, val_loader, device, optim

def training_loop(model, train_loader, val_loader, optim, device, num_epochs=10):
    # Lists to store training progress metrics
    epoch_nums = []
    training_loss = []
    validation_acc = []
    validation_f1 = []

    # Loop through each epoch
    for epoch in range(num_epochs):
        # Initialize loss for this epoch
        total_loss = 0
        
        # Lists to store all predictions and labels for this epoch (for metrics calculation)
        all_preds = []
        all_labels = []
        
        # Training phase
        for batch in train_loader:
            # Zero out any gradients from the previous iteration
            optim.zero_grad()
            
            # Transfer batch tensors to the computation device (CPU or GPU)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass: compute model predictions based on input
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            
            # Compute the loss
            loss = outputs.loss
            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # Update model parameters based on computed gradients
            optim.step()
            
            # Accumulate the training loss
            total_loss += loss.item()

        # Evaluation phase
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                # Transfer batch tensors to the computation device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Compute model predictions
                outputs = model(input_ids, attention_mask=attention_mask)
                # Get the predicted class labels
                _, predicted = torch.max(outputs.logits, 1)
                
                # Store predictions and true labels for metrics computation
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        # Compute validation metrics
        validation_accuracy = (predicted == labels).sum().item() / labels.size(0)
        validation_f1_score = f1_score(all_labels, all_preds, average='macro')

        print(f"Epoch: {epoch+1}, Loss: {total_loss/len(train_loader)}, Validation Accuracy: {validation_accuracy}, Validation F1-Score: {validation_f1_score}")

        # Switch model back to training mode for the next epoch
        model.train()

        # Store metrics values for this epoch
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
def predict(input_text, model, tokenizer, le, device, df, material_types, families):
    # Convert input text to lowercase
    input_text = input_text.lower()

    # Check if the input text matches any known material types or families
    if input_text in material_types:
        print(f"material type: {input_text}, Probability: 1.0")
        return
    if input_text in families:
        associated_product_types = df[df['family'] == input_text]['material_type'].unique()
        for pt in associated_product_types:
            print(f"material type: {pt}, Probability: 1.0/{len(associated_product_types)}")
        return

    # If not, proceed with model prediction
    model.eval()

    # Tokenize the input text for model input
    input_text_encoded = tokenizer(input_text, truncation=True, padding=True, return_tensors='pt').to(device)

    # Run the model inference to get logits
    with torch.no_grad():
        logits = model(**input_text_encoded).logits

    # Convert logits to probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)
    top_5_probs, top_5_labels = torch.topk(probs, 5)

    # Decode the top predicted labels back to material type strings
    top_5_labels = le.inverse_transform(top_5_labels.cpu().detach().numpy()[0])
    top_5_probs = top_5_probs.cpu().detach().numpy()[0]

    # Convert top predictions and their probabilities to lists for easier processing
    predicted_product_types = list(top_5_labels)
    predicted_probabilities = list(top_5_probs)

    # Normalize the predicted probabilities to ensure they sum to 1
    predicted_probabilities = [p / sum(predicted_probabilities) for p in predicted_probabilities]

    # Print out the predicted material types along with their associated probabilities
    for pt, prob in zip(predicted_product_types, predicted_probabilities):
        print(f"material type: {pt}, Probability: {prob}")
