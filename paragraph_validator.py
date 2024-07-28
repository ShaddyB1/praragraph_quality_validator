import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import optuna
import spacy
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


nlp = spacy.load("en_core_web_sm")

class ParagraphQualityClassifier(nn.Module):
    def __init__(self, max_features=100):
        super(ParagraphQualityClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(768, 128)
        self.fc2 = nn.Linear(128 + max_features, 2)

    def forward(self, input_ids, attention_mask, linguistic_features):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        bert_output = self.fc1(dropout_output)
        combined = torch.cat((bert_output, linguistic_features), dim=1)
        return self.fc2(combined)

def extract_linguistic_features(paragraph, max_features=100):
    doc = nlp(paragraph)
    
   
    pos_counts = {pos: 0 for pos in nlp.pipe_labels['tagger']}
    for token in doc:
        pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
    
  
    transition_words = set(['however', 'therefore', 'thus', 'consequently', 'furthermore'])
    coherence_score = sum(1 for token in doc if token.text.lower() in transition_words)
    
  
    features = torch.tensor([*pos_counts.values(), coherence_score], dtype=torch.float)
    if len(features) < max_features:
        features = torch.cat([features, torch.zeros(max_features - len(features))])
    else:
        features = features[:max_features]
    
    return features

def train_model(model, train_dataloader, val_dataloader, epochs=5, lr=2e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            linguistic_features = batch['linguistic_features'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, linguistic_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                linguistic_features = batch['linguistic_features'].to(device)
                
                outputs = model(input_ids, attention_mask, linguistic_features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        train_losses.append(train_loss / len(train_dataloader))
        val_losses.append(val_loss / len(val_dataloader))
        val_accuracies.append(100 * correct / total)
        
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_losses[-1]:.4f}')
        print(f'Validation Loss: {val_losses[-1]:.4f}')
        print(f'Validation Accuracy: {val_accuracies[-1]:.2f}%')
    
    return train_losses, val_losses, val_accuracies

def validate_paragraph(model, tokenizer, paragraph, max_features=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    inputs = tokenizer(paragraph, return_tensors='pt', truncation=True, padding=True, max_length=512)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    linguistic_features = extract_linguistic_features(paragraph, max_features).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, linguistic_features)
        _, predicted = torch.max(outputs, 1)
    
    return "High quality" if predicted.item() == 1 else "Low quality"

class ParagraphDataset(Dataset):
    def __init__(self, paragraphs, labels, tokenizer, max_features=100):
        self.paragraphs = paragraphs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_features = max_features

    def __len__(self):
        return len(self.paragraphs)

    def __getitem__(self, idx):
        paragraph = self.paragraphs[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(paragraph, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
        linguistic_features = extract_linguistic_features(paragraph, self.max_features)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'linguistic_features': linguistic_features
        }

def optimize_hyperparameters(train_dataset, val_dataset):
    def objective(trial):
        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        model = ParagraphQualityClassifier(max_features=100)
        _, _, val_accuracies = train_model(model, train_loader, val_loader, epochs=3, lr=lr)
        
        return val_accuracies[-1]

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)  
    
    print('Best trial:')
    trial = study.best_trial
    print(f'Value: {trial.value}')
    print('Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')
    
    return trial.params

def plot_learning_curves(train_losses, val_losses, val_accuracies):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Example usage:
if __name__ == "__main__":
    # example paragraphs
    paragraphs = [
        "This is a high-quality paragraph with good structure and coherence.",
        "Poor paragraph no good writing here.",
        "Another excellent paragraph demonstrating clear thoughts and proper grammar.",
        "Bad paragraph structure confusing ideas.",
        "Well-written paragraph with logical flow and proper use of transition words.",
        "Lacking coherence and proper structure in this paragraph.",
        "Clear and concise writing in this high-quality paragraph.",
        "Grammatical errors and poor organization make this a low-quality paragraph.",
        "Effective use of language and strong arguments in this paragraph.",
        "Rambling and unfocused writing in this low-quality example."
    ]
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 for high quality, 0 for low quality
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Split data
    train_paragraphs, temp_paragraphs, train_labels, temp_labels = train_test_split(
        paragraphs, labels, test_size=0.4, random_state=42, stratify=labels
    )
    val_paragraphs, test_paragraphs, val_labels, test_labels = train_test_split(
        temp_paragraphs, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    max_features = 100  
    
    # Create datasets
    train_dataset = ParagraphDataset(train_paragraphs, train_labels, tokenizer, max_features)
    val_dataset = ParagraphDataset(val_paragraphs, val_labels, tokenizer, max_features)
    test_dataset = ParagraphDataset(test_paragraphs, test_labels, tokenizer, max_features)
    
   
    best_params = optimize_hyperparameters(train_dataset, val_dataset)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=best_params['batch_size'])
    test_dataloader = DataLoader(test_dataset, batch_size=best_params['batch_size'])
    
  
    model = ParagraphQualityClassifier(max_features)
    train_losses, val_losses, val_accuracies = train_model(model, train_dataloader, val_dataloader, epochs=5, lr=best_params['lr'])
    
 
    plot_learning_curves(train_losses, val_losses, val_accuracies)

    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            linguistic_features = batch['linguistic_features']
            
            outputs = model(input_ids, attention_mask, linguistic_features)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())
    
    plot_confusion_matrix(y_true, y_pred)
    
    # Example of using the model to validate a new paragraph
    new_paragraph = "This is a new paragraph to validate. It demonstrates good writing skills."
    result = validate_paragraph(model, tokenizer, new_paragraph, max_features)
    print(f"The new paragraph is classified as: {result}")
