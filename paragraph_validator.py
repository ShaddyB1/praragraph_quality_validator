import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
import numpy as np
import spacy
import re


nlp = spacy.load("en_core_web_sm")

class ParagraphFeatures:
    def __init__(self, text):
        self.text = text
        self.doc = nlp(text)
    
    def get_sentence_count(self):
        return len(list(self.doc.sents))
    
    def get_word_count(self):
        return len([token for token in self.doc if not token.is_punct])
    
    def get_average_sentence_length(self):
        sentences = list(self.doc.sents)
        if not sentences:
            return 0
        return sum(len([token for token in sent if not token.is_punct]) for sent in sentences) / len(sentences)
    
    def get_coherence_score(self):
        sentences = list(self.doc.sents)
        if len(sentences) < 2:
            return 0
        
        coherence_scores = []
        for i in range(len(sentences) - 1):
            current_sent = set(token.lemma_ for token in sentences[i] if token.is_alpha)
            next_sent = set(token.lemma_ for token in sentences[i+1] if token.is_alpha)
            overlap = len(current_sent.intersection(next_sent))
            coherence_scores.append(overlap / max(len(current_sent), len(next_sent)))
        
        return sum(coherence_scores) / len(coherence_scores)
    
    def has_topic_sentence(self):
        first_sentence = next(self.doc.sents)
        return any(token.dep_ == "nsubj" for token in first_sentence)
    
    def get_features(self):
        return torch.tensor([
            self.get_sentence_count(),
            self.get_word_count(),
            self.get_average_sentence_length(),
            self.get_coherence_score(),
            int(self.has_topic_sentence())
        ], dtype=torch.float)

class ParagraphQualityClassifier(nn.Module):
    def __init__(self):
        super(ParagraphQualityClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(768 + 5, 256)  
        self.fc2 = nn.Linear(256, 2)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, features):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        combined = torch.cat((pooled_output, features), dim=1)
        x = self.dropout(combined)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class ParagraphDataset(Dataset):
    def __init__(self, paragraphs, labels, tokenizer, max_length=128):
        self.paragraphs = paragraphs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.paragraphs)

    def __getitem__(self, idx):
        paragraph = self.paragraphs[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            paragraph,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        features = ParagraphFeatures(paragraph).get_features()

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'features': features,
            'labels': torch.tensor(label, dtype=torch.long)
        }
class ParagraphQualityScorer(nn.Module):
    def __init__(self):
        super(ParagraphQualityScorer, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        x = self.dropout(pooled_output)
        x = self.fc(x)
        return self.sigmoid(x)

def classify_paragraph(model, tokenizer, paragraph):
    inputs = tokenizer(paragraph, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        score = model(inputs['input_ids'], inputs['attention_mask']).item()
    
    if score < 0.33:
        return f"Low Quality (Score: {score:.2f})"
    elif score < 0.67:
        return f"Medium Quality (Score: {score:.2f})"
    else:
        return f"High Quality (Score: {score:.2f})"
        
def train_model(model, train_loader, val_loader, epochs=10, lr=2e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                features = batch['features'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask, features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Validation Loss: {val_loss/len(val_loader):.4f}')
        print(f'Validation Accuracy: {100 * correct / total:.2f}%')
        print()

    return model

def main():
   
    paragraphs = [
        "This is a well-structured paragraph with clear ideas and proper grammar. It contains multiple sentences that are coherent and related to a single topic. The sentences flow logically from one to the next, creating a unified whole.",
        "Poor writing confusing ideas no structure. One sentence only.",
        "The author presents a compelling argument with strong evidence and logical flow. The paragraph begins with a clear topic sentence, followed by supporting details and examples. It concludes with a sentence that reinforces the main idea.",
        "Lacks coherence jumbled thoughts grammatical errors. Sentences don't connect. Ideas jump around. No clear topic or purpose.",
        "A concise and informative paragraph that effectively communicates its main point. Although brief, it contains multiple sentences that work together to convey a single idea. The language is clear and precise.",
        "Rambling sentences no clear topic unfocused writing. This text goes on and on without making a point. It's hard to follow because there's no central theme. The ideas are all over the place and don't connect well.",
        "One",
        "The",
        "The quick brown fox jumps over the lazy dog.",
        "This paragraph demonstrates excellent use of vocabulary and varied sentence structure. It begins with a strong topic sentence that introduces the main idea. The following sentences elaborate on this idea, providing specific details and examples. The paragraph concludes with a sentence that ties everything together, reinforcing the central theme.",
        "Run on sentence with no punctuation or capitalization just a stream of consciousness that goes on and on without any clear breaks or pauses making it difficult to understand or follow the intended meaning if there even is one",
        "Vague generalities without specific examples or supporting evidence. Some say things happen. Others disagree. It's complicated. No one really knows for sure.",
        "This well-crafted paragraph showcases the author's command of language. It starts with an engaging topic sentence that captures the reader's attention. The subsequent sentences develop the main idea logically, using specific details and vivid language. The paragraph concludes with a thoughtful statement that leaves a lasting impression on the reader."
    ]
    labels = [1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1]  # 1 for proper paragraph, 0 for not a proper paragraph

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_paragraphs, val_paragraphs, train_labels, val_labels = train_test_split(
        paragraphs, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_dataset = ParagraphDataset(train_paragraphs, train_labels, tokenizer)
    val_dataset = ParagraphDataset(val_paragraphs, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)

    model = ParagraphQualityClassifier()

    trained_model = train_model(model, train_loader, val_loader, epochs=10)

    print("Training completed.")

    torch.save(trained_model.state_dict(), 'paragraph_structure_model.pth')
    print("Model saved successfully.")

    # Tests
    test_paragraphs = [
        "This is a well-written test paragraph to demonstrate the model's capability. It contains multiple sentences that are coherent and related to a single topic. The sentences flow logically, creating a unified whole.",
        "Bad grammar no sense. One sentence only.",
        "One",
        "The quick brown fox jumps over the lazy dog.",
        "This insightful analysis is supported by relevant examples and clear argumentation. The paragraph begins with a strong thesis statement, followed by several sentences that provide evidence and explanation. It concludes by reinforcing the main point, tying all the ideas together effectively."
    ]

    trained_model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for test_paragraph in test_paragraphs:
        test_encoding = tokenizer.encode_plus(
            test_paragraph,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        test_features = ParagraphFeatures(test_paragraph).get_features().unsqueeze(0)

        with torch.no_grad():
            outputs = trained_model(
                test_encoding['input_ids'].to(device),
                test_encoding['attention_mask'].to(device),
                test_features.to(device)
            )
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        quality = "Proper Paragraph" if predicted_class == 1 else "Not a Proper Paragraph"
        print(f"\nTest Paragraph: {test_paragraph}")
        print(f"Prediction: {quality}")
        print(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    main()
