import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from torch.optim import AdamW
from PIL import Image
import json

def process_load_data(path):
    print(f"Data augmentation and Data loading from {path} ...")
    train_dir = path + '/train'
    valid_dir = path + '/valid'
    test_dir = path + '/test'
    
    train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),  
    transforms.RandomHorizontalFlip(),  
    transforms.RandomRotation(10),        
    transforms.ToTensor(),             
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])
    test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),               
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=test_transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)
    
    print("Finished Data augmentation and Data loading.")
    
    return train_loader, valid_loader, test_loader

def build_model(architecture='vgg19', hidden_units=512):
    print(f"Building model with architecture {architecture} and {hidden_units}...")
    if architecture =='vgg19':
        model = models.vgg19(pretrained=True)
        input_units=25088
    elif architecture =='densenet':
        model = models.densenet121(pretrained=True)
        input_units=1024
    
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = nn.Sequential(
          nn.Linear(input_units, hidden_units),
          nn.ReLU(),
          nn.Linear(hidden_units, 102),
          nn.LogSoftmax(dim = 1)
    )
    
    print("Finished building model.")
    
    return model

def train_model(model, train_loader, valid_loader,test_loader, epochs=3, learning_rate=0.0008, gpu= False):
    print("Staring training the model")
    criterion = nn.NLLLoss()
    weight_decay = 1e-4
    optimizer = optim.AdamW(model.classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    device = "cuda" if gpu else "cpu"
    if(device == "cuda"):
        print("Using Gpu for training")
    model.to(device)
    
    epochs = epochs
    print_every = 50
    steps = 0
    train_loss = 0
    best_valid_loss = float('inf')
    epochs_no_improve = 0
    patience = 5
    best_valid_acc = 0

    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                valid_accuracy = 0

                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)

                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()

                        # Calculate validation accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}, "
                      f"Train loss: {train_loss/print_every:.3f}, "
                      f"Valid loss: {valid_loss/len(valid_loader):.3f}, "
                      f"Valid accuracy: {(valid_accuracy/len(valid_loader))*100 :.3f}")

                train_loss = 0

                 # Early stopping logic
                if valid_accuracy > best_valid_acc:
                    best_valid_acc = valid_accuracy
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience and best_valid_acc >= 80:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

        scheduler.step(valid_loss)
        if epochs_no_improve >= patience and best_valid_acc >= 80:
            break
            
#     evaluation = model_evaluation(model, test_loader, criterion, device)
#     print("Finished training model.")
#     print(f"Model accuracy on Testing dataset is {evaluation} percent")
    
    return model

def model_evaluation(model, test_loader, criterion, device):
    test_loss = 0
    test_accuracy = 0
    model.eval() 
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    print("\n Test Results:")
    print(f"\n Test loss: {test_loss/len(test_loader):.3f}, "
          f"\n Test accuracy: {(test_accuracy/len(test_loader))*100:.2f}")
    running_loss = 0
    
def save_checkpoint(model, save_dir, architecture='vgg19', hidden_units=512, epochs=5, learning_rate=0.0008):
    checkpoint = {
        'architecture': architecture,
        'hidden_units': hidden_units,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'classifier': model.classifier,
        'model_state_dict': model.state_dict(),
    }
    checkpoint_dir = save_dir + "checkpoint.pth"
    torch.save(checkpoint, checkpoint_dir)
    
    print(f"Model saved at {checkpoint_dir} percent")
    
#FUNCTIONS USED IN PREDICT.PY

def load_model(checkpoint_path):
    model_info = torch.load(checkpoint_path)
    model = getattr(models, model_info['architecture'])(pretrained=True)
    model.classifier = model_info['classifier']
    model.load_state_dict(model_info['model_state_dict'])
    model.eval()
    
    return model

def process_image(image):
    test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),               
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])
    unprocessed_img = Image.open(image)
    processed_img = test_transform(unprocessed_img)
    return processed_img

def predict(image_path, model, topk=5):
    model.eval()
    device = next(model.parameters()).device 
    image_tensor = process_image(image_path).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs, classes = torch.exp(outputs).topk(topk) #dim=1
        
        return probs[0].tolist(), classes[0].add(1).tolist()
    