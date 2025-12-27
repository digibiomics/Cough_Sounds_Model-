import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score, recall_score, precision_score
from tqdm import tqdm
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        else:
            BCE_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

class MFCCFusionDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, Feats, Time)
        self.X = self.X.repeat(1, 3, 1, 1)  # RGB-style for CNN if wanted
        self.Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.Y)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

class CNN_BiLSTM_Fusion(nn.Module):
    def __init__(self, input_channels=3, feat_bands=64):
        super().__init__()
        # 1. CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(64, 128, (3,3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Dropout(0.3)
        )
        self.rnn_input = feat_bands // 4 * 128
        self.lstm = nn.LSTM(input_size=self.rnn_input, hidden_size=128, num_layers=2, bidirectional=True, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(128*2, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 1)
        )
    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0,3,1,2).contiguous().view(x.size(0), x.size(3), -1)
        _, (h_n, _) = self.lstm(x)
        h = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        return self.classifier(h)

def load_data(path):
    X_train = np.load(os.path.join(path, 'X_train.npy'))
    Y_train = np.load(os.path.join(path, 'Y_train.npy'))
    X_val = np.load(os.path.join(path, 'X_val.npy'))
    Y_val = np.load(os.path.join(path, 'Y_val.npy'))
    return X_train, Y_train, X_val, Y_val

def evaluate_model(model, loader, criterion=None, device="cpu"):
    model.eval()
    total_loss, all_labels, all_probs = 0.0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            if criterion: total_loss += criterion(out, y).item() * x.size(0)
            probs = torch.sigmoid(out)
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    n = len(loader.dataset)
    avg_loss = total_loss/n if criterion else 0
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs).flatten()
    preds = (all_probs > 0.5).astype(int)
    return {
        "avg_loss": avg_loss,
        "acc": accuracy_score(all_labels, preds),
        "f1": f1_score(all_labels, preds, zero_division=0),
        "recall": recall_score(all_labels, preds, zero_division=0),
        "precision": precision_score(all_labels, preds, zero_division=0),
        "auc_roc": roc_auc_score(all_labels, all_probs),
        "report": classification_report(all_labels, preds, target_names=["Negative (0)", "Positive (1)"], zero_division=0),
        "cm": confusion_matrix(all_labels, preds)
    }

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, best_path, writer, device="cpu"):
    best_auc, epochs_no_improve = 0, 0
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-6)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1} [TRAIN]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)
        val_metrics = evaluate_model(model, val_loader, criterion, device=device)
        scheduler.step(val_metrics["auc_roc"])
        writer.add_scalar("Loss/Train", train_loss, epoch+1)
        writer.add_scalar("Loss/Validation", val_metrics["avg_loss"], epoch+1)
        writer.add_scalar("AUC/Validation", val_metrics["auc_roc"], epoch+1)
        print(f"\nEpoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_metrics['avg_loss']:.4f}, Val AUC {val_metrics['auc_roc']:.4f}")
        # Early stopping
        if val_metrics["auc_roc"] > best_auc:
            best_auc = val_metrics["auc_roc"]
            torch.save(model.state_dict(), best_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 8:
                print("Early stopping.")
                break
    writer.close()
    print("Training complete.")
    return best_auc

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PROCESSED_DATA_PATH = '/home/ubuntu/lung_project/TB/processed_features_mfcc'
    MODELS_PATH = '/home/ubuntu/lung_project/TB/18/models_pytorch/'
    TENSORBOARD_PATH = '/home/ubuntu/lung_project/TB/18/tensorboard_logs_pytorch/'
    RUN_VERSION = 'cnn_bilstm_fusion'
    MODEL_FILE_NAME = f'{RUN_VERSION}_best.pth'
    BATCH_SIZE, N_EPOCHS, LR, WEIGHT_DECAY = 64, 40, 1e-4, 5e-4

    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(TENSORBOARD_PATH, exist_ok=True)
    X_train, Y_train, X_val, Y_val = load_data(PROCESSED_DATA_PATH)

    pos_weight = torch.tensor([np.sum(Y_train==0)/np.sum(Y_train==1)], dtype=torch.float32).to(device)
    use_focal = False  # Set True to activate focal loss
    if use_focal:
        criterion = FocalLoss(alpha=1, gamma=2, logits=True)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader = DataLoader(MFCCFusionDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(MFCCFusionDataset(X_val, Y_val), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    model = CNN_BiLSTM_Fusion(feat_bands=X_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    writer = SummaryWriter(log_dir=os.path.join(TENSORBOARD_PATH, RUN_VERSION))
    best_model_path = os.path.join(MODELS_PATH, MODEL_FILE_NAME)

    print("Beginning Training Loop")
    best_auc = train_model(model, train_loader, val_loader, criterion, optimizer, N_EPOCHS, best_model_path, writer, device=device)
    
    # Evaluation
    final_model = CNN_BiLSTM_Fusion(feat_bands=X_train.shape[1]).to(device)
    final_model.load_state_dict(torch.load(best_model_path))
    final_metrics = evaluate_model(final_model, val_loader, criterion, device=device)
    print("\nFinal Evaluation:")
    for k, v in final_metrics.items():
        if isinstance(v, (np.ndarray, list, dict)):
            print(f"{k}: {v}")
        else:
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    with open(os.path.join(MODELS_PATH, f"metrics_{RUN_VERSION}.txt"), "w") as f:
        f.write(str(final_metrics))
