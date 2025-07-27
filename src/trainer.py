import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os
from tqdm import tqdm
import json
from src.rf_detr_model import build_rf_detr
from src.loss import build_criterion
from src.data_loader import create_data_loaders

class RFDETRTrainer:
    def __init__(self, dataset_path, num_classes, num_queries=100, batch_size=4, 
                 learning_rate=1e-4, device='cuda'):
        self.device = device
        self.num_classes = num_classes
        
        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader, self.categories = create_data_loaders(
            dataset_path, batch_size
        )
        
        # Build model
        self.model = build_rf_detr(num_classes, num_queries).to(device)
        
        # Build criterion
        self.criterion = build_criterion(num_classes)
        
        # Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
    def prepare_targets(self, batch):
        """Convert batch format to target format expected by criterion"""
        targets = []
        for i in range(len(batch['labels'])):
            num_objects = batch['num_objects'][i].item()
            if num_objects > 0:
                target = {
                    'labels': batch['labels'][i][:num_objects],
                    'boxes': batch['boxes'][i][:num_objects]
                }
            else:
                target = {
                    'labels': torch.tensor([], dtype=torch.long, device=self.device),
                    'boxes': torch.tensor([], dtype=torch.float32, device=self.device).reshape(0, 4)
                }
            targets.append(target)
        return targets
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            # Move to device
            images = batch['image'].to(self.device)
            targets = self.prepare_targets(batch)
            
            # Forward pass
            outputs = self.model(images)
            
            # Compute loss
            loss_dict = self.criterion(outputs, targets)
            weight_dict = self.criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.optimizer.step()
            
            total_loss += losses.item()
            pbar.set_postfix({'loss': losses.item()})
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                images = batch['image'].to(self.device)
                targets = self.prepare_targets(batch)
                
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, targets)
                weight_dict = self.criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                
                total_loss += losses.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self, num_epochs, save_dir='checkpoints'):
        """Train the model"""
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # Update scheduler
            self.scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                }, os.path.join(save_dir, 'best_model.pth'))
                print("Saved best model!")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Save training history
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump({
                'train_losses': self.train_losses,
                'val_losses': self.val_losses
            }, f)
        
        print("Training completed!")

if __name__ == "__main__":
    # Example usage
    dataset_path = "path/to/your/dataset"  # Update this
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    trainer = RFDETRTrainer(
        dataset_path=dataset_path,
        num_classes=1,  # Car detection
        batch_size=4,
        learning_rate=1e-4,
        device=device
    )
    
    trainer.train(num_epochs=20)
