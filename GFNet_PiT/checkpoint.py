import os
import torch

# Đường dẫn lưu checkpoint
checkpoint_path = "./Check_Point/model_checkpoint.pth"
checkpoint_path_best = "./Check_Point/model_bestcheckpoint.pth"

# Hàm lưu checkpoint
def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")
 
# Hàm load checkpoint
def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded! Resuming from Epoch {epoch + 1}, Loss: {loss}")
        return model, optimizer, epoch + 1, loss
    else:
        print("No checkpoint found, starting from scratch.")
        return model, optimizer, 0, None