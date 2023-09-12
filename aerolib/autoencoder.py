import torch
import torch.nn as nn
import time
import numpy as np

class reconstruction_loss(nn.Module):
    def __init__(self, alpha, *args, **kwargs) -> None:    
        super().__init__(*args, **kwargs)
        self.criterion1 = nn.MSELoss()
        self.criterion2 = nn.L1Loss()
        self.alpha = alpha

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mse_loss = self.criterion1(x, y)
        l1_loss = self.criterion2(x, y)
        return (l1_loss + self.alpha*mse_loss) / (self.alpha)

def train(md, model, train_x, valid_x):

    print("-"*50)
    print("Setup Training...")
    lr = 0.0001
    betas = (0.9, 0.98)
    eps = 1e-9
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps)
    criterion = reconstruction_loss(alpha=md.alpha)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Used Device: {device}")
    print(f"Optimizer | lr: {lr} | betas: {betas} | eps: {eps}")
    print(f"Criterion alpha: {md.alpha}")
    print("\n")


    train_epoch_loss, valid_epoch_loss = [], []
    train_batch_loss, valid_batch_loss = [], []
    train_total_loss, valid_total_loss = [], []
    
    print("-"*50)
    print("Starting Training...")
    time_start = time.time()
    print(f"Epochs: {md.epochs}")
    for epoch in np.arange(0, md.epochs):
        
        print(f"Epoch: {epoch+1}/{md.epochs}", end="")

        ### TRAINING PHASE ###
        for x_batch, y_batch in (train_x):
            # print("x_batch shape: ", x_batch.shape)
            # print("y_batch shape: ", y_batch.float().shape)
            
           
            y_train = model.forward(x_batch.float())
            train_loss = criterion(y_train.float(), y_batch.float())


            train_batch_loss += [train_loss]
            train_epoch_loss += [train_loss.item()]

            # Backpropagation
            optimizer.zero_grad()
            train_loss.backward()

            # Update the parameters
            optimizer.step()

              
        ### VALIDATION PHASE ###
        for x_batch, y_batch in (valid_x):

            with torch.no_grad():
                model.eval()

                y_valid = model.forward(x_batch.float())
                valid_loss = criterion(y_valid.float(), y_batch.float())

                valid_batch_loss += [valid_loss]
                valid_epoch_loss += [valid_loss.item()]


        print(f"\t Train loss = {sum(train_epoch_loss)/len(train_epoch_loss):.05}, \
                Validation Loss = {sum(valid_epoch_loss)/len(valid_epoch_loss):.05}")

        train_total_loss.append(sum(train_epoch_loss)/len(train_epoch_loss))
        valid_total_loss.append(sum(valid_epoch_loss)/len(valid_epoch_loss))

        train_epoch_loss, valid_epoch_loss = [], []
        train_batch_loss, valid_batch_loss = [], []
    time_end = time.time()
    
    md.train_loss = train_total_loss[-1]
    md.valid_loss = valid_total_loss[-1]
    md.train_time = time_end - time_start



def calculateMSE(x, x_hat):

    MSE = nn.MSELoss(reduction="none")

    mse_per_window = torch.mean(MSE(x,x_hat),dim=[2]) 
    mse_per_sensor = torch.mean(mse_per_window,dim=[0]) 
    mse_overall = torch.mean(mse_per_sensor,dim=[0])
    
    # Find the worst window
    worst_windows_values, worst_windows_idx = torch.max(mse_per_window, dim=0, keepdim=True)
    _ , worst_sensor_idx = torch.max(worst_windows_values, dim=1, keepdim=True)
    s_idx = worst_sensor_idx[0].item()
    w_idx = worst_windows_idx[0][s_idx].item()

    return mse_overall.item(), mse_per_sensor.tolist(), [w_idx, s_idx]



