import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import ast
# Define the architecture for the encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=2048):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.act(self.fc4(x))
        x = self.act(self.fc5(x))
        mean = self.mean(x)
        logvar = self.logvar(x)
        return mean, logvar

# Define the architecture for the decoder
# Be careful, the input_dim here means output_dim for decoder
class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=2048):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, input_dim)
        self.act = nn.LeakyReLU()

    def forward(self, z):
        # x = torch.cat([z, x], dim=1)
        z = self.act(self.fc1(z))
        z = self.act(self.fc2(z))
        z = self.act(self.fc3(z))
        z = self.act(self.fc4(z))
        z = self.act(self.fc5(z))
        x_hat = self.out(z)
        return x_hat

# Define the CVAE
class CVAE(nn.Module):
    def __init__(self, input_dim=9, latent_dim=6, hidden_dim=2048):
        super(CVAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim, hidden_dim)
        self.decoder = Decoder(input_dim, latent_dim, hidden_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        # x_pos = x[:, :3]
        # x_degrees = x[:, 3:]
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x_recon = self.decoder(z)
        return x_recon, mean, logvar

# Define the loss function
# Define the loss function
def loss_function(x_recon, x, mean, logvar):
    pos_recon = x_recon[:, 6:9]
    pos = x[:, 6:9]
    pos_loss = nn.functional.mse_loss(pos_recon, pos)
    degrees_recon = x_recon[:, :6]
    degrees = x[:, :6]
    degrees_loss = nn.functional.mse_loss(degrees_recon, degrees)
    kld_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    if torch.isnan(pos_loss) or torch.isnan(degrees_loss) or torch.isnan(kld_loss):
        print(f"x: {x}")
        print(f"x_recon: {x_recon}")
        print(f"kl: {kld_loss}")
    return pos_loss, degrees_loss, kld_loss



class MyDataset(Dataset):
    def __init__(self, data_file):
        
        # Load data from file
        file = open("valid_data_0710.txt", "r")
        Lines = file.readlines()
        self.data = []
        for idx, line in enumerate(Lines):
            if idx == 0:
                continue
            data_list = ast.literal_eval(line)
            self.data.append(torch.FloatTensor(data_list))
        
        # # read data in another way
        # valid_poses = np.loadtxt("valid_data_0708.txt", delimiter=',', usecols=range(9), skiprows=1)
        # print(f"first data: {valid_poses[0]}")
        
        # self.data = torch.FloatTensor(valid_poses)
        # print(f"data size: {self.data.size()}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256, help="please input batch_size")
    parser.add_argument("--n_epochs", type=int, default=10000, help="number of epochs")
    parser.add_argument("--lrate", type=float, default=1e-6, help="learning rate")
    parser.add_argument("--test_fre", type=int, default=400, help="frequency of testing")
    parser.add_argument("--device", default="cuda:0", help="device")
    parser.add_argument("--logdir", default="log/cvae", help="log directory for tensorboard")
    parser.add_argument("--latent_dim", default=6, help="latent_dim")
    parser.add_argument("--input_dim", default=9, help="input_dim")
    parser.add_argument("--beta", type=float, default=0.005, help="weight for position to degree")
    parser.add_argument("--checkpoint_dir", default="checkpoint", help="folder name of checkpoint")
    args = parser.parse_args()
    # Assuming your data is stored in a Numpy array called `data`
    writer = SummaryWriter(args.logdir)
    dataset = MyDataset("./valid_data_0710.txt")
    train_dataset = dataset

    cvae = CVAE(input_dim=args.input_dim, latent_dim=args.latent_dim)
    cvae.to(args.device)
    optimizer = optim.Adam(cvae.parameters(), lr=args.lrate)

    for epoch in range(args.n_epochs):
        cvae.train()
        train_loss = 0.0
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        for batch_idx, inputs in tqdm(enumerate(train_dataloader)):
            inputs = inputs.to(args.device)
            optimizer.zero_grad()
            x_recon, mean, logvar = cvae(inputs)
            pos_loss, degrees_loss, kld_loss = loss_function(x_recon, inputs, mean, logvar)
            # loss = epoch/args.n_epochs * pos_loss + degrees_loss + epoch/args.n_epochs * kld_loss
            loss = degrees_loss + epoch/args.n_epochs * pos_loss + args.beta * epoch/args.n_epochs * kld_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
 
            writer.add_scalar("train_pos_loss", pos_loss, epoch)
            writer.add_scalar("train_degrees_loss", degrees_loss, epoch)
            writer.add_scalar("train_kld_loss", kld_loss, epoch)
        # print(f"pos_loss: {pos_loss}, degrees_loss: {degrees_loss}, kld_loss: {kld_loss}")
        print('Epoch: {} Train loss: {:.6f}'.format(epoch+1, train_loss/len(train_dataloader)))
        # print("========================================")

        if epoch % args.test_fre == 0:
            # test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
            cvae.eval()
            test_loss = 0.0
            with torch.no_grad():
                for batch_idx, inputs in enumerate(train_dataloader):
                    inputs = inputs.to(args.device)
                    x_recon, mean, logvar = cvae(inputs)
                    pos_loss, degrees_loss, kld_loss = loss_function(x_recon, inputs, mean, logvar)
                    writer.add_scalar("test_pos_loss", pos_loss, epoch)
                    writer.add_scalar("test_degrees_loss", degrees_loss, epoch)
                    writer.add_scalar("test_kld_loss", kld_loss, epoch)
                    # loss = pos_loss + degrees_loss + beta * epoch/args.n_epochs * kld_loss
                    loss = degrees_loss + epoch/args.n_epochs * pos_loss
                    test_loss += loss.item()
            checkpoint = {
                'state_dict': cvae.state_dict(),
                'epoch': epoch,
                'loss': test_loss
                }
            torch.save(checkpoint, f'./{args.checkpoint_dir}/model_{epoch}.pth')
            print('Epoch: {} Test loss: {:.6f}'.format(epoch+1, test_loss/len(train_dataloader)))
            print("========================================")
    





