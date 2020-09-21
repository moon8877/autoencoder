import torch
import torch.utils.data as data
import torchvision
import matplotlib.pyplot as plt
train_set = torchvision.datasets.MNIST(
    root = './mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor()
)

train_loader = data.DataLoader(train_set,batch_size=128,shuffle=True)

class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(784,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,16),
            torch.nn.ReLU(),
            torch.nn.Linear(16,10),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(10,16),
            torch.nn.ReLU(),
            torch.nn.Linear(16,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,784),
            torch.nn.ReLU(),
        )

    def forward(self,x):
        codes = self.encoder(x)
        decoded = self.decoder(codes)
        return codes,decoded

model = AutoEncoder()
optimizer = torch.optim.Adam(model.parameters(),lr=0.008)
loss_fn = torch.nn.MSELoss()

for epoch in range(5):
    for data,label in train_loader:
        inputs = data.view(-1,784)
        codes , decoded = model(inputs)
        optimizer.zero_grad()
        loss = loss_fn(decoded,inputs)
        loss.backward()
        optimizer.step()
    print('[{}/{} Loss:{}]'.format(epoch+1,10,loss.item()))

torch.save(model.state_dict(),'./autoencoder/autocoder.pkl')
model.eval()
t_a = torch.rand(1,784)
t_b = torch.bernoulli(t_a)
code,output = model(t_b)
output = output.view(28,28)

plt.imshow(output.data.numpy(),cmap='gray')
plt.show()
