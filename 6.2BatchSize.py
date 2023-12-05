import torch
import torch.utils.data as Data

torch.manual_seed(1234)  # reproducible
batch_size = 5

x = torch.linspace(1, 10, 10)  # this is x data (torch tensor)
y = torch.linspace(10, 1, 10)  # this is y data (torch tensor)

torch_dataset = Data.TensorDataset(x, y)

if __name__ == '__main__':
    loader = Data.DataLoader(
        dataset=torch_dataset,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,  # random shuffle for training
        num_workers=2  # subprocesses for loading data
    )

    for epoch in range(5):  # train entire dataset 5 times
        # for each training step
        for step, (batch_x, batch_y) in enumerate(loader):
            # write your code here to train data in mini-batch
            print('Epoch: ', epoch, ' | Step: ', step, ' | batch x: ',
                  batch_x.numpy(), ' | batch y: ', batch_y.numpy())
