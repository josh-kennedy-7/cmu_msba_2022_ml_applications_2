import torch
from tqdm import tqdm

def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    max_loss = 0.0
    min_loss = 9e9

    with tqdm(total=len(dataloader),leave=False) as t:

        for batch, (X, y) in enumerate(dataloader):
            X = X.cuda()
            y = y.cuda()

            # Compute prediction and loss
            if device == 'encoder':
                pred = model(X.type(torch.long)).flatten()
            elif device == 'linmod':
                pred = model(X).flatten()
            loss = loss_fn(pred, y)

            if loss.item() > max_loss:
                max_loss = loss.item()

            if loss.item() < min_loss:
                min_loss = loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t.set_postfix(current_loss=loss.item(), refresh=False)
            t.update()

        torch.cuda.empty_cache()
        print(f"\nEpoch Done, max loss:{max_loss:.3e}, min loss: {min_loss:.3e}, final loss: {loss.item():.3e}")


def test_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    loss = torch.zeros(1).cuda()

    with torch.no_grad():
        for X, y in dataloader:
            X = X.cuda()
            y = y.cuda()

            if device == 'encoder':
                pred = model(X.type(torch.long)).flatten()
            elif device == 'linmod':
                pred = model(X).flatten()

            loss += loss_fn(pred, y).item()


    torch.cuda.empty_cache()
    loss /= num_batches
    print(f"Test Error: \n Avg loss: {loss.cpu().numpy()[0]:.3e} \n")
    return loss.cpu().numpy()