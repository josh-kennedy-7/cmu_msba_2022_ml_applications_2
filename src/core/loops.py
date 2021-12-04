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
            pred = model(X.type(torch.long)).flatten()
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
        print(f"\nEpoch Done, max loss:{max_loss:.3f}, min loss: {min_loss:.3f}, final loss: {loss.item():.3f}")


def test_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    loss = torch.zeros(1).cuda()

    with torch.no_grad():
        for X, y in dataloader:
            X = X.cuda()
            y = y.cuda()

            pred = model(X.type(torch.long)).flatten()
            loss += loss_fn(pred, y).item()
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    torch.cuda.empty_cache()
    loss /= num_batches
    print(f"Test Error: \n Avg loss: {loss} \n")
    return loss