import torch
from tqdm import tqdm

def train_loop(dataloader, model, loss_fn, optimizer, method, in_device, board=None, epoch=0):
    size = len(dataloader.dataset)
    max_loss = 0.0
    min_loss = 9e9

    with tqdm(total=len(dataloader),leave=False) as t:

        for batch, (X, y) in enumerate(dataloader):
            X = X.to(device=in_device)
            y = y.to(device=in_device)

            # Compute prediction and loss
            if method == 'encoder':
                pred = model(X.type(torch.long)).flatten()
            elif method == 'linmod':
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

        if board:
            board.add_scalar("train_loss", loss, epoch)

        torch.cuda.empty_cache()
        print(f"\nEpoch Done, max loss:{max_loss:.3e}, min loss: {min_loss:.3e}, final loss: {loss.item():.3e}")


def test_loop(dataloader, model, loss_fn, method, in_device, board=None, epoch=0):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    loss = torch.zeros(1).to(device=in_device)

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device=in_device)
            y = y.to(device=in_device)

            if method == 'encoder':
                pred = model(X.type(torch.long)).flatten()
            elif method == 'linmod':
                pred = model(X).flatten()

            loss += loss_fn(pred, y).item()



    torch.cuda.empty_cache()
    loss /= num_batches
    loss = loss.cpu().numpy()[0]
    if board:
        board.add_scalar("val_loss", loss, epoch)
    avg_loss = loss * num_batches / size

    print(f"Validation Error: {loss:.3e}\n")
    return loss