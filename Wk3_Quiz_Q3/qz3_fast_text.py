import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data.dataloader import DataLoader
from qz3_data_stuff import TwitterCovidDataset
from torch.utils.data.dataset import random_split
import time
from torch import nn
import torch

# using tensorboard for visualizing results
# do 
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text[0])

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
         label_list.append(_label)
         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
         text_list.append(processed_text)
         offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)

class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        
        # log in tensorboard
        writer.add_scalar("Loss/train", loss)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        
        writer.add_scalar("accuracy/train", total_acc/total_count)
        
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count



if __name__=="__main__":

    train_iter = DataLoader(TwitterCovidDataset("Data/Corona_NLP_train.csv"), batch_size=1,
                            shuffle=False, num_workers=0)

    vocab = build_vocab_from_iterator(yield_tokens(train_iter))

    text_pipeline = lambda x: vocab.lookup_indices(tokenizer(x))
    label_pipeline = lambda x: int(x) - 1

    dataloader = DataLoader(TwitterCovidDataset("Data/Corona_NLP_train.csv"),
                            batch_size=8, shuffle=False, collate_fn=collate_batch)


    num_class = len(set([label for (label, text) in train_iter]))
    vocab_size = len(vocab)
    emsize = 128
    model = TextClassificationModel(vocab_size, emsize, num_class).to(device)

    # Hyperparameters
    EPOCHS = 20 # epoch
    LR = 6.0  # learning rate
    BATCH_SIZE = 128 # batch size for training

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    total_accu = None

    train_dataloader = DataLoader(TwitterCovidDataset("Data/Corona_NLP_train.csv"),
                                batch_size=BATCH_SIZE,
                                shuffle=True, 
                                num_workers=0,
                                collate_fn=collate_batch)

    test_dataloader = DataLoader(TwitterCovidDataset("Data/Corona_NLP_test.csv"),
                                batch_size=BATCH_SIZE,
                                shuffle=True, 
                                num_workers=0,
                                collate_fn=collate_batch)


    # writer.add_graph(model)

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        
        train(train_dataloader)
        writer.flush()
        
        accu_val = evaluate(train_dataloader)
        writer.add_scalar("Acc/val", accu_val, epoch)
        
        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val
        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
            'valid accuracy {:8.3f} '.format(epoch,
                                            time.time() - epoch_start_time,
                                            accu_val))
        print('-' * 59)
        
    
    writer.close()