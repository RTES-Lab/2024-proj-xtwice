# trainer.py

def train(model, train_loader, optimizer, device, criterion, batch_size):
    model.train()
    train_loss = 0
    correct = 0
    for _, (data, label) in enumerate(train_loader):
        data = data.to(device)
        data = data.unsqueeze(1)
        label = label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        prediction = output.max(1, keepdim = True)[1]
        correct += prediction.eq(label.view_as(prediction)).sum().item()
        prediction =  prediction.view_as(label)

    train_loss /= (len(train_loader.dataset) / batch_size)
    train_accuracy = correct / len(train_loader.dataset)

    return train_loss, train_accuracy