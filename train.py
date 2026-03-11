import torch
from torch.utils.data import DataLoader
from utils.dataset_loader import DeepfakeDataset
from model import FrequencyCNN
from tqdm import tqdm

# load dataset
dataset = DeepfakeDataset("dataset")

train_loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)

# device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
model = FrequencyCNN().to(device)

# loss
criterion = torch.nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.00001
)

epochs = 40

for epoch in range(epochs):

    model.train()

    total_loss = 0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{epochs}",
        leave=True
    )

    for images, labels in progress_bar:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        # show running loss in progress bar
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch+1} Average Loss {avg_loss}")

# save model
torch.save(model.state_dict(), "models/deepfake_model.pth")