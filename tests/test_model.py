import torch
from train import NeuralNet  # Import your model

def test_model_output_shape():
    model = NeuralNet()
    sample_input = torch.randn(1, 1, 28, 28)  # 1 sample, 1 channel, 28x28
    output = model(sample_input)
    assert output.shape == (1, 10), "Output shape incorrect!"

def test_model_training():
    model = NeuralNet()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    sample_input = torch.randn(5, 1, 28, 28)
    sample_labels = torch.tensor([1, 2, 3, 4, 5])  # Dummy labels

    initial_loss = criterion(model(sample_input), sample_labels)
    optimizer.zero_grad()
    initial_loss.backward()
    optimizer.step()
    
    new_loss = criterion(model(sample_input), sample_labels)
    
    assert new_loss < initial_loss, "Loss did not decrease!"
