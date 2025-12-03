import mlflow
import torch
from models.pytorch_model import SimpleMLP

mlflow.set_experiment("agentic_civic_ai")

with mlflow.start_run():
  
    input_dim = 10
    hidden_dim = 20
    output_dim = 1
    lr = 0.001
    
    mlflow.log_params({"input_dim": input_dim, "hidden_dim": hidden_dim, "output_dim": output_dim, "lr": lr})
    
    # Dummy training
    model = SimpleMLP(input_dim, hidden_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    
    x = torch.randn(100, input_dim)
    y = torch.randn(100, output_dim)
    
    for epoch in range(5):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch} Loss: {loss.item()}")
        mlflow.log_metric("loss", loss.item(), step=epoch)
    
    mlflow.pytorch.log_model(model, "model")
