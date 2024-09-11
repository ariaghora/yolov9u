import torch

model_info = torch.load("./yolov9-e-converted.pt", weights_only=False)
print(model_info["model"].yaml_file)
model: torch.nn.Module = model_info["model"].to("cpu")

torch.save(model.state_dict(), "./yolov9-e.pt")
