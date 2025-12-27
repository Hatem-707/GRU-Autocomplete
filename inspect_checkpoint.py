import torch

# Load checkpoint
checkpoint = torch.load("best_model.pt", map_location="cpu")

# Print available keys
print("Checkpoint keys:", checkpoint.keys())

# Check hyperparameters if available
if "hyper_parameters" in checkpoint:
    print("\nHyperparameters:")
    for k, v in checkpoint["hyper_parameters"].items():
        print(f"  {k}: {v}")

# Check state_dict keys to understand architecture
if "state_dict" in checkpoint:
    print("\nState dict keys (first 20):")
    for k in list(checkpoint["state_dict"].keys())[:20]:
        v = checkpoint["state_dict"][k]
        print(f"  {k}: {v.shape}")

    # Get embedding and fc layer shapes
    print("\nModel Architecture Details:")
    for k, v in checkpoint["state_dict"].items():
        if "embedding" in k and "weight" in k:
            print(f"Embedding shape: {v.shape}")
        if k == "fc.weight":
            print(f"FC weight shape: {v.shape}")
