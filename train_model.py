# prompt: get the dataset for training a pytorch nn

import csv
import os
import time
from functools import partial

import numpy as np
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch import nn

from vit import ViT

from attn import (
    Attention,
    create_attention_class,
    phi_plus,
    phi_plus_plus,
    relu_kernel,
    exp_kernel,
    combined_hadamard_matrices,
    givens_rotation_g2,
    learnable_g,
    fixed_random_g
)

# Evaluation function
def evaluate_model(model, loader, criterion):
    model.eval()
    total_loss, total_correct = 0, 0
    inference_times = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            start_time = time.time()
            preds = model(x)
            end_time = time.time()
            inference_times.append(end_time - start_time)
            loss = criterion(preds, y)
            total_loss += loss.item()
            total_correct += (preds.argmax(1) == y).sum().item()
    avg_inference_time = np.mean(inference_times)
    accuracy = total_correct / len(loader.dataset)
    return total_loss / len(loader), accuracy, avg_inference_time


# Training function with speed measurement
def train_with_speed(model, loader, optimizer, criterion):
    model.train()
    total_loss, total_correct = 0, 0
    batch_times = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        start_time = time.time()
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        end_time = time.time()
        batch_times.append(end_time - start_time)
        total_loss += loss.item()
        total_correct += (preds.argmax(1) == y).sum().item()
    avg_batch_time = np.mean(batch_times)
    accuracy = total_correct / len(loader.dataset)
    return total_loss / len(loader), accuracy, avg_batch_time


# Evaluation framework
def evaluate_framework(model, model_name, train_loader, test_loader, epochs=50):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    print(f"\nEvaluating {model_name}...")
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []
    train_times, inference_times = [], []

    for epoch in range(epochs):
        train_loss, train_acc, train_time = train_with_speed(
            model, train_loader, optimizer, criterion
        )
        test_loss, test_acc, avg_inference_time = evaluate_model(
            model, test_loader, criterion
        )

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        train_times.append(train_time)
        inference_times.append(avg_inference_time)

        print(
            f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Train Time/Batch: {train_time:.4f}s, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, "
            f"Inference Time/Batch: {avg_inference_time:.4f}s"
        )

    # Calculate variance in accuracy
    accuracy_variance = np.var(test_accuracies)

    # Save the model
    # save_model(model, model_name)

    with open(
        f"traiin_logs/{model_name}_results.txt", "w"
    ) as f:
        csv.writer(f).writerow(train_losses)
        csv.writer(f).writerow(test_losses)
        csv.writer(f).writerow(train_accuracies)
        csv.writer(f).writerow(test_accuracies)
        csv.writer(f).writerow(train_times)
        csv.writer(f).writerow(inference_times)

    # Summary of results
    print(f"\nSummary for {model_name}:")
    print(f"Final Test Accuracy: {test_accuracies[-1]:.4f}")
    print(f"Accuracy Variance: {accuracy_variance:.4f}")
    print(f"Average Training Time per Batch: {np.mean(train_times):.4f}s")
    print(f"Average Inference Time per Batch: {np.mean(inference_times):.4f}s")


if __name__ == "__main__":
    std = (0.2023, 0.1994, 0.2010)
    mean = (0.4914, 0.4822, 0.4465)

    # Define a transform to normalize the data
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    # Download the training dataset
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    # Create a data loader for the training dataset
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=2
    )

    # Download the test dataset
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    # Create a data loader for the test dataset
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=256, shuffle=False, num_workers=2
    )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    

    """### Experiments

    #### Standard Transformer
    """

    model = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0,
        emb_dropout=0,
        channels=3,
        attention_class=Attention
    )

    evaluate_framework(model, 'Vanilla ViT', trainloader, testloader)

    """#### Axial Rope Attention"""

    attention_cl = create_attention_class(
        kernel_func=None,
        rope_theta=100.0,
        apply_rope=True,
        rope_mixed=False
    )

    model = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.,
        emb_dropout=0.,
        channels=3,
        attention_class=attention_cl
    )

    evaluate_framework(model, 'Axial Rope Attention', trainloader, testloader)

    """#### Mixed Rope Attention"""

    attention_cl = create_attention_class(
        kernel_func=None,
        rope_theta=10.0,
        apply_rope=True,
        rope_mixed=True
    )

    model = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.,
        emb_dropout=0.,
        channels=3,
        attention_class=attention_cl
    )

    evaluate_framework(model, 'Mixed Rope Attention', trainloader, testloader)

    """#### Performer Phi Plus"""

    kernel_func = partial(phi_plus, m=8)
    attention_cl = create_attention_class(
        kernel_func=kernel_func,
        rope_theta=10.0,
        apply_rope=False
    )

    model = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0,
        emb_dropout=0,
        channels=3,
        attention_class=attention_cl
    )

    evaluate_framework(model, 'Performer Phi Plus', trainloader, testloader)

    """#### Performer Phi Plus + Axial RoPE"""

    kernel_func = partial(phi_plus, m=8)
    attention_cl = create_attention_class(
        kernel_func=kernel_func,
        rope_theta=100.0,
        apply_rope=True,
        rope_mixed=False
    )

    model = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.,
        emb_dropout=0.,
        channels=3,
        attention_class=attention_cl
    )

    evaluate_framework(model, 'Performer Phi Plus + Axial RoPE', trainloader, testloader)

    """#### Performer Phi Plus + Mixed RoPE"""

    kernel_func = partial(phi_plus, m=8)
    attention_cl = create_attention_class(
        kernel_func=kernel_func,
        rope_theta=10.0,
        apply_rope=True,
        rope_mixed=True
    )

    model = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.,
        emb_dropout=0.,
        channels=3,
        attention_class=attention_cl
    )

    evaluate_framework(model, 'Performer Phi Plus + Mixed RoPE', trainloader, testloader)

    """#### Performer Phi Plus Plus"""

    kernel_func = partial(phi_plus_plus, m=8)
    attention_cl = create_attention_class(
        kernel_func=kernel_func,
        rope_theta=10.0,
        apply_rope=False
    )

    model = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.,
        emb_dropout=0.,
        channels=3,
        attention_class=attention_cl
    )

    evaluate_framework(model, 'Performer Phi Plus Plus', trainloader, testloader)

    """#### Performer Phi Plus Plus + Axial RoPE"""

    kernel_func = partial(phi_plus_plus, m=8)
    attention_cl = create_attention_class(
        kernel_func=kernel_func,
        rope_theta=100.0,
        apply_rope=True,
        rope_mixed=False
    )

    model = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.,
        emb_dropout=0.,
        channels=3,
        attention_class=attention_cl
    )

    evaluate_framework(model, 'Performer Phi Plus Plus + Axial RoPE', trainloader, testloader)

    """#### Performer Phi Plus Plus + Mixed RoPE"""

    kernel_func = partial(phi_plus_plus, m=8)
    attention_cl = create_attention_class(
        kernel_func=kernel_func,
        rope_theta=10.0,
        apply_rope=True,
        rope_mixed=True
    )

    model = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.,
        emb_dropout=0.,
        channels=3,
        attention_class=attention_cl
    )

    evaluate_framework(model, 'Performer Phi Plus Plus + Mixed RoPE', trainloader, testloader)

    """#### Φ(u) = ReLU(u)"""

    kernel_func = partial(relu_kernel)
    attention_cl = create_attention_class(
        kernel_func=kernel_func,
        rope_theta=10.0,
        apply_rope=False,
        rope_mixed=False
    )

    model = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.,
        emb_dropout=0.,
        channels=3,
        attention_class=attention_cl
    )

    evaluate_framework(model, 'f-ReLU', trainloader, testloader)

    """#### ReLU(u) + Axial RoPE"""

    kernel_func = partial(relu_kernel)
    attention_cl = create_attention_class(
        kernel_func=kernel_func,
        rope_theta=100.0,
        apply_rope=True,
        rope_mixed=False
    )

    model = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.,
        emb_dropout=0.,
        channels=3,
        attention_class=attention_cl
    )

    evaluate_framework(model, 'ReLU + Axial RoPE', trainloader, testloader)

    """#### ReLU(u) + Mixed RoPE"""

    kernel_func = partial(relu_kernel)
    attention_cl = create_attention_class(
        kernel_func=kernel_func,
        rope_theta=10.0,
        apply_rope=True,
        rope_mixed=True
    )

    model = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.,
        emb_dropout=0.,
        channels=3,
        attention_class=attention_cl
    )

    evaluate_framework(model, 'ReLU + Mixed RoPE', trainloader, testloader)

    """#### Φ(u) = exp(u)"""

    kernel_func = partial(exp_kernel)
    attention_cl = create_attention_class(
        kernel_func=kernel_func,
        rope_theta=10.0,
        apply_rope=False,
        rope_mixed=False
    )

    model = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.,
        emb_dropout=0.,
        channels=3,
        attention_class=attention_cl
    )

    evaluate_framework(model, 'f-exp', trainloader, testloader)

    """#### exp(u) + Axial RoPE"""

    kernel_func = partial(exp_kernel)
    attention_cl = create_attention_class(
        kernel_func=kernel_func,
        rope_theta=100.0,
        apply_rope=True,
        rope_mixed=False
    )

    model = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.,
        emb_dropout=0.,
        channels=3,
        attention_class=attention_cl
    )

    evaluate_framework(model, 'exp + Axial RoPE', trainloader, testloader)

    """#### exp(u) + Mixed RoPE"""

    kernel_func = partial(exp_kernel)
    attention_cl = create_attention_class(
        kernel_func=kernel_func,
        rope_theta=10.0,
        apply_rope=True,
        rope_mixed=True
    )

    model = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.,
        emb_dropout=0.,
        channels=3,
        attention_class=attention_cl
    )

    evaluate_framework(model, 'exp + Mixed RoPE', trainloader, testloader)

    """#### Learnable G Matrix"""

    g_func = partial(learnable_g, 512 // 8)
    attention_cl = create_attention_class(
        g_func=g_func,
        rope_theta=10.0,
        apply_rope=False,
        rope_mixed=False
    )

    model = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.,
        emb_dropout=0.,
        channels=3,
        attention_class=attention_cl
    )

    evaluate_framework(model, 'Learnable G Matrix', trainloader, testloader)

    """#### Learnable G Matrix + Axial RoPE"""

    g_func = partial(learnable_g, 512 // 8)
    attention_cl = create_attention_class(
        g_func=g_func,
        rope_theta=100.,
        apply_rope=True,
        rope_mixed=False
    )

    model = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.,
        emb_dropout=0.,
        channels=3,
        attention_class=attention_cl
    )

    evaluate_framework(model, 'Learnable G Matrix + Axial RoPE', trainloader, testloader)

    """#### Learnable G Matrix + Mixed RoPE"""

    g_func = partial(learnable_g, 512 // 8)
    attention_cl = create_attention_class(
        g_func=g_func,
        rope_theta=10.,
        apply_rope=True,
        rope_mixed=True
    )

    model = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.,
        emb_dropout=0.,
        channels=3,
        attention_class=attention_cl
    )

    evaluate_framework(model, 'Learnable G Matrix + Mixed RoPE', trainloader, testloader)

    """#### G-Hadamard"""

    g_func = partial(combined_hadamard_matrices, 512 // 8, num_blocks=3)
    attention_cl = create_attention_class(
        g_func=g_func,
        rope_theta=10.0,
        apply_rope=False,
        rope_mixed=False
    )

    model = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.,
        emb_dropout=0.,
        channels=3,
        attention_class=attention_cl
    )

    evaluate_framework(model, 'Hadamard G Matrix', trainloader, testloader)

    """#### G-Hadamard + Axial RoPE"""

    g_func = partial(combined_hadamard_matrices, 512 // 8, num_blocks=3)
    attention_cl = create_attention_class(
        g_func=g_func,
        rope_theta=100.,
        apply_rope=True,
        rope_mixed=False
    )

    model = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.,
        emb_dropout=0.,
        channels=3,
        attention_class=attention_cl
    )

    evaluate_framework(model, 'Hadamard G Matrix + Axial RoPE', trainloader, testloader)

    """#### G-Hadamard + Mixed RoPE"""

    g_func = partial(combined_hadamard_matrices, 512 // 8, num_blocks=3)
    attention_cl = create_attention_class(
        g_func=g_func,
        rope_theta=10.0,
        apply_rope=True,
        rope_mixed=True
    )

    model = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.,
        emb_dropout=0.,
        channels=3,
        attention_class=attention_cl
    )

    evaluate_framework(model, 'Hadamard G Matrix + Mixed RoPE', trainloader, testloader)

    """#### G-Givens Rotation Matrix"""

    g_func = partial(givens_rotation_g2, 512 // 8, 5)
    attention_cl = create_attention_class(
        g_func=g_func,
        rope_theta=10.0,
        apply_rope=False,
        rope_mixed=False
    )

    model = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.,
        emb_dropout=0.,
        channels=3,
        attention_class=attention_cl
    )

    evaluate_framework(model, 'Givens Rotation Matrix', trainloader, testloader)

    """#### G-Givens + Axial RoPE"""

    g_func = partial(givens_rotation_g2, 512 // 8, 5)
    attention_cl = create_attention_class(
        g_func=g_func,
        rope_theta=100.0,
        apply_rope=True,
        rope_mixed=False
    )

    model = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.,
        emb_dropout=0.,
        channels=3,
        attention_class=attention_cl
    )

    evaluate_framework(model, 'Givens Rotation Matrix + Axial RoPE', trainloader, testloader)

    """#### G-Givens + Mixed RoPE"""

    g_func = partial(givens_rotation_g2, 512 // 8, 5)
    attention_cl = create_attention_class(
        g_func=g_func,
        rope_theta=10.0,
        apply_rope=True,
        rope_mixed=True
    )

    model = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.,
        emb_dropout=0.,
        channels=3,
        attention_class=attention_cl
    )

    evaluate_framework(model, 'Givens Rotation Matrix + Mixed RoPE', trainloader, testloader)

    """#### Random Fixed G"""

    g_func = partial(fixed_random_g, 512 // 8)
    attention_cl = create_attention_class(
        g_func=g_func,
        rope_theta=10.0,
        apply_rope=False,
        rope_mixed=False
    )

    model = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.,
        emb_dropout=0.,
        channels=3,
        attention_class=attention_cl
    )

    evaluate_framework(model, 'Random Fixed G', trainloader, testloader)

    """#### Random Fixed G + Axial RoPE"""

    g_func = partial(fixed_random_g, 512 // 8)
    attention_cl = create_attention_class(
        g_func=g_func,
        rope_theta=100.0,
        apply_rope=True,
        rope_mixed=False
    )

    model = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.,
        emb_dropout=0.,
        channels=3,
        attention_class=attention_cl
    )

    evaluate_framework(model, 'Random Fixed G + Axial RoPE', trainloader, testloader)

    """#### Random Fixed G + Mixed RoPE"""

    g_func = partial(fixed_random_g, 512 // 8)
    attention_cl = create_attention_class(
        g_func=g_func,
        rope_theta=10.0,
        apply_rope=True,
        rope_mixed=True
    )

    model = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.,
        emb_dropout=0.,
        channels=3,
        attention_class=attention_cl
    )

    evaluate_framework(model, 'Random Fixed G + Mixed RoPE', trainloader, testloader)

