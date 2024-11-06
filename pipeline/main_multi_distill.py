import os
from src.utils import get_project_root
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F 
import timm
from time import time
from src.utils import * 
from src.multi_distill import * 
from src.dino import *
from PIL import Image 
from src.utils import get_project_root
from src.multi_distill import MultiscaleDistillaionCropWrapper, Loss
from src.dino import Head, vit_small 
PYTORCH_ENABLE_MPS_FALLBACK=1 

PROJECT_DIR = get_project_root() 
print(f"Project dir: {PROJECT_DIR}")

# Assuming MultiCropWrapper, Head, MultiscaledDistillationModel, and eval_transforms are defined elsewhere

def get_args():
    return SimpleNamespace(
        device="mps",
        batch_size=32,
        weight_decay=0.4,
        logging_freq=200,
        momentum_teacher=0.9995,
        n_epochs=5,
        out_dim=2,
        clip_grad=2.0,
        norm_last_layer=True,
        batch_size_eval=64,
        teacher_temp=0.04,
        student_temp=0.1,
    )

def initialize_models(args):
    student_device = torch.device(args.device)
    teacher_device = torch.device(args.device)
    student256_vit =  vit_small(patch_size=16, image_size=256)   
    teacher_vit =  vit_small(patch_size=16, image_size=256)    
    # student256_vit = timm.create_model('vit_small_patch16_224', img_size=256, pretrained=True, num_classes=args.out_dim)
    # teacher_vit = timm.create_model('vit_small_patch16_224', img_size=256, pretrained=True, num_classes=args.out_dim)
    
    student256 = MultiscaleDistillaionCropWrapper(
        student256_vit,
        Head(384, args.out_dim, norm_last_layer=args.norm_last_layer)
    )
    teacher = MultiscaleDistillaionCropWrapper(
        teacher_vit,
        Head(384, args.out_dim)
    )
    
    student256.to(student_device)
    teacher.to(teacher_device)
    teacher.load_state_dict(student256.state_dict())
    
    for p in teacher.parameters():
        p.requires_grad = False
    
    return student256, teacher, student_device, teacher_device

def initialize_loss(args, student_device):
    return DINOLoss(
        args.out_dim,
        teacher_temp=args.teacher_temp,
        student_temp=args.student_temp,
    ).to(student_device)

def initialize_optimizer(student256, args):
    lr = 0.0005 * args.batch_size / 256
    return torch.optim.AdamW(
        student256.parameters(),
        lr=lr,
        weight_decay=args.weight_decay,
    )

def main(args, x):
    student256, teacher, student_device, teacher_device = initialize_models(args)
    dino_loss = initialize_loss(args, student_device)
    optimizer = initialize_optimizer(student256, args)
    
    model = MultiscaledDistillationModel(
        student_model=student256,
        teacher_model=teacher,
        student_device=student_device,
        teacher_device=teacher_device,
    )
    model.eval()  # Set model to evaluation mode
    
    # Start time for monitoring execution
    start = time()
    
   
    # Perform forward pass with no gradient computation
    with torch.no_grad():
        student_logits, student_embeddings, teacher_logits, teacher_embeddings = model.forward(x)
        print("Student logits shape:", student_logits.shape)
        print("Teacher logits shape:", teacher_logits.shape)
        
        # Calculate loss
        loss = dino_loss(student_logits, teacher_logits)
        print("Loss:", loss.item())
    
    # EMA update for the teacher model
    with torch.no_grad():
        for student_ps, teacher_ps in zip(student256.parameters(), teacher.parameters()):
            teacher_ps.data.mul_(args.momentum_teacher)
            teacher_ps.data.add_((1 - args.momentum_teacher) * student_ps.detach().data)
    

if __name__ == "__main__":
    start = time()
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print (x)
    else:
        print ("MPS device not found.") 
    region = Image.open(f'{PROJECT_DIR}/data/image_samples/image_4k.png')
    # Load and preprocess image (Assuming `region` is defined or passed to main)
    input = eval_transforms()(region).unsqueeze(dim=0)  # Apply evaluation transformations and add batch dimension
 
    args = get_args()
    main(args, input)

    print("Execution time (minutes):", (time() - start) / 60)

