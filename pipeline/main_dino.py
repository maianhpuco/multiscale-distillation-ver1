import os
from src.utils import get_project_root
from types import SimpleNamespace
import torch
import timm
from time import time
from src.utils import * 
from src.multi_distill import * 
from src.dino import *
from PIL import Image 
from src.utils import get_project_root
from src.dino import *


PROJECT_DIR = get_project_root() 
print(f"Project dir: {PROJECT_DIR}") 

def get_args():
    return  SimpleNamespace(**{
        "batch_size": 32,
        "device": "cpu",
        "image_size": 512, 
        "patch_size": 64, 
        "logging_freq": 200,
        "momentum_teacher": 0.9995,
        "n_crops": 4,
        "n_epochs": 5,
        "out_dim": 1024,
        "tensorboard_dir": "logs",
        "clip_grad": 2.0,
        "norm_last_layer": False,
        "batch_size_eval": 64,
        "teacher_temp": 0.04,
        "student_temp": 0.1,
        "pretrained": True,
        "weight_decay": 0.4
    }) 
    
def initialize_models(args): 
    # Neural network related
    student_vit = timm.create_model(vit_name, pretrained=args.pretrained, img_size=args.image_size)
    teacher_vit = timm.create_model(vit_name, pretrained=args.pretrained)

    student = MultiCropWrapper(
        student_vit,
        Head(
            dim,
            args.out_dim,
            norm_last_layer=args.norm_last_layer,
        ),
    )
    teacher = MultiCropWrapper(
        teacher_vit,
        Head(dim, args.out_dim)
        )

    student, teacher = student.to(device), teacher.to(device)

    teacher.load_state_dict(student.state_dict())

    for p in teacher.parameters():
        p.requires_grad = False 
        
    return student, teacher, student_device, teacher_device 

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

def main(args, x) 
    student, teacher, student_device, teacher_device = initialize_models(args)
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
    
   
    # Perform forward pass with no gradient comput 