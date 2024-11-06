import torch 
import torch.nn as nn  
import torch.nn.functional as F  
import torchvision.transforms as transforms 
from einops import rearrange, repeat
 
class MultiscaleDistillaionCropWrapper(nn.Module):
    """Convenience class for forward pass of multiple crops.

    Parameters
    ----------
    backbone : timm.models.vision_transformer.VisionTransformer
        Instantiated Vision Transformer. Note that we will take the `head`
        attribute and replace it with `nn.Identity`.

    new_head : Head
        New head that is going to be put on top of the `backbone`.
    """
    def __init__(self, backbone, new_head):
        super().__init__()
        backbone.head = nn.Identity()  # deactivate original head
        self.backbone = backbone
        self.new_head = new_head

    def forward(self, x):
        """Run the forward pass.

        The different crops are concatenated along the batch dimension
        and then a single forward pass is fun. The resulting tensor
        is then chunked back to per crop tensors.

        Parameters
        ----------
        x : list
            List of `torch.Tensor` each of shape `(n_samples, 3, size, size)`.

        Returns
        -------
        tuple
            Tuple of `torch.Tensor` each of shape `(n_samples, out_dim)` where
            `output_dim` is determined by `Head`.
        """
        # x.shape (batch_size , 3, size, size)
        cls_embedding = self.backbone(x)  # (n_samples * n_crops, in_dim)
        # print(cls_embedding.shape)
        logits = self.new_head(cls_embedding)  # (n_samples * n_crops, out_dim)

        return logits, cls_embedding


class MultiscaledDistillationModel(nn.Module):
    def __init__(
            self,
            student_model=None,
            teacher_model=None,
            image_size=4096, # test with 516 
            patch_size=256,  # test with 32
            student_device=torch.device('cpu'),
            teacher_device=torch.device('cpu'),      
            args = None# Match the output dim of teacher model
        ):
        super().__init__()
        self.student        = student_model.to(student_device)
        self.teacher        = teacher_model.to(teacher_device)
        self.student_device = student_device
        self.teacher_device = teacher_device
        self.image_size     = image_size
        self.patch_size     = patch_size
        self.args           = args 
         
    def forward(self, x):
        
        batch_256, w_256, h_256 = self.prepare_img_tensor(x)                    # 1. [1 x 3 x W x H]
        batch_256 = batch_256.unfold(2, self.patch_size , self.patch_size ).unfold(3, self.patch_size , self.patch_size)           # 2. [1 x 3 x w_256 x h_256 x 256 x 256]
        batch_256 = rearrange(batch_256, 'b c p1 p2 w h -> (b p1 p2) c w h')    # 2. [B x 3 x 256 x 256], where B = (1*w_256*h_256)


        for mini_bs in range(0, batch_256.shape[0], self.patch_size):
            minibatch_256 = batch_256[mini_bs:mini_bs+self.patch_size].to(self.student_device, non_blocking=True)
            student_logits, student_embeddings = self.student(minibatch_256)

        self.teacher.eval() 

        resize_transform = transforms.Resize((self.patch_size, self.patch_size))
        print("---shape of x", x.shape)
        if x.shape[0] == 1:
            
            x_teacher = resize_transform(x.squeeze(0)).to(self.teacher_device)
            x_teacher = x_teacher.unsqueeze(0) 
            print("x_teacher.shape", x_teacher.shape)
        else: 
            x_teacher = resize_transform(x.squeeze(0)).to(self.teacher_device) 
        teacher_logits, teacher_embeddings = self.teacher(x_teacher)        
        return student_logits, student_embeddings, teacher_logits, teacher_embeddings
    
    def update_moving_average(self):
        with torch.no_grad():
            for student_ps, teacher_ps in zip(self.student.parameters(), self.teacher.parameters()):
                teacher_ps.data.mul_(self.args.momentum_teacher)
                teacher_ps.data.add_((1 - self.args.momentum_teacher) * student_ps.detach().data)  
    
    def prepare_img_tensor(self, img: torch.Tensor):
        patch_size = self.patch_size
        make_divisble = lambda l, patch_size: (l - (l % patch_size))
        b, c, w, h = img.shape
        load_size = make_divisble(w, patch_size), make_divisble(h, patch_size)
        w_256, h_256 = w // patch_size, h // patch_size
        img_new = transforms.CenterCrop(load_size)(img)

        return img_new, w_256, h_256

class Loss(nn.Module):
    def __init__(
            self,
            out_dim,
            teacher_temp = 0.04, # T is higher -> sharpen -> make sure student does not predict the result that lead to the uniform distribution
            student_temp = 0.1, # prevent student from mode collapsing
            center_momentum = 0.9
    ):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(
            self,
            student_output,
            teacher_output,
    ):
        student_temp = [s / self.student_temp for s in student_output]
        teacher_temp = [(t - self.center) / self.teacher_temp for t in teacher_output]

        student_sm = [F.log_softmax(s, dim=-1) for s in student_temp]
        teacher_sm = [F.softmax(t, dim=-1).detach() for t in teacher_temp]

        total_loss = 0
        n_loss_terms = 0

        for t_ix, t in enumerate(teacher_sm):
            for s_ix, s in enumerate(student_sm):
                if t_ix == s_ix:
                    continue

                loss = torch.sum(-t * s, dim=-1)  # (n_samples,)
                total_loss += loss.mean()  # scalar
                n_loss_terms += 1

        total_loss /= n_loss_terms
        self.update_center(teacher_output)

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """Update center used for teacher output.

        Compute the exponential moving average.

        Parameters
        ----------
        teacher_output : tuple
            Tuple of tensors of shape `(n_samples, out_dim)` where each
            tensor represents a different crop.
        """
        teacher_output = (teacher_output)
        batch_center = torch.cat(teacher_output).mean(
            dim=0, keepdim=True
        )  # (1, out_dim)
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )
 
 
 
  