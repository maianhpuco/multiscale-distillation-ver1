import torch 
import torch.nn as nn  
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
            teacher_device=torch.device('cpu'),               # Match the output dim of teacher model
        ):
        super().__init__()
        self.student        = student_model.to(student_device)
        self.teacher        = teacher_model.to(teacher_device)
        self.student_device = student_device
        self.teacher_device = teacher_device
        self.image_size     = image_size
        self.patch_size     = patch_size
         
    def forward(self, x):
        
        batch_256, w_256, h_256 = self.prepare_img_tensor(x)                    # 1. [1 x 3 x W x H]
        batch_256 = batch_256.unfold(2, self.patch_size , self.patch_size ).unfold(3, self.patch_size , self.patch_size)           # 2. [1 x 3 x w_256 x h_256 x 256 x 256]
        batch_256 = rearrange(batch_256, 'b c p1 p2 w h -> (b p1 p2) c w h')    # 2. [B x 3 x 256 x 256], where B = (1*w_256*h_256)


        for mini_bs in range(0, batch_256.shape[0], self.patch_size):
            minibatch_256 = batch_256[mini_bs:mini_bs+self.patch_size].to(self.student_device, non_blocking=True)
            student_logits, student_embeddings = self.student(minibatch_256)

        self.teacher.eval() 

        resize_transform = transforms.Resize((self.patch_size, self.patch_size))
        x_teacher = resize_transform(x.squeeze(0)).unsqueeze(0).to(self.teacher_device)
        teacher_logits, teacher_embeddings = self.teacher(x_teacher)
        
        return student_logits, student_embeddings, teacher_logits, teacher_embeddings

    def prepare_img_tensor(self, img: torch.Tensor):
        patch_size = self.patch_size
        make_divisble = lambda l, patch_size: (l - (l % patch_size))
        b, c, w, h = img.shape
        load_size = make_divisble(w, patch_size), make_divisble(h, patch_size)
        w_256, h_256 = w // patch_size, h // patch_size
        img_new = transforms.CenterCrop(load_size)(img)

        return img_new, w_256, h_256

