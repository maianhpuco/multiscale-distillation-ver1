import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOLoss(nn.Module):
    def __init__(
            self,
            out_dim,
            teacher_temp=0.04,  # T is higher -> sharpen -> prevent uniform distribution
            student_temp=0.1,  # Prevent student from mode collapsing
            center_momentum=0.9
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
        # Apply temperature scaling
        student_output = student_output / self.student_temp
        teacher_output = (teacher_output - self.center) / self.teacher_temp

        # Compute softmax for teacher and log-softmax for student
        student_sm = F.log_softmax(student_output, dim=-1)
        teacher_sm = F.softmax(teacher_output, dim=-1).detach()  # Stop gradients from teacher

        # Compute loss by broadcasting teacher_sm to match student_sm shape
        loss = torch.sum(-teacher_sm * student_sm, dim=-1)  # (256,)


        total_loss = loss.mean()  # Average over all samples

        # Update center based on teacher output
        self.update_center(teacher_output)

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """Update center used for teacher output with exponential moving average.

        Parameters
        ----------
        teacher_output : torch.Tensor
            Tensor of shape `(1, out_dim)`, representing the single teacher output.
        """
        # Calculate the batch mean of the teacher output
        batch_center = teacher_output.mean(dim=0, keepdim=True)  # Shape: (1, out_dim)

        # Update the center with exponential moving average
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
