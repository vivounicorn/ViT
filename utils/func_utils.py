import torch
import torch.nn.functional as F


def soft_distillation(teacher_logits, student_logits, temperature, balancing, labels, num_classes):
    loss_function = torch.nn.CrossEntropyLoss()
    kl_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1).detach(),
        reduction='batchmean')

    kl_loss *= (balancing * temperature ** 2)

    student_loss = loss_function(student_logits.view(-1, num_classes), labels.view(-1))
    return (1 - balancing) * student_loss + kl_loss


def hard_distillation(teacher_logits, student_logits, balancing, labels, num_classes):
    loss_function = torch.nn.CrossEntropyLoss()
    teacher_labels = teacher_logits.argmax(dim=-1)
    teacher_loss = loss_function(teacher_logits.view(-1, num_classes), teacher_labels.view(-1))
    student_loss = loss_function(student_logits.view(-1, num_classes), labels.view(-1))

    return (1 - balancing) * student_loss + balancing * teacher_loss
