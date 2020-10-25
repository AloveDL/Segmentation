import torch
import numpy as np
# import multiresolutionimageinterface as mir
import torch
from torch.autograd import Function


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
        # print(s)
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        # print("c[0],c[1]", c[0], c[1])
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

def get_jaccard(output, target):
    if output.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
        # print(s)
    else:
        s = torch.FloatTensor(1).zero_()
    eps = 0.0001
    for i, c in enumerate(zip(output, target)):
        # print("c[0],c[1]", c[0], c[1])

        numpy1 = c[0].cpu().numpy()
        numpy2 = c[1].cpu().numpy()
        inter = (numpy1*numpy2).sum()+eps
        union = numpy1.sum() + numpy2.sum() + eps - inter
        if union<=0:
            continue
        # print(inter)
        # print("x",union)
        s = s + (inter/union)
    return s / (i + 1)


def pixel_accurary(output,target):
    if output.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
        # print(s)
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(output, target)):
        # print("c[0],c[1]", c[0], c[1])
        total = c[0].size(0)*c[0].size(1)*c[0].size(2)
        numpy_1 = c[0].cpu().numpy()
        numpy_2 = c[1].cpu().numpy()
        corrent = (numpy_1==numpy_2).sum()
        s = s + corrent/total

    return s / (i + 1)

def get_recall(output,target):

    if output.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
        # print(s)
    else:
        s = torch.FloatTensor(1).zero_()
    for i, c in enumerate(zip(output, target)):

        numpy1 = c[0].cpu().numpy()
        numpy2 = c[1].cpu().numpy()
        seg_inv, gt_inv = np.logical_not(numpy1), np.logical_not(numpy2)
        true_pos = float(np.logical_and(numpy1, numpy2).sum())  # float for division
        true_neg = np.logical_and(seg_inv, gt_inv).sum()
        false_pos = np.logical_and(numpy1, gt_inv).sum()
        false_neg = np.logical_and(seg_inv, numpy2).sum()
        prec = true_pos / (true_pos + false_pos + 1e-6)
        rec = true_pos / (true_pos + false_neg + 1e-6)
        s = s+rec
    return s / (i + 1)


def get_precision(output, target):

    if output.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
        # print(s)
    else:
        s = torch.FloatTensor(1).zero_()
    for i, c in enumerate(zip(output, target)):

        numpy1 = c[0].cpu().numpy()
        numpy2 = c[1].cpu().numpy()
        seg_inv, gt_inv = np.logical_not(numpy1), np.logical_not(numpy2)
        true_pos = float(np.logical_and(numpy1, numpy2).sum())  # float for division
        true_neg = np.logical_and(seg_inv, gt_inv).sum()
        false_pos = np.logical_and(numpy1, gt_inv).sum()
        false_neg = np.logical_and(seg_inv, numpy2).sum()
        prec = true_pos / (true_pos + false_pos + 1e-6)
        rec = true_pos / (true_pos + false_neg + 1e-6)
        s = s+prec
    return s / (i + 1)

def get_mask_img():

    reader = mir.MultiResolutionImageReader()
    mr_image = reader.open('ACDC_challenge/Images/13111-1.tif')
    annotation_list = mir.AnnotationList()
    xml_repository = mir.XmlRepository(annotation_list)
    xml_repository.setSource('ACDC_challenge/Annotations/13111-1.xml')
    xml_repository.load()
    annotation_mask = mir.AnnotationToMask()
    output_path = 'ACDC_challenge/Images/13111-1_tumor_annotations.tif'
    annotation_mask.convert(annotation_list, output_path, mr_image.getDimensions(), mr_image.getSpacing())


def get_tif():

    reader = mir.MultiResolutionImageReader()
    mr_image = reader.open('ACDC_challenge/Images/13111-1.tif')
    level = 2
    ds = mr_image.getLevelDownsample(level)
    image_patch = mr_image.getUCharPatch(int(568 * ds), int(732 * ds), 300, 200, level)