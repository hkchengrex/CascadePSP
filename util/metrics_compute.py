import torch.nn.functional as F

from util.util import compute_tensor_iu

def get_new_iou_hook(values, size):
    return 'iou/new_iou_%s'%size, values['iou/new_i_%s'%size]/values['iou/new_u_%s'%size]

def get_orig_iou_hook(values):
    return 'iou/orig_iou', values['iou/orig_i']/values['iou/orig_u']

def get_iou_gain(values, size):
    return 'iou/iou_gain_%s'%size, values['iou/new_iou_%s'%size] - values['iou/orig_iou']

iou_hooks_to_be_used = [
        get_orig_iou_hook,
        lambda x: get_new_iou_hook(x, '224'), lambda x: get_iou_gain(x, '224'),
        lambda x: get_new_iou_hook(x, '56'), lambda x: get_iou_gain(x, '56'),
        lambda x: get_new_iou_hook(x, '28'), lambda x: get_iou_gain(x, '28'),
        lambda x: get_new_iou_hook(x, '28_2'), lambda x: get_iou_gain(x, '28_2'),
        lambda x: get_new_iou_hook(x, '28_3'), lambda x: get_iou_gain(x, '28_3'),
        lambda x: get_new_iou_hook(x, '56_2'), lambda x: get_iou_gain(x, '56_2'),
    ]

iou_hooks_final_only = [
    get_orig_iou_hook,
    lambda x: get_new_iou_hook(x, '224'), lambda x: get_iou_gain(x, '224'),
]

# Compute common loss and metric for generator only
def compute_loss_and_metrics(images, para, detailed=True, need_loss=True, has_lower_res=True):

    """
    This part compute loss and metrics for the generator
    """

    loss_and_metrics = {}

    gt = images['gt']
    seg = images['seg']

    pred_224 = images['pred_224']
    if has_lower_res:
        pred_28 = images['pred_28']
        pred_56 = images['pred_56']
        pred_28_2 = images['pred_28_2']
        pred_28_3 = images['pred_28_3']
        pred_56_2 = images['pred_56_2']

    if need_loss:
        # Loss weights
        ce_weights = para['ce_weight']
        l1_weights = para['l1_weight']
        l2_weights = para['l2_weight']

        # temp holder for losses at different scale
        ce_loss = [0] * 6
        l1_loss = [0] * 6
        l2_loss = [0] * 6
        loss = [0] * 6

        ce_loss[0] = F.binary_cross_entropy_with_logits(images['out_224'], (gt>0.5).float())
        if has_lower_res:
            ce_loss[1] = F.binary_cross_entropy_with_logits(images['out_28'], (gt>0.5).float())
            ce_loss[2] = F.binary_cross_entropy_with_logits(images['out_56'], (gt>0.5).float())
            ce_loss[3] = F.binary_cross_entropy_with_logits(images['out_28_2'], (gt>0.5).float())
            ce_loss[4] = F.binary_cross_entropy_with_logits(images['out_28_3'], (gt>0.5).float())
            ce_loss[5] = F.binary_cross_entropy_with_logits(images['out_56_2'], (gt>0.5).float())

        l1_loss[0] = F.l1_loss(pred_224, gt)
        if has_lower_res:
            l2_loss[0] = F.mse_loss(pred_224, gt)
            l1_loss[1] = F.l1_loss(pred_28, gt)
            l2_loss[1] = F.mse_loss(pred_28, gt)
            l1_loss[2] = F.l1_loss(pred_56, gt)
            l2_loss[2] = F.mse_loss(pred_56, gt)

        if has_lower_res:
            l1_loss[3] = F.l1_loss(pred_28_2, gt)
            l2_loss[3] = F.mse_loss(pred_28_2, gt)
            l1_loss[4] = F.l1_loss(pred_28_3, gt)
            l2_loss[4] = F.mse_loss(pred_28_3, gt)
            l1_loss[5] = F.l1_loss(pred_56_2, gt)
            l2_loss[5] = F.mse_loss(pred_56_2, gt)

        loss_and_metrics['grad_loss'] = F.l1_loss(images['gt_sobel'], images['pred_sobel'])

        # Weighted loss for different levels
        for i in range(6):
            loss[i] = ce_loss[i] * ce_weights[i] + \
                    l1_loss[i] * l1_weights[i] + \
                    l2_loss[i] * l2_weights[i]
        
        loss[0] += loss_and_metrics['grad_loss'] * para['grad_weight']

    """
    Compute IOU stats
    """
    orig_total_i, orig_total_u = compute_tensor_iu(seg>0.5, gt>0.5)
    loss_and_metrics['iou/orig_i'] = orig_total_i
    loss_and_metrics['iou/orig_u'] = orig_total_u

    new_total_i, new_total_u = compute_tensor_iu(pred_224>0.5, gt>0.5)
    loss_and_metrics['iou/new_i_224'] = new_total_i
    loss_and_metrics['iou/new_u_224'] = new_total_u

    if has_lower_res:
        new_total_i, new_total_u = compute_tensor_iu(pred_56>0.5, gt>0.5)
        loss_and_metrics['iou/new_i_56'] = new_total_i
        loss_and_metrics['iou/new_u_56'] = new_total_u

        new_total_i, new_total_u = compute_tensor_iu(pred_28>0.5, gt>0.5)
        loss_and_metrics['iou/new_i_28'] = new_total_i
        loss_and_metrics['iou/new_u_28'] = new_total_u

        new_total_i, new_total_u = compute_tensor_iu(pred_28_2>0.5, gt>0.5)
        loss_and_metrics['iou/new_i_28_2'] = new_total_i
        loss_and_metrics['iou/new_u_28_2'] = new_total_u

        new_total_i, new_total_u = compute_tensor_iu(pred_28_3>0.5, gt>0.5)
        loss_and_metrics['iou/new_i_28_3'] = new_total_i
        loss_and_metrics['iou/new_u_28_3'] = new_total_u

        new_total_i, new_total_u = compute_tensor_iu(pred_56_2>0.5, gt>0.5)
        loss_and_metrics['iou/new_i_56_2'] = new_total_i
        loss_and_metrics['iou/new_u_56_2'] = new_total_u
        
    """
    All done.
    Now gather everything in a dict for logging
    """

    if need_loss:
        loss_and_metrics['total_loss'] = 0
        for i in range(6):
            loss_and_metrics['ce_loss/s_%d'%i] = ce_loss[i]
            loss_and_metrics['l1_loss/s_%d'%i] = l1_loss[i]
            loss_and_metrics['l2_loss/s_%d'%i] = l2_loss[i]
            loss_and_metrics['loss/s_%d'%i] = loss[i]

            loss_and_metrics['total_loss'] += loss[i]

    return loss_and_metrics

