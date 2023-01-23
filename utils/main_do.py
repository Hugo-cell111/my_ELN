from dataset.common import *
from dataset.transforms import *
from dataset.dataloader import dataloader
from utils.losses import *
from utils.visualize import *
import torch, wandb


def init(
    batch_size_labeled, batch_size_unlabeled, split, sets_id, data_set, shuffle=True
):
    if data_set == "voc":
        base = base_voc
        workers = 2

        transform_train = Compose(
            [
                ToTensor(),
                RandomResize(min_size=(505, 505)),
                RandomHorizontalFlip(flip_prob=0.5),
                Normalize(),
            ]
        )

        transform_unlabel = Compose(
            [
                ToTensor(),
                RandomResize(min_size=(505, 505)),
                RandomHorizontalFlip(flip_prob=0.5),
            ]
        )
        transform_test = Compose(
            [ToTensor(), RandomResize(min_size=(505, 505)), Normalize()]
        )

    elif data_set == "city":
        base = base_city
        workers = 2
        transform_train = Compose(
            [
                # ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                ToTensor(),
                RandomCrop(size=(512, 1024)),
                RandomHorizontalFlip(flip_prob=0.5),
                Normalize(),
            ]
        )

        transform_unlabel = Compose(
            [
                ToTensor(),
                RandomCrop(size=(512, 1024)),
                RandomHorizontalFlip(flip_prob=0.5),
            ]
        )
        transform_test = Compose([ToTensor(), Normalize()])

    test_set = dataloader(
        root=base,
        image_set="val",
        transforms=transform_test,
        label_state=True,\
        data_set=data_set,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=4,
        num_workers=workers,
        shuffle=False,
        pin_memory=True,
    )

    labeled_set = dataloader(
        root=base,
        image_set=(str(split) + "_labeled_" + str(sets_id)),
        transforms=transform_train,
        label_state=True,
        data_set=data_set,
    )
    labeled_sampler = torch.utils.data.DistributedSampler(labeled_set)
    labeled_loader = torch.utils.data.DataLoader(
        dataset=labeled_set,
        sampler=labeled_sampler,
        batch_size=batch_size_labeled,
        num_workers=workers,
        pin_memory=True,
    )

    unlabeled_set = dataloader(
        root=base,
        data_set=data_set,
        mask_type=".png",
        image_set=(str(split) + "_unlabeled_" + str(sets_id)),
        transforms=transform_unlabel,
        label_state=False,
    )
    unlabeled_sampler = torch.utils.data.DistributedSampler(unlabeled_set)
    unlabeled_loader = torch.utils.data.DataLoader(
        dataset=unlabeled_set,
        sampler=unlabeled_sampler,
        batch_size=batch_size_unlabeled,
        num_workers=workers,
        pin_memory=True
    )

    return labeled_loader, unlabeled_loader, val_loader


def typical_segtrain(self, inputs_l, labels_l):
    x4, x1 = self.enc(inputs_l) #low-level and high-level features
    cls_output, _ = self.dec(x4, x1)
    # prediction ¡ú log_softmax ¡ú nn.NLLLoss
    # 255 means ignore_index
    # pixels whose value are 255 in gt labels don't need to be calculated
    # which is a common setting in dataset such as cityscapes
    sup_loss = CEloss(cls_output, labels_l, 255)
    return sup_loss


def get_cls_output_and_featuer_vec(self, inputs_ul):

    a, b = self.enc(inputs_ul)
    cls_output_unsup, feature_vec = self.dec(a, b)

    return cls_output_unsup, feature_vec


def calculate_pseudo_loss(cls_output, final_candid, final_indices):
    pseudo_label = final_indices
    pseudo_label = torch.where(final_candid == 1, pseudo_label, 255)
    CE_pseudo_loss = CEloss(cls_output, pseudo_label, 255)  
    # this function is a little different from the formula(9)
    # If writing codes in a traditional way, B should be pixel-wise multiplied with Y
    # but here pixels whose values are 1 in B are set to 255 in pseudo_label
    # when calculating the CELoss, these pixels are naturally ignored by the "ignore_index=255"
    # I think this avoid mountainous computational of element-wise matrix calculation
    return CE_pseudo_loss, pseudo_label


def typical_submodule_segtrain_loss_constrain(self, inputs_l, labels_l, sup_loss):
    losses_list = 0

    constrain_candid = [20, 50, 100]  # loss constrain x20, x50, x100

    with torch.no_grad():
        a, b = self.enc(inputs_l)

    for idx, sd in enumerate(self.dec_list):  #
        cls_output, _ = sd(a.detach(), b.detach())
        sub_sup_loss = CEloss(cls_output, labels_l)

        if sub_sup_loss >= sup_loss * constrain_candid[idx]:
            losses_list += sub_sup_loss

        wandb.log(
            {
                "numeric_metric/sup_loss_"
                + str(constrain_candid[idx]): sub_sup_loss.item()
            },
            commit=False,
        )

    return losses_list  # / len(self.dec_list)


def train_ELN(self, inputs_l, labels_l):
    losses_list = 0
    constrain_candid = [20, 50]

    with torch.no_grad():
        a, b = self.enc(inputs_l)
        cls_output, _ = self.dec(a, b)

    for idx, sd in enumerate(self.dec_list):
        with torch.no_grad():
            cls_output_aux, _ = sd(a, b)

        eln = corrector_loss(self, inputs_l, labels_l, cls_output_aux)
        losses_list += eln
        wandb.log(
            {
                "numeric_metric/eln_" + str(constrain_candid[idx]): eln.item(),
            },
            commit=False,
        )

    eln_1 = corrector_loss(self, inputs_l, labels_l, cls_output)
    losses_list += eln_1

    wandb.log(
        {
            "numeric_metric/eln_100": eln_1.item(),
        }
    )

    return losses_list


def corrector_loss(self, inputs_l, labels_l, cls_output):
    # inputs_l.shape: [bs, channel, h, w]
    # labels_l.shape:[bs, h, w]
    # cls_output.shape:[bs, num_classes, h/4, w/4]
    correction_map_1 = self.eln(inputs_l, cls_output)  
    # correction_map_1.shape:[bs, 1, h/4, w/4]
    correction_map_1 = F.interpolate(
        correction_map_1, size=labels_l.shape[1:], mode="bilinear", align_corners=True
    ) 
    # correction_map_1.shape:[bs, 1, h, w]
    cls_output = F.interpolate(
        cls_output, size=inputs_l.shape[2:], mode="bilinear", align_corners=True
    )
    # cls_output.shape: [bs, num_classes, h, w]
    label_for_correction_1 = torch.where(
        cls_output.argmax(1) == labels_l, 1.0, 0.0
    ).cuda() # to judge which pixels are classified correctly
    # label_for_correction_1.shape: [bs, h, w]
    correction_loss_sup1 = nn.BCEWithLogitsLoss(reduction="none")(
        correction_map_1, label_for_correction_1.unsqueeze(1)
    )
    # correction_loss_sup1.shape: [bs, 1, h, w]
    # label_for_correction_1.unsqueeze(1).shape: [bs, 1, h, w]
    zero_loss = correction_loss_sup1.squeeze(1)[label_for_correction_1 == 0].numel()
    one_loss = correction_loss_sup1.squeeze(1)[label_for_correction_1 == 1].numel()
    mul_fact = one_loss / zero_loss
    # correction_loss_sup1.unsqueeze(1).shape: [bs, 1, 1, h, w]
    correction_loss_sup1.squeeze(1)[label_for_correction_1 == 0] *= mul_fact
    return correction_loss_sup1.mean()
