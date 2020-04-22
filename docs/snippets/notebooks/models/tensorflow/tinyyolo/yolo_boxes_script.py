# This will be loaded as script hence no imports
# import torch


def nms(boxes):
    # assuming only one image in one batch
    boxes = boxes.squeeze()
    nms_thresh = 0.45
    conf_thresh = 0.2
    no_of_valid_elems = (boxes[:, 4] > conf_thresh).nonzero().numel()
    boxes_confs_inv = 1 - boxes[:, 4]
    _, sort_ids = torch.sort(boxes_confs_inv)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    for index in range(no_of_valid_elems):
        i = sort_ids[index]
        new_ind = index + 1
        if float(boxes[i, 4]) > conf_thresh:
            xx1 = torch.max(x1[i], x1[sort_ids[new_ind:]])
            yy1 = torch.max(y1[i], y1[sort_ids[new_ind:]])
            xx2 = torch.min(x2[i], x2[sort_ids[new_ind:]])
            yy2 = torch.min(y2[i], y2[sort_ids[new_ind:]])
            w = torch.max(torch.zeros(1, device=boxes.device), xx2 - xx1 + 1)
            h = torch.max(torch.zeros(1, device=boxes.device), yy2 - yy1 + 1)
            overlap = (w * h) / area[sort_ids[new_ind:]]
            higher_nms_ind = (overlap > nms_thresh).nonzero()
            boxes[sort_ids[new_ind:][higher_nms_ind]] = torch.zeros(7, device=boxes.device)
    return boxes.unsqueeze(0)


def get_region_boxes(output):
    conf_thresh = 0.2
    num_classes = 20
    num_anchors = 5
    anchor_step = 2
    anchors_ = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]

    anchors = torch.empty(num_anchors * 2, device=output.device)
    for i in range(num_anchors * 2):
        anchors[i] = anchors_[i]

    batch = output.size(0)
    # assert(output.size(1) == (5+num_classes)*num_anchors)
    h = output.size(2)
    w = output.size(3)

    output = output.view(batch*num_anchors, 5+num_classes, h*w).transpose(0,1).contiguous().view(5+num_classes, batch*num_anchors*h*w)

    grid_x = torch.linspace(0, w-1, w, device=output.device).repeat(h,1).repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w)
    grid_y = torch.linspace(0, h-1, h, device=output.device).repeat(w,1).t().repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w)
    xs = torch.sigmoid(output[0]) + grid_x
    ys = torch.sigmoid(output[1]) + grid_y

    anchor_w = anchors.view(num_anchors, anchor_step).index_select(1, torch.zeros(1, device=output.device).long())
    anchor_h = anchors.view(num_anchors, anchor_step).index_select(1, torch.ones(1, device=output.device).long())
    anchor_w = anchor_w.repeat(batch, 1).repeat(1, 1, h*w).view(batch*num_anchors*h*w)
    anchor_h = anchor_h.repeat(batch, 1).repeat(1, 1, h*w).view(batch*num_anchors*h*w)
    ws = torch.exp(output[2]) * anchor_w.float()
    hs = torch.exp(output[3]) * anchor_h.float()

    det_confs = torch.sigmoid(output[4])

    cls_confs = torch.softmax(output[5: 5+num_classes].transpose(0,1), dim=1)
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids = cls_max_ids.view(-1)

    sz_hw = h * w
    boxes = torch.zeros(batch, h * w * num_anchors, 7)
    # assuming only one image in a batch
    x1 = xs / w
    y1 = ys / h
    x2 = ws / w
    y2 = hs / h
    higher_confs = ((det_confs * cls_max_confs) > conf_thresh).nonzero()
    no_selected_elems = higher_confs.numel()
    if no_selected_elems > 0:
        boxes[:, 0:no_selected_elems] = torch.stack(
            [x1, y1, x2, y2, det_confs, cls_max_confs, cls_max_ids.float()], dim=1)[higher_confs.squeeze()]
    return boxes


def boxes_from_tf(output):
    boxes = get_region_boxes(output.permute(0, 3, 1, 2).contiguous())
    boxes = nms(boxes)
    return boxes
