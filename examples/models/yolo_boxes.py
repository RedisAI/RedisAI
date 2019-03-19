# import torch

def bbox_iou(box1, box2):
    mx = float(torch.min(box1[0] - box1[2] / 2.0, box2[0] - box2[2] / 2.0))
    Mx = float(torch.max(box1[0] + box1[2] / 2.0, box2[0] + box2[2] / 2.0))
    my = float(torch.min(box1[1] - box1[3] / 2.0, box2[1] - box2[3] / 2.0))
    My = float(torch.max(box1[1] + box1[3] / 2.0, box2[1] + box2[3] / 2.0))
    w1 = float(box1[2])
    h1 = float(box1[3])
    w2 = float(box2[2])
    h2 = float(box2[3])
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0.0
    if cw <= 0.0 or ch <= 0.0:
        iou = 0.0
    else:
        area1 = w1 * h1
        area2 = w2 * h2
        carea = cw * ch
        uarea = area1 + area2 - carea
        iou = carea/uarea
    return iou


def nms(boxes):
    nms_thresh = 0.5

    for b in range(boxes.shape[0]):
        det_confs = torch.zeros(boxes.shape[1])

        for i in range(boxes.shape[1]):
            det_confs[i] = 1.0 - boxes[b, i, 4]                

        _, sort_ids = torch.sort(det_confs)

        for i in range(boxes.shape[1]):
            box_i = boxes[b, sort_ids[i]]
            if float(box_i[4]) > 0.:
                for j in range(boxes.shape[1] - (i+1)):
                    j_idx = sort_ids[j + i+1]
                    box_j = boxes[b, j_idx]
                    if bbox_iou(box_i, box_j) > nms_thresh:
                        boxes[b, j_idx].zero_()

    return boxes


def get_region_boxes(output):
    conf_thresh = 0.25
    num_classes = 20
    num_anchors = 5
    anchor_step = 2
    anchors_ = [1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52]

    anchors = torch.empty(num_anchors * 2)
    for i in range(num_anchors * 2):
        anchors[i] = anchors_[i]

    batch = output.size(0)
    # assert(output.size(1) == (5+num_classes)*num_anchors)
    h = output.size(2)
    w = output.size(3)

    output = output.view(batch*num_anchors, 5+num_classes, h*w).transpose(0,1).contiguous().view(5+num_classes, batch*num_anchors*h*w)

    grid_x = torch.linspace(0, w-1, w).repeat(h,1).repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w)
    grid_y = torch.linspace(0, h-1, h).repeat(w,1).t().repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w)
    xs = torch.sigmoid(output[0]) + grid_x
    ys = torch.sigmoid(output[1]) + grid_y

    anchor_w = anchors.view(num_anchors, anchor_step).index_select(1, torch.zeros(1).long())
    anchor_h = anchors.view(num_anchors, anchor_step).index_select(1, torch.ones(1).long())
    anchor_w = anchor_w.repeat(batch, 1).repeat(1, 1, h*w).view(batch*num_anchors*h*w)
    anchor_h = anchor_h.repeat(batch, 1).repeat(1, 1, h*w).view(batch*num_anchors*h*w)
    ws = torch.exp(output[2]) * anchor_w.float()
    hs = torch.exp(output[3]) * anchor_h.float()

    det_confs = torch.sigmoid(output[4])

    cls_confs = torch.softmax(output[5: 5+num_classes].transpose(0,1), dim=1)
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids = cls_max_ids.view(-1)

    sz_hw = h*w
    sz_hwa = sz_hw*num_anchors

    boxes = torch.zeros(batch, h, w, num_anchors, 7)

    for b in range(batch):
        for cy in range(h):
            for cx in range(w):
                for i in range(num_anchors):
                    ind = b*sz_hwa + i*sz_hw + cy*w + cx
                    det_conf =  det_confs[ind]
                    conf = det_confs[ind] * cls_max_confs[ind]
    
                    if bool(conf > conf_thresh):
                        bcx = xs[ind]
                        bcy = ys[ind]
                        bw = ws[ind]
                        bh = hs[ind]
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        box = [bcx/w, bcy/h, bw/w, bh/h, det_conf, cls_max_conf, cls_max_id]
                        for j in range(7):
                            boxes[b, cy, cx, i, j] = box[j]

    boxes = boxes.view(batch, h*w*num_anchors, -1)            

    return boxes


def boxes_from_tf(output):

    boxes = get_region_boxes(output.permute(0, 3, 1, 2).contiguous())
    boxes = nms(boxes)

    # TODO: remove zero confidence boxes and resize boxes tensor

    return boxes

