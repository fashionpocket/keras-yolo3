#coding: utf-8

import cv2

def IoU(x, y):
    """ get IoU(Intersection Over Union) overlap ratio
    @params x,y : [left, top, right, bottom]
    """
    lx,tx,rx,bx = x
    ly,ty,ry,by = y

    area_x = (rx-lx)+(bx-tx)
    area_y = (ry-ly)+(by-ty)

    # Think About Overlap Square.
    lo = max(lx,ly)
    ro = min(rx,ry)
    to = max(tx,ty)
    bo = min(bx,by)

    inter_w = ro-lo
    inter_h = bo-to

    if inter_w < 0 or inter_h < 0:
        # No Overlap.
        return 0
    
    area_inter = inter_w*inter_h
    return area_inter / (area_x + area_y - area_inter)

def correction_bboxes(detector_results, trackerresults):
    """
    @params results : (label,confidence,top,bottom,left,right).
    """
    raise NotImprementedError()
