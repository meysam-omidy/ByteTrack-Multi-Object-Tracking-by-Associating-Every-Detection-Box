import os
import pickle
import numpy as np
from bytetrack import ByteTracker
from track import *
from evaluate import evaluate
from utils import count_time

@count_time
def run():
    seqs = ['MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-FRCNN', 'MOT17-09-FRCNN', 'MOT17-10-FRCNN', 'MOT17-11-FRCNN', 'MOT17-13-FRCNN', ]

    os.makedirs('outputs/bytetrack-self', exist_ok=True)

    seqmap = open('./trackeval/seqmap/mot17/custom.txt', 'w')
    seqmap.write('name\n')
    for seq in seqs:
        
        seqmap.write(f'{seq}\n')
        file = open(f'outputs/bytetrack-self/{seq}.txt', 'w')
        detections = np.loadtxt(f'detections/bytetrack_x_mot17/{seq}.txt', delimiter=',')
        gt_dets_file = np.loadtxt(f'../../.Datasets/MOT17/train/{seq}/gt/gt.txt', delimiter=',')

        tracker = ByteTracker()

        for i,frame_number in enumerate(np.unique(gt_dets_file[:,0])):
            dets = detections[detections[:, 0] == frame_number][:, 1:]
            tracker.update(dets)
            for track in Track.TRACKING_TRACKS:
                file.write(f'{track.mot_format}\n')
        file.close()
    seqmap.close()




if __name__ == '__main__':
    print('tracking...')
    run()
    print('evaluating...')
    evaluate()