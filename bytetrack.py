from track import *
from utils import match_bboxes, select_indices, stars, hashtags
from pydantic import BaseModel
import numpy as np
import time
import logging



class ByteTrackerConfig(BaseModel):
    max_age : int = 30
    min_box_area : int = 10
    max_aspect_ratio : float = 1.6
    high_score_det_threshold : float = 0.5
    low_score_det_threshold : float = 0.1
    init_track_score_threshold : float = 0.6
    match_high_score_dets_with_confirmed_trks_threshold : float = 0.2
    match_low_score_dets_with_confirmed_trks_threshold : float = 0.5
    match_remained_high_score_dets_with_unconfirmed_trks_threshold : float = 0.3


class ByteTracker:
    def __init__(self, config:dict={}):
        self.config = ByteTrackerConfig.model_validate(config)
        Track.init(self.config.max_age, self.config.min_box_area, self.config.max_aspect_ratio)

        logging.basicConfig(
            filename='app.log',
            filemode='w',  # Overwrite the log file each run
            format='%(message)s',
            level=logging.INFO
        )
        self.logger = logging.getLogger("app")
        self.logger.info("Application started")

    def update(self, boxes): 
        Track.predict_all()

        logger = self.logger
        logger.info(stars(''))
        logger.info(stars(f'FRAME {Track.FRAME_NUMBER}'))
        # logger.info(stars(f'FRAME {Track.FRAME_NUMBER + 1}'))
        logger.info(stars(''))
        logger.info(F'\n')

        high_confidence_detections = boxes[boxes[:, 4] >= self.config.high_score_det_threshold][:, :4]
        high_scores = boxes[boxes[:, 4] >= self.config.high_score_det_threshold][:, 4]
        low_confidence_detections = boxes[np.logical_and(boxes[:, 4] <= self.config.high_score_det_threshold, boxes[:, 4] >= self.config.low_score_det_threshold)][:, :4]
        low_scores = boxes[np.logical_and(boxes[:, 4] <= self.config.high_score_det_threshold, boxes[:, 4] >= self.config.low_score_det_threshold)][:, 4]
        logger.info(f'len high confidence dets -> {len(high_confidence_detections)}')
        logger.info(f'len low confidence dets -> {len(low_confidence_detections)}')

        logger.info('')
        logger.info(hashtags('LEVEL 1'))
        confirmed_tracks = Track.CONFIRMED_TRACKS
        confirmed_trks = [t.tlbr for t in confirmed_tracks]            
        matches, unmatched_confirmed_track_indices, unmatched_high_confidence_detection_indices = match_bboxes(confirmed_trks, high_confidence_detections, self.config.match_high_score_dets_with_confirmed_trks_threshold)
        for t_i, d_i in matches:
            confirmed_tracks[t_i].update(high_confidence_detections[d_i], score=high_scores[d_i])
        logger.info(f'confirmed_tracks -> {np.array(confirmed_trks)}')
        logger.info(f'high_confidence_detections -> {np.array(high_confidence_detections)}')
        logger.info(f'len confirmed tracks -> {len(confirmed_tracks)}')
        logger.info(f'len matches -> {len(matches)}')
        logger.info(f'len unmatched_confirmed_track_indices -> {len(unmatched_confirmed_track_indices)}')
        logger.info(f'len unmatched_high_confidence_detection_indices -> {len(unmatched_high_confidence_detection_indices)}')
        logger.info(f'matches -> {matches}')
        logger.info(f'unmatched_confirmed_track_indices -> {unmatched_confirmed_track_indices}')
        logger.info(f'unmatched_high_confidence_detection_indices -> {unmatched_high_confidence_detection_indices}')

        logger.info('')
        logger.info(hashtags('LEVEL 2'))
        remained_confirmed_tracks = select_indices(confirmed_tracks, unmatched_confirmed_track_indices)
        remained_tracking_tracks = [t for t in remained_confirmed_tracks if t.state == STATE_TRACKING]
        remained_tracking_trks = [t.tlbr for t in remained_tracking_tracks]
        matches, unmatched_remained_track_indices, unmatched_low_score_detection_indices = match_bboxes(remained_tracking_trks, low_confidence_detections, self.config.match_low_score_dets_with_confirmed_trks_threshold)
        for t_i, d_i in matches:
            remained_tracking_tracks[t_i].update(low_confidence_detections[d_i], score=low_scores[d_i])
        logger.info(f'remained_tracking_trks -> {np.array(remained_tracking_trks)}')
        logger.info(f'low_confidence_detections -> {np.array(low_confidence_detections)}')
        logger.info(f'len remained_tracking_tracks -> {len(remained_tracking_tracks)}')
        logger.info(f'len matches -> {len(matches)}')
        logger.info(f'len unmatched_remained_track_indices -> {len(unmatched_remained_track_indices)}')
        logger.info(f'len unmatched_low_score_detection_indices -> {len(unmatched_low_score_detection_indices)}')
        logger.info(f'matches -> {matches}')
        logger.info(f'unmatched_remained_track_indices -> {unmatched_remained_track_indices}')
        logger.info(f'unmatched_low_score_detection_indices -> {unmatched_low_score_detection_indices}')

        logger.info('')
        logger.info(hashtags('LEVEL 3'))
        remained_high_confidence_detections = select_indices(high_confidence_detections, unmatched_high_confidence_detection_indices)
        remained_high_scores = select_indices(high_scores, unmatched_high_confidence_detection_indices)
        unconfirmed_tracks = Track.UNCONFIRMED_TRACKS
        unconfirmed_trks = [t.tlbr for t in unconfirmed_tracks]
        matches, unmatched_unconfirmed_track_indices, unmatched_remained_high_score_detection_indices = match_bboxes(unconfirmed_trks, remained_high_confidence_detections, self.config.match_remained_high_score_dets_with_unconfirmed_trks_threshold)
        for t_i, d_i in matches:
            unconfirmed_tracks[t_i].update(remained_high_confidence_detections[d_i], score=remained_high_scores[d_i])
        logger.info(f'unconfirmed_trks -> {np.array(unconfirmed_trks)}')
        logger.info(f'remained_high_confidence_detections -> {np.array(remained_high_confidence_detections)}')
        logger.info(f'len remained_high_confidence_detections -> {len(remained_high_confidence_detections)}')
        logger.info(f'len unconfirmed_tracks -> {len(unconfirmed_tracks)}')
        logger.info(f'len matches -> {len(matches)}')
        logger.info(f'len unmatched_unconfirmed_track_indices -> {len(unmatched_unconfirmed_track_indices)}')
        logger.info(f'len unmatched_remained_high_score_detection_indices -> {len(unmatched_remained_high_score_detection_indices)}')
        logger.info(f'matches -> {matches}')
        logger.info(f'unmatched_unconfirmed_track_indices -> {unmatched_unconfirmed_track_indices}')
        logger.info(f'unmatched_remained_high_score_detection_indices -> {unmatched_remained_high_score_detection_indices}')
        
        unmatched_remained_high_score_detections = select_indices(remained_high_confidence_detections, unmatched_remained_high_score_detection_indices)
        unmatched_remained_high_scores = select_indices(remained_high_scores, unmatched_remained_high_score_detection_indices)
        inited = 0
        for d, s in zip(unmatched_remained_high_score_detections, unmatched_remained_high_scores):
            if s < self.config.init_track_score_threshold:
                continue
            inited += 1
            if Track.FRAME_NUMBER == 1:
            # if Track.FRAME_NUMBER == 0:
                Track(d, s, state=STATE_TRACKING)
            else:
                Track(d, s, state=STATE_UNCONFIRMED)
        logger.info('')
        logger.info(f'inited -> {inited}')

        # Track.predict_all()
        Track.output_tracks()

        logger.info('')
        logger.info(f'TRACKING_TRACKS -> {len(Track.TRACKING_TRACKS)}')
        logger.info(f'LOST_TRACKS -> {len(Track.LOST_TRACKS)}')
        logger.info(f'ALIVE_TRACKS -> {len(Track.ALIVE_TRACKS)}')
        logger.info(f'CONFIRMED_TRACKS -> {len(Track.CONFIRMED_TRACKS)}')
        logger.info(f'UNCONFIRMED_TRACKS -> {len(Track.UNCONFIRMED_TRACKS)}')
        logger.info(f'DELETED_TRACKS -> {len(Track.DELETED_TRACKS)}')
        
