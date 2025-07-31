import numpy as np
import textwrap
from track_state import STATE_UNCONFIRMED, STATE_TRACKING, STATE_LOST, STATE_DELETED
from utils import z_to_tlwh, z_to_tlbr, tlbr_to_z, tlbr_to_tlwh
# from filterpy.kalman import KalmanFilter
from kalman_filter import KALMAN_FILTER

class Track:
    @staticmethod
    def init(max_age, min_box_area, max_aspect_ratio):
        Track.INSTANCES = []
        Track.ID_COUNTER = 1
        Track.FRAME_NUMBER = 0
        Track.MAX_AGE = max_age
        Track.MIN_BOX_AREA = min_box_area
        Track.MAX_ASPECT_RATIO = max_aspect_ratio
        Track.ALIVE_TRACKS:list['Track'] = []
        Track.DELETED_TRACKS:list['Track'] = []
        Track.CONFIRMED_TRACKS:list['Track'] = []
        Track.UNCONFIRMED_TRACKS:list['Track'] = []
        Track.TRACKING_TRACKS:list['Track'] = []
        Track.LOST_TRACKS:list['Track'] = []

    @staticmethod
    def predict_all() -> None:
        Track.FRAME_NUMBER += 1
        for track in Track.INSTANCES:
            if track.state not in [STATE_DELETED]:
                track.predict()
    
    @staticmethod
    def output_tracks():
        Track.ALIVE_TRACKS = [track for track in Track.INSTANCES if track.state not in [STATE_DELETED]]
        Track.DELETED_TRACKS = [track for track in Track.INSTANCES if track.state in [STATE_DELETED]]
        Track.CONFIRMED_TRACKS = [track for track in Track.INSTANCES if track.state in [STATE_TRACKING, STATE_LOST]]
        Track.UNCONFIRMED_TRACKS = [track for track in Track.INSTANCES if track.state in [STATE_UNCONFIRMED]]
        Track.TRACKING_TRACKS = [track for track in Track.INSTANCES if track.state in [STATE_TRACKING]]
        Track.LOST_TRACKS = [track for track in Track.INSTANCES if track.state in [STATE_LOST]]
    
    @property
    def mot_format(self):
        return f"{int(Track.FRAME_NUMBER)},{int(self.id)},{round(self.tlwh[0], 1)},{round(self.tlwh[1], 1)},{round(self.tlwh[2], 1)},{round(self.tlwh[3], 1)},{round(self.score, 2)},-1,-1,-1"

    @property
    def clean_format(self):
        return textwrap.dedent(f"""
            **************************************************************************************************************
            id         -> {self.id}
            state      -> {self.state.name}
            bbox       -> {self.tlwh}
            age        -> {self.age}
            score      -> {self.score}
            entered    -> {self.entered_frame}
            {f'exited     -> {self.exited_frame}' if self.state == STATE_DELETED else ''}
            {f'last state -> {self.last_state.name}' if self.last_state else ''}
            """).strip()
    
    @property
    def compressed_format(self):
        return f"{self.state.name}    {self.id}    {self.tlwh}    {self.age}    {self.score}    {self.entered_frame}    {self.exited_frame}    {self.last_state.name if self.last_state else ''}"

    @property
    def score(self):
        if len(self.scores) > 0:
            # return float(np.mean(self.scores))
            return self.scores[-1]
        else:
            return 0

    @property
    def tlwh(self):
        return z_to_tlwh(np.array(self.mean))

    @property
    def tlbr(self):
        return z_to_tlbr(np.array(self.mean))
    
    @property
    def valid(self):
        invalid_conditions = [
            self.age > Track.MAX_AGE,
            self.state == STATE_UNCONFIRMED and self.age >= 2,
            (self.tlwh[2] * self.tlwh[3]) < Track.MIN_BOX_AREA,
            (self.tlwh[2] / self.tlwh[3]) > Track.MAX_ASPECT_RATIO,
            np.any(np.isnan(self.tlwh)) or np.any(self.tlwh[2:] <= 0)
        ]   
        if any(invalid_conditions):
            return False
        else:
            return True

    def __init__(self, bbox, score, state=None):
        if state == None:
            self.state = STATE_UNCONFIRMED
        else:
            self.state = state
        self.last_state = None
        self.mean, self.covariance = KALMAN_FILTER.initiate(tlbr_to_z(bbox))
        self.predict_history = []
        self.update_history = [self.tlwh]
        self.scores = [float(score)]
        self.age = 0
        self.entered_frame = Track.FRAME_NUMBER + 1
        self.exited_frame = -1
        self.id = Track.ID_COUNTER
        Track.ID_COUNTER += 1
        Track.INSTANCES.append(self)

    def __str__(self):
        return self.clean_format

    def predict(self):
        self.age += 1
        mean = self.mean
        if self.state != STATE_TRACKING:
            mean[7] = 0
        self.mean, self.covariance = KALMAN_FILTER.predict(mean, self.covariance)
        if not self.valid:
            self.last_state = self.state
            self.state = STATE_DELETED
            self.exited_frame = Track.FRAME_NUMBER
            return
        self.predict_history.append(self.tlwh)
        if self.state == STATE_TRACKING and self.age >= 2:
            self.state = STATE_LOST
            
    def update(self, bbox, score):
        self.mean, self.covariance = KALMAN_FILTER.update(self.mean, self.covariance, tlbr_to_z(bbox))
        self.update_history.append(tlbr_to_tlwh(bbox))
        self.scores.append(float(score))
        self.age = 0
        if self.state == STATE_UNCONFIRMED:
            self.state = STATE_TRACKING
        if self.state == STATE_LOST:
            self.state = STATE_TRACKING
            self.last_state = None