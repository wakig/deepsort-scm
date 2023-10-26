# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import particle_filter
from . import linear_assignment
from . import iou_matching
from .track import Track

INFTY_COST = 1e+5

def intersection(box1, box2):
    """Computes intersection.

    Parameters
    ----------
    box1, box2 : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.

    Returns
    -------
    int
        The intersection between box1 and box2.

    """

    b1_l = box1[0]
    b1_r = box1[0] + box1[2]
    b1_t = box1[1]
    b1_b = box1[1] + box1[3]

    b2_l = box2[0]
    b2_r = box2[0] + box2[2]
    b2_t = box2[1]
    b2_b = box2[1] + box2[3]

    x_overlap = max(0, min(b1_r,b2_r) - max(b1_l,b2_l))
    y_overlap = max(0, min(b1_b,b2_b) - max(b1_t,b2_t))

    return x_overlap * y_overlap

class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=300, n_init=3, lam=0.0, method='baseline', gating_dim=4): # max_age also affects prop algo!!!
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1
        self.lam = lam
        self.method = method
        self.gating_dim = gating_dim

        self.particle_filter = particle_filter.ParticleFilter()

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        if self.method != 'pf':
            for track in self.tracks:
                track.predict(self.kf)
        else:
            for track in self.tracks:
                track.predict_pf(self.pf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        if self.method != 'pf':
            for track_idx, detection_idx in matches:
                self.tracks[track_idx].update(
                    self.kf, detections[detection_idx])
        else:
            for track_idx, detection_idx in matches:
                self.tracks[track_idx].update_pf(
                    self.pf, detections[detection_idx])

        # KALMAN FILTER
        if self.method == 'baseline':
            for track_idx in unmatched_tracks:
                self.tracks[track_idx].mark_missed()

        # SCM
        if self.method == 'scm':
            for track_idx in unmatched_tracks:
                max_intersection = 0
                max_t = -1  # track with highest intersection w/ the unmatched track
                for t_idx, d_idx in matches:
                    box1 = self.tracks[track_idx].to_tlwh() # box in (top left x, top left y, width, height) format
                    box2 = self.tracks[t_idx].to_tlwh()
                    # print(box1,box2)
                    temp_intersection = intersection(box1,box2) # computes intersection of two boxes
                    # print(temp_intersection)
                    if temp_intersection > max_intersection:
                        max_intersection = temp_intersection
                        max_t = t_idx
                if max_t != -1:
                    self.tracks[track_idx].mean[0] = self.tracks[max_t].mean[0] # same x as occluder
                    self.tracks[track_idx].mean[1] = self.tracks[max_t].mean[1] # same y as occluder
                    # self.tracks[track_idx].mean[4] = 0 # velocity of x (should we do this?)
                    # self.tracks[track_idx].mean[5] = 0 # velocity of y (should we do this?)
                    self.tracks[track_idx].mean[6] = 0 # velocity of aspect ratio
                    self.tracks[track_idx].mean[7] = 0 # velocity of height
                    self.tracks[track_idx].mark_hidden()
                else:
                    self.tracks[track_idx].mark_missed()
                # print(max_t)

        # PARTICLE FILTER
        if self.method == 'pf':
            # print('ye')
            for track_idx in unmatched_tracks:
                self.tracks[track_idx].mark_missed()
        
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)       # for the appearance features and gating
            cost_matrix[cost_matrix > self.metric.matching_threshold] = INFTY_COST
            cost_matrix[cost_matrix != INFTY_COST] *= (1.0-self.lam)
            cost_matrix = linear_assignment.gate_cost_matrix(           # for the bbox gating
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices, self.lam, self.gating_dim)

            # print(cost_matrix)
            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1

    def dual_distance(self):
        pass