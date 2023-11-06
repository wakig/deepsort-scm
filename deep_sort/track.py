import numpy as np

# vim: expandtab:ts=4:sw=4

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None, method='baseline'):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age
        self.history = []
        self.particles = np.array(0)
        self.pmean = []
        self.method = method

    def initialize_particles(self, pf):
        self.particles = pf.initialize_particles(self.to_tlwh())

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, min y, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def get_midpoint(self):
        """Get midpoint of bounding box in format `(x, y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlbr()
        return np.array([int((ret[0]+ret[2])/2), int((ret[1]+ret[3])/2)])

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def predict_pf(self, pf, kf):
        # print(self.particles)
        # self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.particles = pf.apply_velocity(self.particles)
        self.particles = pf.apply_noise(self.particles)
        # x = np.mean(self.particles[:,0])
        # y = np.mean(self.particles[:,1])
        # self.mean[0] = x
        # self.mean[1] = y
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        # print(self.covariance)
        
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

        if self.state == TrackState.Confirmed:
            center = self.to_tlwh()
            # self.history.append((center[0]+center[2]/2,center[1]+center[3]/2))
            
            #self.history.append(self.to_tlwh())
    
    def update_pf(self, pf, detection, kf):
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        # print(self.covariance)
        errors = pf.compute_errors(self.particles, detection.to_tlbr())
        weights = pf.compute_weights(self.particles, errors)
        self.particles, self.pmean = pf.resample(self.particles, weights)
        # self.particles = pf.apply_noise(self.particles)
        det_xyah = detection.to_xyah()
        # self.mean[0] = det_xyah[0]
        # self.mean[1] = det_xyah[1]
        self.mean[2] = det_xyah[2]
        self.mean[3] = det_xyah[3]
        self.mean[0] = self.pmean[0]
        self.mean[1] = self.pmean[1]

        self.features.append(detection.feature)
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
            self.particles = []
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted
            self.particles = []

    def mark_hidden(self):
        if self.time_since_update > 30: # (Arbitrary) lifespan of the occluded box
            self.state = TrackState.Deleted
            self.particles = []

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
