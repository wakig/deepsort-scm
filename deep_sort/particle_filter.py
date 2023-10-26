import numpy as np

class ParticleFilter(object):
    def __init__(self):
        np.random.seed(0)
        self.NUM_PARTICLES = 5000 # default = 5000
        self.VEL_RANGE = 0.5
        self.TARGET_COLOR = np.array((66,63, 105))
        self.POS_SIGMA = 0.75
        self.VEL_SIGMA = 0.1

    def get_frames(self, filename):
        video = cv2.VideoCapture(filename)
        while video.isOpened():
            ret, frame = video.read()
            if ret:
                yield frame 
            else:
                break
        video.release()        
        yield None

    def display(self, frame, particles, location):
        if len(particles)> 0:
            for i in range(NUM_PARTICLES):
                x = int(particles[i,0])
                y = int(particles[i,1])
                cv2.circle(frame,(x,y),1,(0,255,0),1)
                print(x,y)
        if len(location) > 0:
            cv2.circle(frame,location,15,(0,0,255),5)
        cv2.imshow('frame', frame)
        #stop the video if pressing the escape button
        if cv2.waitKey(30)==27:
            if cv2.waitKey(0)==27:
                return True 

        return False

    def initialize_particles(self):
        self.particles = np.random.rand(NUM_PARTICLES,4)
        self.particles = self.particles * np.array((self.frame_width,self.frame_height, self.VEL_RANGE,self.VEL_RANGE))
        self.particles[:,2:4] -= self.VEL_RANGE/2.0
        return self.particles

    def apply_velocity(self, particles):
        particles[:,0] += particles[:,2]
        particles[:,1] += particles[:,3]
        return particles

    def enforce_edges(self, particles):
        for i in range(NUM_PARTICLES):
            particles[i,0] = max(0,min(frame_width-1, particles[i,0]))
            particles[i,1] = max(0,min(frame_height-1, particles[i,1]))
        return particles

    def compute_errors(self, particles, frame):
        errors = np.zeros(NUM_PARTICLES)
        for i in range(NUM_PARTICLES):
            x = int(particles[i,0])
            y= int(particles[i,1])
            pixel_color = frame[y, x, :]
            errors[i] = np.sum((TARGET_COLOR - pixel_color)**2)
        return errors

    def compute_weights(self, errors):
        weights = np.max(errors) - errors
        
        weights[
            (particles[:,0]==0) |
            (particles[:,0]==frame_width-1) |
            (particles[:,1]==0) |
            (particles[:,1]==frame_height-1) ] = 0  
        
        weights = weights**8
        
        return weights

    def resample(self, particles, weights):
        probabilities = weights / np.sum(weights)
        index_numbers = np.random.choice(
            NUM_PARTICLES,
            size=NUM_PARTICLES,
            p=probabilities)
        particles = particles[index_numbers, :]
        # print(probabilities)
        
        x = np.mean(particles[:,0])
        y= np.mean(particles[:,1])
        # print(x,y)
        # print(particles)
        
        return particles, [int(x), int(y)]

    def apply_noise(self, particles):
        noise= np.concatenate(
        (
            np.random.normal(0.0, POS_SIGMA, (NUM_PARTICLES,1)),
            np.random.normal(0.0, POS_SIGMA, (NUM_PARTICLES,1)),
            np.random.normal(0.0, VEL_SIGMA, (NUM_PARTICLES,1)),
            np.random.normal(0.0, VEL_SIGMA, (NUM_PARTICLES,1))
        ),
        axis=1)
        
        particles += noise
        return particles