import numpy as np

class ParticleFilter(object):
    def __init__(self):
        np.random.seed(0)
        self.NUM_PARTICLES = 10 # default = 5000
        self.VEL_RANGE = 5.0 # 0.5
        # self.TARGET_COLOR = np.array((0, 0, 0))
        self.POS_SIGMA = 7.5 # 0.75
        self.VEL_SIGMA = 1.0 # 0.1
        self.image = None
        self.frame_width = 0
        self.frame_height = 0

    def initialize_particles(self, box):
        self.frame_width = self.image.shape[1]
        self.frame_height = self.image.shape[0]
        particles = np.random.rand(self.NUM_PARTICLES,4)
        # particles = particles * np.array((self.frame_width, self.frame_height, self.VEL_RANGE, self.VEL_RANGE))
        particles = particles * np.array((box[2], box[3], self.VEL_RANGE, self.VEL_RANGE))
        particles[:,:2] += box[:2]
        particles[:,2:4] -= self.VEL_RANGE/2.0
        # particles = self.enforce_edges(particles, box)
        # print(particles)
        # print(type(particles))
        return particles

    def apply_velocity(self, particles):
        particles[:,0] += particles[:,2]
        particles[:,1] += particles[:,3]
        particles = self.enforce_edges(particles, [0, 0, self.frame_width, self.frame_height])
        return particles

    # def set_target_color(self, pt):
    #     self.TARGET_COLOR = np.array(self.image[pt[0]][pt[1]])

    def enforce_edges(self, particles, box):
        for i in range(self.NUM_PARTICLES):
            particles[i,0] = max(box[0],min(box[2]-1, particles[i,0]))
            particles[i,1] = max(box[1],min(box[3]-1, particles[i,1]))
        return particles

    # def compute_errors(self, particles, pt):
    #     target_color = np.array(self.image[pt[1]][pt[0]])
    #     errors = np.zeros(self.NUM_PARTICLES)
    #     for i in range(self.NUM_PARTICLES):
    #         x = int(particles[i,0])
    #         y = int(particles[i,1])
    #         pixel_color = self.image[y, x] # self.image[y, x, :]
    #         errors[i] = np.sum((target_color - pixel_color)**2)
    #     print(errors)
    #     return errors

    def compute_errors(self, particles, t_box):
        t_box = t_box.astype(int)
        halfwidth = int((t_box[2] - t_box[0]) / 2)
        halfheight = int((t_box[3] - t_box[1]) / 2)
        r_hist_t = np.zeros(256)
        g_hist_t = np.zeros(256)
        b_hist_t = np.zeros(256)
        for i in range(t_box[0], t_box[2]):
            if i < 0 or i >= self.frame_width:
                continue
            for j in range(t_box[1], t_box[3]):
                if j < 0 or j >= self.frame_height:
                    continue
                rgb_t = self.image[j][i]
                r_hist_t[rgb_t[0]] += 1
                g_hist_t[rgb_t[1]] += 1
                b_hist_t[rgb_t[2]] += 1
        errors = np.zeros(self.NUM_PARTICLES)
        for i in range(self.NUM_PARTICLES):
            r_hist_p = np.zeros(256)
            g_hist_p = np.zeros(256)
            b_hist_p = np.zeros(256)
            p_x = int(particles[i,0])
            p_y = int(particles[i,1])
            p_box = [p_x - halfwidth, p_y - halfheight, p_x + halfwidth, p_y + halfheight]
            for j in range(p_box[0], p_box[2]):
                if j < 0 or j >= self.frame_width:
                    continue
                for k in range(p_box[1], p_box[3]):
                    if k < 0 or k >= self.frame_height:
                        continue
                    rgb_p = self.image[k][j]
                    r_hist_p[rgb_p[0]] += 1
                    g_hist_p[rgb_p[1]] += 1
                    b_hist_p[rgb_p[2]] += 1
            errors[i] = np.sum((r_hist_t - r_hist_p)**2) # red errors
            # print(errors)
            errors[i] += np.sum((g_hist_t - g_hist_p)**2) # green errors
            errors[i] += np.sum((b_hist_t - b_hist_p)**2) # blue errors
        # print(errors)
        return errors


    def compute_weights(self, particles, errors):
        weights = np.max(errors) - errors   # normalize the errors
        # print(weights)
        weights[
            (particles[:,0]==0) |
            (particles[:,0]==self.frame_width-1) |
            (particles[:,1]==0) |
            (particles[:,1]==self.frame_height-1) ] = 0  
        
        weights = weights**8
        
        return weights

    def resample(self, particles, weights):
        probabilities = weights / np.sum(weights)
        # print(probabilities)
        if np.isnan(probabilities).any():
            return particles, [-1, -1]
        max_prob_index = -1
        max_prob_value = -1
        for i in range(len(probabilities)):
            if probabilities[i] > max_prob_value:
                max_prob_value = probabilities[i]
                max_prob_index = i
        # print(weights)
        index_numbers = np.random.choice(
            self.NUM_PARTICLES,
            size=self.NUM_PARTICLES,
            p=probabilities)
        particles = particles[index_numbers, :]
        # print(probabilities)
        
        # x = np.mean(particles[:,0])
        # y = np.mean(particles[:,1])
        x = particles[max_prob_index,0]
        y = particles[max_prob_index,1]
        # print(x,y)
        # print(particles)
        return particles, [int(x), int(y)]

    def apply_noise(self, particles):
        noise= np.concatenate(
        (
            np.random.normal(0.0, self.POS_SIGMA, (self.NUM_PARTICLES,1)),
            np.random.normal(0.0, self.POS_SIGMA, (self.NUM_PARTICLES,1)),
            np.random.normal(0.0, self.VEL_SIGMA, (self.NUM_PARTICLES,1)),
            np.random.normal(0.0, self.VEL_SIGMA, (self.NUM_PARTICLES,1))
        ),
        axis=1)
        
        particles += noise
        return particles

    def update_image(self, image):
        self.image = image