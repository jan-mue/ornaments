import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16

from ornaments.data_loading import load_data
from ornaments.visualization import show_images
from ornaments.utils import gaussian_filter, triu_indices, filtered_peaks


class LatticeExtraction:

    def __init__(self, input_shape, alpha_l=(5, 7, 15, 15, 15), phi_percentile=80, delta=0.65, max_peaks=100):
        self.alpha_l = alpha_l
        self.phi_percentile = phi_percentile
        self.delta = delta
        self.max_peaks = max_peaks

        self.model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        self.model = Model(inputs=self.model.inputs, outputs=[l.output for l in self.model.layers if "pool" in l.name])
        self.vx = tf.expand_dims(tf.cast(tf.range(input_shape[0]), tf.float32), 1)
        self.vy = tf.expand_dims(tf.cast(tf.range(input_shape[1]), tf.float32), 1)

        ox, oy = tf.meshgrid(tf.range(input_shape[0]), tf.range(input_shape[1]))
        self.o = tf.concat([tf.reshape(ox, [-1, 1]), tf.reshape(oy, [-1, 1])], 1)
        self.o = tf.expand_dims(self.o, 1)

        self.sigma_l = [input_shape[1] / s[2] for s in self.model.output_shape]

    @tf.function
    def _calc_d_ij(self, peaks):
        if peaks.shape[0] == 0 or peaks.shape[0] == 1:
            return tf.zeros((0, 2), tf.int32)

        peaks = tf.random.shuffle(peaks)[:self.max_peaks]
        ind = triu_indices(tf.shape(peaks)[0])
        p_i = tf.gather_nd(peaks, tf.expand_dims(ind[:, 0], 1))
        p_j = tf.gather_nd(peaks, tf.expand_dims(ind[:, 1], 1))
        return tf.abs(p_i - p_j)

    def _calculate_displacements(self, features):
        disp_vectors = []
        for l, layer in enumerate(features):
            sigma = self.sigma_l[l]
            fmaps = gaussian_filter(layer, sigma=10/sigma)
            fmaps = tf.expand_dims(fmaps, 4)

            disp_layer = []
            for fi in range(layer.shape[3]):
                # extract peaks from each filter
                peaks = filtered_peaks(fmaps[:, :, :, fi], 32 // sigma)

                d_fl = []
                for batch_idx, batch_peaks in enumerate(peaks):
                    batch_peaks = batch_peaks[:, ::-1]

                    # calculate displacement vectors
                    d_ij = self._calc_d_ij(batch_peaks) * tf.cast(sigma, tf.int32)
                    d_fl.append(d_ij)

                disp_layer.append(d_fl)
            disp_vectors.append(disp_layer)

        return disp_vectors

    def _calc_votes(self, d_ij, sigma):
        if d_ij.shape[0] == 0:
            return 0.0

        d_ij = np.expand_dims(d_ij, 0)
        v_flij_x = np.exp(-(self.vx - d_ij[:, :, 0])**2 / (2*sigma**2)) / (2*np.pi*sigma**2)
        v_flij_y = np.exp(-(self.vy - d_ij[:, :, 1])**2 / (2*sigma**2)) / (2*np.pi*sigma**2)

        return np.einsum("ij,kj->ik", v_flij_x, v_flij_y) / d_ij.shape[1]

    def _calculate_lattice(self, disp_vectors, input_shape):
        # displacement vector voting
        voting_space = [np.zeros(input_shape[1:3]) for _ in range(input_shape[0])]

        for l, disp_layer in enumerate(disp_vectors):
            sigma = self.sigma_l[l]
            for fl, d_fl in enumerate(disp_layer):
                for batch_idx, d_ij in enumerate(d_fl):
                    voting_space[batch_idx] += self._calc_votes(d_ij, sigma)

        # calculate most consistent displacement vector
        d_star = np.array([(np.argmax(v[:, 0]), np.argmax(v[0, :])) for v in voting_space])

        return d_star

    def _calculate_consistent_displacements(self, d_star, disp_vectors):
        # calculate displacement vectors consistent with d_star
        disp_consistent = []
        disp_consistent_count = []
        for disp_layer, alpha in zip(disp_vectors, self.alpha_l):
            disp_consistent_layer = []
            for d_fl in disp_layer:
                d_fl_star = []
                for batch_disp, batch_d_star in zip(d_fl, d_star):
                    dist = tf.norm(tf.cast(batch_disp - batch_d_star, tf.float32), axis=1)
                    d_fl_star.append(tf.boolean_mask(batch_disp, dist < 3 * alpha))
                disp_consistent_count.append([d.shape[0] for d in d_fl_star])
                disp_consistent_layer.append(d_fl_star)
            disp_consistent.append(disp_consistent_layer)

        # calculate prior phi
        phi = np.percentile(disp_consistent_count, self.phi_percentile, axis=0)

        return disp_consistent, phi

    def _calculate_origin(self, d_star, disp_vectors, input_shape):
        disp_consistent, phi = self._calculate_consistent_displacements(d_star, disp_vectors)

        # calculate consistent filters and the offset of the pattern
        origin_loss = [np.zeros(input_shape[1] * input_shape[2]) for _ in range(input_shape[0])]
        for l, disp_consistent_layer in enumerate(disp_consistent):
            # calculate weights for each filter
            filter_weights = []
            for fl, d_fl_star in enumerate(disp_consistent_layer):
                w_fl = []
                for batch_idx, d_ij in enumerate(d_fl_star):
                    if d_ij.shape[0] == 0:
                        w_fl.append(0)
                        continue
                    dist = np.sum((d_ij - d_star[batch_idx]) ** 2, 1)
                    w_ijfl = np.exp(-dist / (2 * self.alpha_l[l] ** 2)) / (disp_vectors[l][fl][batch_idx].shape[0] + phi[batch_idx])
                    w_fl.append(np.sum(w_ijfl))
                filter_weights.append(w_fl)

            # calculate threshold weight for consistent filters
            threshold = self.delta * np.max(filter_weights, 0)

            # calculate origin coordinates loss
            for fl, d_fl_star in enumerate(disp_consistent_layer):
                for batch_idx, d_ij in enumerate(d_fl_star):
                    # sum loss over consistent filters only
                    if filter_weights[fl][batch_idx] <= threshold[batch_idx]:
                        continue

                    d_ij = np.expand_dims(d_ij, 0)
                    mx = (d_ij[:, :, 0] - self.o[:, :, 0]) % d_star[batch_idx][0]
                    my = (d_ij[:, :, 1] - self.o[:, :, 1]) % d_star[batch_idx][1]
                    d_star_batch = d_star[batch_idx]
                    dist = np.sqrt((mx - d_star_batch[0] / 2) ** 2 + (my - d_star_batch[1] / 2) ** 2)
                    loss = filter_weights[fl][batch_idx] * dist
                    origin_loss[batch_idx] += np.sum(loss, 1)

        o_star = [np.unravel_index(np.argmin(batch_loss), input_shape[1:3]) for batch_loss in origin_loss]

        return o_star

    def __call__(self, images):
        conv_feats = self.model(images)
        disp_vectors = self._calculate_displacements(conv_feats)
        d_star = self._calculate_lattice(disp_vectors, images.shape)
        o_star = self._calculate_origin(d_star, disp_vectors, images.shape)

        return [(a+b//2, [b[0], 0], [0, b[1]]) for a, b in zip(o_star, d_star)]


if __name__ == '__main__':

    train_ds, test_ds = load_data()
    images, labels = next(iter(train_ds.skip(1).batch(1)))
    model = LatticeExtraction(images.shape[1:])
    lattices = model(images)
    show_images(images, lattices, labels)
    print(lattices)
