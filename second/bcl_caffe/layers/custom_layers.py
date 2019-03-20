import numpy as np
import caffe

class InputData(caffe.Layer):
    def _restart(self):
        # make a deep copy
        data = [d.copy() for d in self.data_copy]
        label = [l.copy() for l in self.label_copy]

        # duplicate if necessary to fill batch
        num_samples = len(data)
        if num_samples < self.batch_size:
            idx = np.concatenate((np.tile(np.arange(num_samples), (self.batch_size // num_samples, )),
                                  np.random.permutation(num_samples)[:(self.batch_size % num_samples)]), axis=0)
            data, label = [data[i] for i in idx], [label[i] for i in idx]
            num_samples = self.batch_size

        # sample to a fixed length
        for i in range(num_samples):
            k = len(data[i])
            idx = np.concatenate((np.tile(np.arange(k), (self.sample_size // k, )),
                                  np.random.permutation(k)[:(self.sample_size % k)]), axis=0)
            data[i] = data[i][idx, :]
            label[i] = label[i][idx]

        data = np.concatenate(data, axis=0)     # (NxS) x C
        label = np.concatenate(label, axis=0)   # (NxS)

        # reshape and reset index
        self.data = data[:, self.feat_dims].reshape(num_samples, self.sample_size, -1, 1).transpose(0, 2, 3, 1)
        self.label = label.reshape(num_samples, self.sample_size, -1, 1).transpose(0, 2, 3, 1)
        self.index = 0

    def setup(self, bottom, top):
        params = dict(subset='train', category='02691156', batch_size=32, sample_size=3000,
                      feat_dims='x_y_z',        # choose from 'x', 'y', 'z', 'nx', 'ny', 'nz' and 'one'
                      #jitter_xyz=0.01,          # random displacements
                      #jitter_stretch=0.1,       # random stretching (uniform random within +- this value)
                      #jitter_rotation=10,       # random rotation along three axis (in degrees)
                      root=SHAPENET3D_DATA_DIR)
        params.update(eval(self.param_str))
        self.batch_size = params['batch_size']
        self.sample_size = params['sample_size']
        #self.jitter_xyz = params['jitter_xyz']
        #self.jitter_stretch = params['jitter_stretch']
        #self.jitter_rotation = params['jitter_rotation']

        self.raw_dims = []
        for feat_group in [['x', 'y', 'z'], ['nx', 'ny', 'nz'], ['one']]:
            if np.any([f in feat_group for f in params['feat_dims'].split('_')]):
                self.raw_dims.extend(feat_group)
        self.feat_dims = [self.raw_dims.index(f) for f in params['feat_dims'].split('_')]

        data, _, label, _ = points_single_category(params['subset'], params['category'],
                                                   dims='_'.join(self.raw_dims), root=params['root'])
        self.data_copy = data
        self.label_copy = label
        self.top_names = ['data', 'label']
        self.top_channels = [len(self.raw_dims), 1]

        if len(top) != len(self.top_names):
            raise Exception('Incorrect number of outputs (expected %d, got %d)' %
                            (len(self.top_names), len(top)))

        self._restart()

    def reshape(self, bottom, top):
        for top_index, name in enumerate(self.top_names):
            shape = (self.batch_size, self.top_channels[top_index], 1, self.sample_size)
            top[top_index].reshape(*shape)

    def forward(self, bottom, top):
        top[0].data[...] = self.data[self.index:self.index+self.batch_size]
        top[1].data[...] = self.label[self.index:self.index+self.batch_size]

        self.index += self.batch_size
        if self.index + self.batch_size > len(self.data):
            self._restart()

    def backward(self, top, propagate_down, bottom):
        pass
