import imageio
import torch
from PIL import Image


# imageio read frame as (channel, height, width)
class Video:
    def __init__(self, path, seq_max_len=None,
                 transforms=None,
                 skip_frame_ratio=None,
                 skip_frame_len=None,
                 is_rand_starting=False):
        self.path = path
        self.seq_max_len = seq_max_len
        self.transforms = transforms
        self.skip_frame_ratio = skip_frame_ratio
        self.skip_frame_len = skip_frame_len
        self.is_rand_starting = is_rand_starting

        self.container = imageio.get_reader(path, 'ffmpeg')
        self.length = self.container.count_frames()
        self.fps = self.container.get_meta_data()['fps']

    def get_all_frames(self):
        self.init_head()
        frames = []
        start_frame_idx = 0
        if (self.skip_frame_len is None) and (self.skip_frame_ratio is not None):
            self.skip_frame_len = int(self.length / self.skip_frame_ratio)
        elif self.skip_frame_len is None:
            self.skip_frame_len = 1
        
        if self.is_rand_starting:
            start_frame_idx = random.randint(0, max(0, self.length - self.seq_max_len))

        tm_frame_count = 0
        take_frame = True
        for idx, frame in enumerate(self.container):
            if idx < start_frame_idx:
                continue
            if take_frame:
                frame = Image.fromarray(frame)
                if self.transforms is not None:
                    frame = self.transforms(frame)
                frames.append(frame)

                take_frame = False
                tm_frame_count = 0

            tm_frame_count += 1
            if tm_frame_count > self.skip_frame_len:
                take_frame = True

        self.container.close()
        seq = torch.stack(frames, dim=0).float()
        seq_len = seq.size(0)
        if (self.seq_max_len is not None) and (seq_len > self.seq_max_len):
            seq = seq[:self.seq_max_len, :]
            seq_len = self.seq_max_len
        return seq, seq_len

    def init_head(self):
        self.container.set_image_index(0)

    def get_total_num_frames(self):
        meta_data = self.container.get_meta_data()
        num_frames = meta_data['nframes']
        return num_frames

    def next_frame(self):
        self.container.get_next_data()

    def get(self, key):
        return self.container.get_data(key)

    def __call__(self, key):
        return self.get(key)

    def __len__(self):
        return self.length