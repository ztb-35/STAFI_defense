import os
import io
import json
import torch
from math import pi
import numpy as np
from scipy.interpolate import interp1d
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils import warp, generate_random_params_for_warp
from view_transform import calibration

import utils_comma2k19.orientation as orient
import utils_comma2k19.coordinates as coord

def bgr_to_op6(bgr):
    """
    bgr: uint8 (256,512,3) in bgr order
    returns: uint8 (6,128,256): [Y00, Y01, Y10, Y11, U420, V420]
    """
    H, W = bgr.shape[:2]
    if (H, W) != (256, 512):
        bgr = cv2.resize(bgr, (512, 256), interpolation=cv2.INTER_AREA)
        H, W = 256, 512

    # RGB -> I420 (Y full, then U half, V half)
    yuv_i420 = cv2.cvtColor(bgr, cv2.COLOR_RGB2YUV_I420)

    y_size = H * W
    uv_size = (H // 2) * (W // 2)

    Y = yuv_i420.flatten()[:y_size].reshape(H, W)
    U = yuv_i420.flatten()[y_size:y_size + uv_size].reshape(H // 2, W // 2)
    V = yuv_i420.flatten()[y_size + uv_size:].reshape(H // 2, W // 2)

    # Split Y into 4 checkerboard tiles
    ch0 = Y[0::2, 0::2]  # (128,256)
    ch1 = Y[0::2, 1::2]
    ch2 = Y[1::2, 0::2]
    ch3 = Y[1::2, 1::2]
    ch4 = U              # (128,256)
    ch5 = V              # (128,256)

    op6 = np.stack([ch0, ch1, ch2, ch3, ch4, ch5], axis=0).astype(np.uint8)
    return op6

class PlanningDataset(Dataset):
    def __init__(self, root='data', json_path_pattern='p3_%s.json', split='train'):
        self.samples = json.load(open(os.path.join(root, json_path_pattern % split)))
        print('PlanningDataset: %d samples loaded from %s' % 
              (len(self.samples), os.path.join(root, json_path_pattern % split)))
        self.split = split

        self.img_root = os.path.join(root, 'nuscenes')
        self.transforms = transforms.Compose(
            [
                # transforms.Resize((900 // 2, 1600 // 2)),
                # transforms.Resize((9 * 32, 16 * 32)),
                transforms.Resize((128, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.3890, 0.3937, 0.3851],
                                     [0.2172, 0.2141, 0.2209]),
            ]
        )

        self.enable_aug = False
        self.view_transform = False

        self.use_memcache = False
        if self.use_memcache:
            self._init_mc_()

    def _init_mc_(self):
        from petrel_client.client import Client
        self.client = Client('~/petreloss.conf')
        print('======== Initializing Memcache: Success =======')

    def _get_cv2_image(self, path):
        if self.use_memcache:
            img_bytes = self.client.get(str(path))
            assert(img_bytes is not None)
            img_mem_view = memoryview(img_bytes)
            img_array = np.frombuffer(img_mem_view, np.uint8)
            return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        else:
            return cv2.imread(path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        imgs, future_poses = sample['imgs'], sample['future_poses']

        # process future_poses
        future_poses = torch.tensor(future_poses)
        future_poses[:, 0] = future_poses[:, 0].clamp(1e-2, )  # the car will never go backward

        imgs = list(self._get_cv2_image(os.path.join(self.img_root, p)) for p in imgs)
        imgs = list(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs)  # RGB

        # process images
        if self.enable_aug and self.split == 'train':
            # data augumentation when training
            # random distort (warp)
            w_offsets, h_offsets = generate_random_params_for_warp(imgs[0], random_rate=0.1)
            imgs = list(warp(img, w_offsets, h_offsets) for img in imgs)

            # random flip
            if np.random.rand() > 0.5:
                imgs = list(img[:, ::-1, :] for img in imgs)
                future_poses[:, 1] *= -1
            

        if self.view_transform:
            camera_rotation_matrix = np.linalg.inv(np.array(sample["camera_rotation_matrix_inv"]))
            camera_translation = -np.array(sample["camera_translation_inv"])
            camera_extrinsic = np.vstack((np.hstack((camera_rotation_matrix, camera_translation.reshape((3, 1)))), np.array([0, 0, 0, 1])))
            camera_extrinsic = np.linalg.inv(camera_extrinsic)
            warp_matrix = calibration(camera_extrinsic, np.array(sample["camera_intrinsic"]))
            imgs = list(cv2.warpPerspective(src = img, M = warp_matrix, dsize= (256,128), flags= cv2.WARP_INVERSE_MAP) for img in imgs)

        # cvt back to PIL images
        # cv2.imshow('0', imgs[0])
        # cv2.imshow('1', imgs[1])
        # cv2.waitKey(0)
        imgs = list(Image.fromarray(img) for img in imgs)
        imgs = list(self.transforms(img) for img in imgs)
        input_img = torch.cat(imgs, dim=0)

        return dict(
            input_img=input_img,
            future_poses=future_poses,
            camera_intrinsic=torch.tensor(sample['camera_intrinsic']),
            camera_extrinsic=torch.tensor(sample['camera_extrinsic']),
            camera_translation_inv=torch.tensor(sample['camera_translation_inv']),
            camera_rotation_matrix_inv=torch.tensor(sample['camera_rotation_matrix_inv']),
        )


class SequencePlanningDataset(PlanningDataset):
    def __init__(self, root='data', json_path_pattern='p3_%s.json', split='train'):
        print('Sequence', end='')
        self.fix_seq_length = 18
        super().__init__(root=root, json_path_pattern=json_path_pattern, split=split)

    def __getitem__(self, idx):
        seq_samples = self.samples[idx]
        seq_length = len(seq_samples)
        if seq_length < self.fix_seq_length:
            # Only 1 sample < 28 (==21)
            return self.__getitem__(np.random.randint(0, len(self.samples)))
        if seq_length > self.fix_seq_length:
            seq_length_delta = seq_length - self.fix_seq_length
            seq_length_delta = np.random.randint(0, seq_length_delta+1)
            seq_samples = seq_samples[seq_length_delta:self.fix_seq_length+seq_length_delta]

        seq_future_poses = list(smp['future_poses'] for smp in seq_samples)
        seq_imgs = list(smp['imgs'] for smp in seq_samples)

        seq_input_img = []
        for imgs in seq_imgs:
            imgs = list(self._get_cv2_image(os.path.join(self.img_root, p)) for p in imgs)
            imgs = list(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs)  # RGB
            imgs = list(Image.fromarray(img) for img in imgs)
            imgs = list(self.transforms(img) for img in imgs)
            input_img = torch.cat(imgs, dim=0)
            seq_input_img.append(input_img[None])
        seq_input_img = torch.cat(seq_input_img)

        return dict(
            seq_input_img=seq_input_img,  # torch.Size([28, 10, 3])
            seq_future_poses=torch.tensor(seq_future_poses),  # torch.Size([28, 6, 128, 256])
            camera_intrinsic=torch.tensor(seq_samples[0]['camera_intrinsic']),
            camera_extrinsic=torch.tensor(seq_samples[0]['camera_extrinsic']),
            camera_translation_inv=torch.tensor(seq_samples[0]['camera_translation_inv']),
            camera_rotation_matrix_inv=torch.tensor(seq_samples[0]['camera_rotation_matrix_inv']),
        )


class Comma2k19SequenceDataset(PlanningDataset):
    def __init__(self, split_txt_path, prefix, data_length, mode, use_memcache=True, return_origin=False):
        self.split_txt_path = split_txt_path
        self.prefix = prefix
        self.data_length = data_length
        self.samples = open(split_txt_path).readlines()
        self.samples = [i.strip() for i in self.samples]

        assert mode in ('train', 'val', 'demo')
        self.mode = mode
        if self.mode == 'demo':
            print('Comma2k19SequenceDataset: DEMO mode is on.')

        #self.fix_seq_length = 800 if mode == 'train' else 800
        self.fix_seq_length = self.data_length if mode == 'train' else self.data_length
        self.transforms = transforms.Compose(
            [
                # transforms.Resize((900 // 2, 1600 // 2)),
                # transforms.Resize((9 * 32, 16 * 32)),
                transforms.Resize((128, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.3890, 0.3937, 0.3851],
                                     [0.2172, 0.2141, 0.2209]),
            ]
        )

        self.warp_matrix = calibration(extrinsic_matrix=np.array([[ 0, -1,  0,    0],
                                                                  [ 0,  0, -1, 1.22],
                                                                  [ 1,  0,  0,    0],
                                                                  [ 0,  0,  0,    1]]),
                                       cam_intrinsics=np.array([[910, 0, 582],
                                                                [0, 910, 437],
                                                                [0,   0,   1]]),
                                       device_frame_from_road_frame=np.hstack((np.diag([1, -1, -1]), [[0], [0], [1.22]])))

        self.use_memcache = use_memcache
        if self.use_memcache:
            self._init_mc_()

        self.return_origin = return_origin

        # from OpenPilot
        self.num_pts = 10 * 20  # 10 s * 20 Hz = 200 frames
        self.t_anchors = np.array(
            (0.        ,  0.00976562,  0.0390625 ,  0.08789062,  0.15625   ,
             0.24414062,  0.3515625 ,  0.47851562,  0.625     ,  0.79101562,
             0.9765625 ,  1.18164062,  1.40625   ,  1.65039062,  1.9140625 ,
             2.19726562,  2.5       ,  2.82226562,  3.1640625 ,  3.52539062,
             3.90625   ,  4.30664062,  4.7265625 ,  5.16601562,  5.625     ,
             6.10351562,  6.6015625 ,  7.11914062,  7.65625   ,  8.21289062,
             8.7890625 ,  9.38476562, 10.)
        )
        self.t_idx = np.linspace(0, 10, num=self.num_pts)

    def _get_cv2_vid(self, path):
        if self.use_memcache:
            path = self.client.generate_presigned_url(str(path), client_method='get_object', expires_in=3600)
        else:
            # Resolve to absolute path to avoid OpenCV pipe wildcard interpretation
            path = os.path.abspath(path)
        return cv2.VideoCapture(path)

    def _get_numpy(self, path):
        if self.use_memcache:
            bytes = io.BytesIO(memoryview(self.client.get(str(path))))
            return np.lib.format.read_array(bytes)
        else:
            return np.load(path)

    def __getitem__(self, idx):
        seq_sample_path = self.prefix + self.samples[idx]
        cap = self._get_cv2_vid(seq_sample_path + '/video.hevc')
        if (cap.isOpened() == False):
            raise RuntimeError
        imgs = []  # <--- all frames here
        origin_imgs = []
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                imgs.append(frame)
                # cv2.imshow('frame', frame)
                # cv2.waitKey(0)
                if self.return_origin:
                    origin_imgs.append(frame)
            else:
                break
        cap.release()

        seq_length = len(imgs)

        # Defensive: if VideoCapture opened but returned no frames, skip to next sample
        if seq_length == 0:
            print('Warning: no frames read from', seq_sample_path)
            return self.__getitem__((idx + 1) % len(self.samples))
        # Defensive: if VideoCapture opened but returned no frames, skip to next sample
        if seq_length == 0:
            print('Warning: no frames read from', seq_sample_path)
            return self.__getitem__((idx + 1) % len(self.samples))

        if self.mode == 'demo':
            self.fix_seq_length = seq_length - self.num_pts - 1

        if seq_length < self.fix_seq_length + self.num_pts:
            print('The length of sequence', seq_sample_path, 'is too short',
                  '(%d < %d)' % (seq_length, self.fix_seq_length + self.num_pts))
            return self.__getitem__(idx+1)

        max_delta = seq_length - (self.fix_seq_length + self.num_pts)
        # if there's no extra length, start at 0; otherwise pick a random offset in [1, max_delta]
        if max_delta <= 0:
            seq_length_delta = 0
        else:
            seq_length_delta = np.random.randint(1, max_delta + 1)

        seq_start_idx = seq_length_delta
        seq_end_idx = seq_length_delta + self.fix_seq_length

        # seq_input_img
        imgs = imgs[seq_start_idx-1: seq_end_idx]  # contains one more img
        imgs = [cv2.warpPerspective(src=img, M=self.warp_matrix, dsize=(512,256), flags=cv2.WARP_INVERSE_MAP) for img in imgs]
        imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
        imgs = list(Image.fromarray(img) for img in imgs)
        imgs = list(self.transforms(img)[None] for img in imgs)
        input_img_1 = torch.cat(imgs, dim=0)  # [N+1, 3, H, W]
        del imgs

        input_img_1 = torch.cat((input_img_1[:-1, ...], input_img_1[1:, ...]), dim=1)
        input_img = input_img_1[-1,:,:,:]
        del input_img_1

        # poses
        frame_positions = self._get_numpy(self.prefix + self.samples[idx] + '/global_pose/frame_positions')[seq_start_idx: seq_end_idx+self.num_pts]
        frame_orientations = self._get_numpy(self.prefix + self.samples[idx] + '/global_pose/frame_orientations')[seq_start_idx: seq_end_idx+self.num_pts]

        future_poses = []
        for i in range(self.fix_seq_length):
            ecef_from_local = orient.rot_from_quat(frame_orientations[i])
            local_from_ecef = ecef_from_local.T
            frame_positions_local = np.einsum('ij,kj->ki', local_from_ecef, frame_positions - frame_positions[i]).astype(np.float32)

            # Time-Anchor like OpenPilot
            fs = [interp1d(self.t_idx, frame_positions_local[i: i+self.num_pts, j]) for j in range(3)]
            interp_positions = [fs[j](self.t_anchors)[:, None] for j in range(3)]
            interp_positions = np.concatenate(interp_positions, axis=1)
            
            future_poses.append(interp_positions)
        future_poses = torch.tensor(np.array(future_poses), dtype=torch.float32)

        rtn_dict = dict(
            seq_input_img=input_img,  # torch.Size([N, 6, 128, 256])
            seq_future_poses=future_poses,  # torch.Size([N, num_pts, 3])
            # camera_intrinsic=torch.tensor(seq_samples[0]['camera_intrinsic']),
            # camera_extrinsic=torch.tensor(seq_samples[0]['camera_extrinsic']),
            # camera_translation_inv=torch.tensor(seq_samples[0]['camera_translation_inv']),
            # camera_rotation_matrix_inv=torch.tensor(seq_samples[0]['camera_rotation_matrix_inv']),
        )

        # For DEMO
        if self.return_origin:
            origin_imgs = origin_imgs[seq_start_idx: seq_end_idx]
            origin_imgs = [torch.tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))[None] for img in origin_imgs]
            origin_imgs = torch.cat(origin_imgs, dim=0)  # N, H_ori, W_ori, 3
            rtn_dict['origin_imgs'] = origin_imgs

        return rtn_dict

import torchvision.transforms.functional as F

class To255Float:
    def __call__(self, img):
        # img: PIL Image (uint8)
        x = F.to_tensor(img)
        return x * 255.0         # float32, [0,255]

class Comma2k19SequenceRecurrentDataset(PlanningDataset):
    def __init__(self, split_txt_path, prefix, data_length, frame_stream_length, mode, use_memcache=True, return_origin=False):
        self.split_txt_path = split_txt_path
        self.prefix = prefix
        self.data_length = data_length
        self.frame_stream_length = frame_stream_length#length of frame stream to get a better initial input state for supercombo
        self.samples = open(split_txt_path).readlines()
        self.samples = [i.strip() for i in self.samples]

        assert mode in ('train', 'val', 'demo')
        self.mode = mode
        if self.mode == 'demo':
            print('Comma2k19SequenceDataset: DEMO mode is on.')

        # self.fix_seq_length = 800 if mode == 'train' else 800
        self.fix_seq_length = self.data_length if mode == 'train' else self.data_length
        self.transforms = transforms.Compose(
            [
                # transforms.Resize((900 // 2, 1600 // 2)),
                # transforms.Resize((9 * 32, 16 * 32)),
                #transforms.Resize((128, 256)),
                To255Float(),
                # transforms.ToTensor(),
                # transforms.Normalize([0.3890, 0.3937, 0.3851],
                #                      [0.2172, 0.2141, 0.2209]),
            ]
        )

        self.warp_matrix = calibration(extrinsic_matrix=np.array([[0, -1, 0, 0],
                                                                  [0, 0, -1, 1.22],
                                                                  [1, 0, 0, 0],
                                                                  [0, 0, 0, 1]]),
                                       cam_intrinsics=np.array([[910, 0, 582],
                                                                [0, 910, 437],
                                                                [0, 0, 1]]),
                                       device_frame_from_road_frame=np.hstack(
                                           (np.diag([1, -1, -1]), [[0], [0], [1.22]])))

        self.use_memcache = use_memcache
        if self.use_memcache:
            self._init_mc_()

        self.return_origin = return_origin

        # from OpenPilot
        self.num_pts = 10 * 20  # 10 s * 20 Hz = 200 frames
        self.t_anchors = np.array(
            (0., 0.00976562, 0.0390625, 0.08789062, 0.15625,
             0.24414062, 0.3515625, 0.47851562, 0.625, 0.79101562,
             0.9765625, 1.18164062, 1.40625, 1.65039062, 1.9140625,
             2.19726562, 2.5, 2.82226562, 3.1640625, 3.52539062,
             3.90625, 4.30664062, 4.7265625, 5.16601562, 5.625,
             6.10351562, 6.6015625, 7.11914062, 7.65625, 8.21289062,
             8.7890625, 9.38476562, 10.)
        )
        self.t_idx = np.linspace(0, 10, num=self.num_pts)

    def _get_cv2_vid(self, path):
        if self.use_memcache:
            path = self.client.generate_presigned_url(str(path), client_method='get_object', expires_in=3600)
        else:
            # Resolve to absolute path to avoid OpenCV pipe wildcard interpretation
            path = os.path.abspath(path)
        return cv2.VideoCapture(path)

    def _get_numpy(self, path):
        if self.use_memcache:
            bytes = io.BytesIO(memoryview(self.client.get(str(path))))
            return np.lib.format.read_array(bytes)
        else:
            return np.load(path)

    def __getitem__(self, idx):
        seq_sample_path = self.prefix + self.samples[idx]
        cap = self._get_cv2_vid(seq_sample_path + '/video.hevc')
        if (cap.isOpened() == False):
            raise RuntimeError
        imgs = []  # <--- all frames here
        origin_imgs = []
        while (cap.isOpened()):
            ret, frame = cap.read()

            if ret == True:
                imgs.append(frame)
                # cv2.imshow('frame', frame)
                # cv2.waitKey(0)
                if self.return_origin:
                    origin_imgs.append(frame)
            else:
                break
        cap.release()

        seq_length = len(imgs)
        # Defensive: if VideoCapture opened but returned no frames, skip to next sample
        if seq_length == 0:
            print('Warning: no frames read from', seq_sample_path)
            # wrap around to avoid index error; try next sample
            return self.__getitem__((idx + 1) % len(self.samples))
        


        if self.mode == 'demo':
            self.fix_seq_length = seq_length - self.num_pts - 1

        if seq_length < self.fix_seq_length + self.num_pts:
            print('The length of sequence', seq_sample_path, 'is too short',
                  '(%d < %d)' % (seq_length, self.fix_seq_length + self.num_pts))
            return self.__getitem__(idx + 1)

        max_delta = seq_length - (self.fix_seq_length + self.num_pts)
        # if there's no extra length, start at 0; otherwise pick a random offset in [1, max_delta]
        if max_delta <= 0:
            seq_length_delta = 0
        else:
            seq_length_delta = np.random.randint(1, max_delta + 1)
        # TODO: MANUALLY SET 50
        # seq_length_delta = 50
        
        # print(f"seq len = {seq_length}")
        # print(f"fix seq len = {self.fix_seq_length}")
        # print(f"seq_length_delta = {seq_length_delta}")

        seq_start_idx = seq_length_delta
        seq_end_idx = seq_length_delta + self.fix_seq_length

        # seq_input_img
        imgs = imgs[seq_start_idx - 1: seq_end_idx]  # contains one more img
        imgs = [cv2.warpPerspective(src=img, M=self.warp_matrix, dsize=(512, 256), flags=cv2.WARP_INVERSE_MAP) for img
                in imgs]
        #imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
        imgs = [bgr_to_op6(img) for img in imgs]
        #imgs = list(Image.fromarray(img) for img in imgs)
        imgs = [torch.tensor(img, dtype=torch.float32) for img in imgs]
        input_img_1 = torch.stack(imgs)  # [N+1, 3, H, W]
        del imgs
        #convert input_img_1 a bgr sequence to a YUV sequence

        input_img_1 = torch.cat((input_img_1[:-1, ...], input_img_1[1:, ...]), dim=1)
        input_img = input_img_1[-self.frame_stream_length:, :, :, :]#Here we adapt a frame stream with 10 frames in our recurrent pipeline
        
        del input_img_1

        # poses
        frame_positions = self._get_numpy(self.prefix + self.samples[idx] + '/global_pose/frame_positions')[
                          seq_start_idx: seq_end_idx + self.num_pts]
        frame_orientations = self._get_numpy(self.prefix + self.samples[idx] + '/global_pose/frame_orientations')[
                             seq_start_idx: seq_end_idx + self.num_pts]

        future_poses = []
        for i in range(self.fix_seq_length):
            ecef_from_local = orient.rot_from_quat(frame_orientations[i])
            local_from_ecef = ecef_from_local.T
            frame_positions_local = np.einsum('ij,kj->ki', local_from_ecef,
                                              frame_positions - frame_positions[i]).astype(np.float32)

            # Time-Anchor like OpenPilot
            fs = [interp1d(self.t_idx, frame_positions_local[i: i + self.num_pts, j]) for j in range(3)]
            interp_positions = [fs[j](self.t_anchors)[:, None] for j in range(3)]
            interp_positions = np.concatenate(interp_positions, axis=1)

            future_poses.append(interp_positions)
        future_poses_ = torch.tensor(np.array(future_poses), dtype=torch.float32)
        future_poses = future_poses_[-self.frame_stream_length:, :, :]#Here we adapt a frame stream with 10 frames in our recurrent pipeline
        del future_poses_

        rtn_dict = dict(
            seq_input_img=input_img,  # torch.Size([N, T, 6, 128, 256])
            seq_future_poses=future_poses,  # torch.Size([N, T, num_pts, 3])
            # camera_intrinsic=torch.tensor(seq_samples[0]['camera_intrinsic']),
            # camera_extrinsic=torch.tensor(seq_samples[0]['camera_extrinsic']),
            # camera_translation_inv=torch.tensor(seq_samples[0]['camera_translation_inv']),
            # camera_rotation_matrix_inv=torch.tensor(seq_samples[0]['camera_rotation_matrix_inv']),
        )

        # For DEMO
        if self.return_origin:
            origin_imgs = origin_imgs[seq_start_idx: seq_end_idx]
            origin_imgs = [torch.tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))[None] for img in origin_imgs]
            origin_imgs = torch.cat(origin_imgs, dim=0)  # N, H_ori, W_ori, 3
            rtn_dict['origin_imgs'] = origin_imgs

        return rtn_dict

class Comma2k19SequenceIndexDataset(PlanningDataset):
    def __init__(self, split_txt_path, prefix, data_length, mode, use_memcache=True, return_origin=False):
        self.split_txt_path = split_txt_path
        self.prefix = prefix
        self.data_length = data_length
        self.samples = open(split_txt_path).readlines()
        self.samples = [i.strip() for i in self.samples]

        assert mode in ('train', 'val', 'demo')
        self.mode = mode
        if self.mode == 'demo':
            print('Comma2k19SequenceDataset: DEMO mode is on.')

        # self.fix_seq_length = 800 if mode == 'train' else 800
        self.fix_seq_length = self.data_length if mode == 'train' else self.data_length
        self.transforms = transforms.Compose(
            [
                # transforms.Resize((900 // 2, 1600 // 2)),
                # transforms.Resize((9 * 32, 16 * 32)),
                transforms.Resize((128, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.3890, 0.3937, 0.3851],
                                     [0.2172, 0.2141, 0.2209]),
            ]
        )

        self.warp_matrix = calibration(extrinsic_matrix=np.array([[0, -1, 0, 0],
                                                                  [0, 0, -1, 1.22],
                                                                  [1, 0, 0, 0],
                                                                  [0, 0, 0, 1]]),
                                       cam_intrinsics=np.array([[910, 0, 582],
                                                                [0, 910, 437],
                                                                [0, 0, 1]]),
                                       device_frame_from_road_frame=np.hstack(
                                           (np.diag([1, -1, -1]), [[0], [0], [1.22]])))

        self.use_memcache = use_memcache
        if self.use_memcache:
            self._init_mc_()

        self.return_origin = return_origin

        # from OpenPilot
        self.num_pts = 10 * 20  # 10 s * 20 Hz = 200 frames
        self.t_anchors = np.array(
            (0., 0.00976562, 0.0390625, 0.08789062, 0.15625,
             0.24414062, 0.3515625, 0.47851562, 0.625, 0.79101562,
             0.9765625, 1.18164062, 1.40625, 1.65039062, 1.9140625,
             2.19726562, 2.5, 2.82226562, 3.1640625, 3.52539062,
             3.90625, 4.30664062, 4.7265625, 5.16601562, 5.625,
             6.10351562, 6.6015625, 7.11914062, 7.65625, 8.21289062,
             8.7890625, 9.38476562, 10.)
        )
        self.t_idx = np.linspace(0, 10, num=self.num_pts)

    def _get_cv2_vid(self, path):
        if self.use_memcache:
            path = self.client.generate_presigned_url(str(path), client_method='get_object', expires_in=3600)
        else:
            # Resolve to absolute path to avoid OpenCV pipe wildcard interpretation
            path = os.path.abspath(path)
        return cv2.VideoCapture(path)

    def _get_numpy(self, path):
        if self.use_memcache:
            bytes = io.BytesIO(memoryview(self.client.get(str(path))))
            return np.lib.format.read_array(bytes)
        else:
            return np.load(path)

    def __getitem__(self, idx):
        seq_sample_path = self.prefix + self.samples[idx]
        cap = self._get_cv2_vid(seq_sample_path + '/video.hevc')
        if (cap.isOpened() == False):
            raise RuntimeError
        imgs = []  # <--- all frames here
        origin_imgs = []
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                imgs.append(frame)
                # cv2.imshow('frame', frame)
                # cv2.waitKey(0)
                if self.return_origin:
                    origin_imgs.append(frame)
            else:
                break
        cap.release()

        seq_length = len(imgs)

        if self.mode == 'demo':
            self.fix_seq_length = seq_length - self.num_pts - 1

        if seq_length < self.fix_seq_length + self.num_pts:
            print('The length of sequence', seq_sample_path, 'is too short',
                  '(%d < %d)' % (seq_length, self.fix_seq_length + self.num_pts))
            return self.__getitem__(idx + 1)

        max_delta = seq_length - (self.fix_seq_length + self.num_pts)
        # if there's no extra length, start at 0; otherwise pick a random offset in [1, max_delta]
        if max_delta <= 0:
            seq_length_delta = 0
        else:
            seq_length_delta = np.random.randint(1, max_delta + 1)

        seq_start_idx = seq_length_delta
        seq_end_idx = seq_length_delta + self.fix_seq_length

        # seq_input_img
        imgs = imgs[seq_start_idx - 1: seq_end_idx]  # contains one more img
        imgs = [cv2.warpPerspective(src=img, M=self.warp_matrix, dsize=(512, 256), flags=cv2.WARP_INVERSE_MAP) for img
                in imgs]
        imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
        imgs = list(Image.fromarray(img) for img in imgs)
        imgs = list(self.transforms(img)[None] for img in imgs)
        input_img_1 = torch.cat(imgs, dim=0)  # [N+1, 3, H, W]
        del imgs

        input_img_1 = torch.cat((input_img_1[:-1, ...], input_img_1[1:, ...]), dim=1)
        input_img = input_img_1[-1, :, :, :]
        del input_img_1

        # poses
        frame_positions = self._get_numpy(self.prefix + self.samples[idx] + '/global_pose/frame_positions')[
                          seq_start_idx: seq_end_idx + self.num_pts]
        frame_orientations = self._get_numpy(self.prefix + self.samples[idx] + '/global_pose/frame_orientations')[
                             seq_start_idx: seq_end_idx + self.num_pts]

        future_poses = []
        for i in range(self.fix_seq_length):
            ecef_from_local = orient.rot_from_quat(frame_orientations[i])
            local_from_ecef = ecef_from_local.T
            frame_positions_local = np.einsum('ij,kj->ki', local_from_ecef,
                                              frame_positions - frame_positions[i]).astype(np.float32)

            # Time-Anchor like OpenPilot
            fs = [interp1d(self.t_idx, frame_positions_local[i: i + self.num_pts, j]) for j in range(3)]
            interp_positions = [fs[j](self.t_anchors)[:, None] for j in range(3)]
            interp_positions = np.concatenate(interp_positions, axis=1)

            future_poses.append(interp_positions)
        future_poses = torch.tensor(np.array(future_poses), dtype=torch.float32)

        rtn_dict = dict(
            seq_input_img=input_img,  # torch.Size([N, 6, 128, 256])
            seq_future_poses=future_poses,  # torch.Size([N, num_pts, 3])
            seq_sample_path=seq_sample_path,
            # camera_intrinsic=torch.tensor(seq_samples[0]['camera_intrinsic']),
            # camera_extrinsic=torch.tensor(seq_samples[0]['camera_extrinsic']),
            # camera_translation_inv=torch.tensor(seq_samples[0]['camera_translation_inv']),
            # camera_rotation_matrix_inv=torch.tensor(seq_samples[0]['camera_rotation_matrix_inv']),
        )

        # For DEMO
        if self.return_origin:
            origin_imgs = origin_imgs[seq_start_idx: seq_end_idx]
            origin_imgs = [torch.tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))[None] for img in origin_imgs]
            origin_imgs = torch.cat(origin_imgs, dim=0)  # N, H_ori, W_ori, 3
            rtn_dict['origin_imgs'] = origin_imgs

        return rtn_dict
