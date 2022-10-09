import os
import random
import sys
from os.path import join as pjoin
import numpy as np
import torch
import time
from torch.utils.data import Dataset

BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)
sys.path.insert(0, pjoin(BASEPATH, '..'))
from utils.animation_data import AnimationData
from utils.load_skeleton import Skel

import math

content_types = ['walk', 'run', 'jump', 'kick', 'punch', 'trans']


def rotation_from_quaternion(quaternion, separate=False):
    if 1 - abs(quaternion[0]) < 1e-8:
        axis = torch.tensor([1.0, 0.0, 0.0], device=quaternion.device)
        angle = 0.0
    else:
        s = math.sqrt(1 - quaternion[0] * quaternion[0])
        axis = quaternion[1:4] / s
        angle = 2 * math.acos(quaternion[0])
    return (axis, angle) if separate else axis * angle


def quaternion_inverse(quaternion):
    """Return inverse of quaternion.
    >>> q0 = random_quaternion()
    >>> q1 = quaternion_inverse(q0)
    >>> numpy.allclose(quaternion_multiply(q0, q1), [1, 0, 0, 0])
    True
    """
    q = torch.tensor(quaternion, dtype=torch.float64, device=quaternion.device)
    torch.neg(q[1:], out=q[1:])
    return q / torch.dot(q, q)


def quaternion_multiply(quaternion1, quaternion0):
    """Return multiplication of two quaternions.
    >>> q = quaternion_multiply([4, 1, -2, 3], [8, -5, 6, 7])
    >>> numpy.allclose(q, [28, -44, -14, 48])
    True
    """
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return torch.tensor([
        -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=torch.float64)


def multi_quat_diff(nq1, nq0):
    """return the relative quaternions q1-q0 of N joints"""

    nq_diff = torch.zeros_like(nq0)
    for i in range(nq1.shape[0] // 4):
        ind = slice(4 * i, 4 * i + 4)
        q1 = nq1[ind]
        q0 = nq0[ind]
        nq_diff[ind] = quaternion_multiply(q1, quaternion_inverse(q0))
    return nq_diff


def get_angvel_fd(prev_bquat, cur_bquat, dt):
    q_diff = multi_quat_diff(cur_bquat, prev_bquat)
    n_joint = q_diff.shape[0] // 4
    body_angvel = torch.zeros(n_joint * 3, device=prev_bquat.device)
    for i in range(n_joint):
        body_angvel[3 * i: 3 * i + 3] = rotation_from_quaternion(q_diff[4 * i: 4 * i + 4]) / dt
    return body_angvel


def normalize_motion(motion, mean_pose, std_pose):
    """
    inputs:
    motion: (V, C, T) or (C, T)
    mean_pose: (C, 1)
    std_pose: (C, 1)
    """
    return (motion - mean_pose) / std_pose

# NormData(prefix + "_" + key, pre_computed, raw, extra_data_dir, keep_raw=(key != "style2d"))
class NormData:
    def __init__(self, name, pre_computed, raw, data_dir, keep_raw=False):  # data_dir stores the .npz
        '''
        :param name: string, prefix + "_" + key
        :param pre_computed: boolean
        :param raw: list, containing 1452 motion clips (content/style3d/style2d)
        :param data_dir: string, './data/xia_norm'
        :param keep_raw: boolean, true if not processing style2d
        '''
        """
        raw:
        - nrot: N * [J * 4 + 4, T], raw=content 1452*128*32
        - rfree: N * [J * 3 + 4, T]?, raw=style3d 1452*64*32
        - proj: N * [V, J * 2, T]?, raw=style2d 1452*10*42*32
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.norm_path = os.path.join(data_dir, name + ".npz")
        # if there are mean and std values, just load them
        if os.path.exists(self.norm_path):
            norm = np.load(self.norm_path, allow_pickle=True)
            self.mean, self.std = norm['mean'], norm['std']
        # otherwise, we compute them
        else:
            # check if conflicts exist
            if pre_computed:
                assert 0, f'Error! {self.norm_path} not found!'
            '''
            raw=content 1452*128*32,data shape 1/3:(128, 46464),
                                    data shape 2/3: (46464, 128),
                                    data shape 3/3: (46464, 128)
            raw=style3d 1452*64*32, data shape 1/3: (64, 46464),
                                    data shape 2/3: (46464, 64),
                                    data shape 3/3: (46464, 64)
            raw=style2d 1452*10*42*32,  data shape 1/3: (10, 42, 46464),
                                        data shape 2/3: (10, 46464, 42),
                                        data shape 3/3: (464640, 42)
            '''
            # a list of [V, J * 2, T] / [J * 3/4 + 4, T]
            # --> [V, J * 2, sumT] / [J * 3/4 + 4, sumT]
            data = np.concatenate(raw, axis=-1)
            print("data shape 1/3:", data.shape)
            # [V, sumT, J * 2] / [sumT, J * 3/4 + 4]
            data = data.swapaxes(-1, -2)
            print("data shape 2/3:", data.shape)
            # [V * sumT, J * 2] / [sumT, J * 3/4 + 4]
            data = data.reshape((-1, data.shape[-1]))
            print("data shape 3/3:", data.shape)

            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0)
            self.std[np.where(self.std == 0)] = 1e-9
            os.makedirs(data_dir, exist_ok=True)
            np.savez(self.norm_path, mean=self.mean, std=self.std)
            print("mean and std saved at {}".format(self.norm_path))

        self.mean = torch.tensor(self.mean, dtype=torch.float, device=self.device).unsqueeze(-1)
        self.std = torch.tensor(self.std, dtype=torch.float, device=self.device).unsqueeze(-1)  # [C, 1]
        if keep_raw:
            self.raw = raw
            self.norm = {}
            for i in range(len(self.raw)):
                self.raw[i] = torch.tensor(self.raw[i], dtype=torch.float, device=self.device)

    def get_raw(self, index):
        return self.raw[index]

    def get_norm(self, index):
        if index not in self.norm:
            self.norm[index] = normalize_motion(self.raw[index], self.mean, self.std)
        return self.norm[index]

    def normalize(self, raw):
        return normalize_motion(raw, self.mean, self.std)

    def normalize_cond_train(self, raw):

        raw_split = raw.view(10, 128, 22)
        for i in range(10):
            raw_split[i, :, :] = (raw_split[i, :, :] - self.mean.cpu()) / self.std.cpu()
        raw = raw_split.view(1280, 22)
        return raw

    def get_mean_std(self):
        return self.mean, self.std


class VelocityData:
    def __init__(self, content):
        self.data = []
        start = time.time()
        for i in range(len(content.raw)):
            motion = content.raw[i]
            normed = normalize_motion(motion, content.mean, content.std)
            root = normed[-4:, :]
            data = [torch.zeros(3, device=torch.device('cuda'))]
            for k in range(motion.shape[1] - 1):
                v_ = get_angvel_fd(root[:, k], root[:, k + 1], 1)
                data.append(v_)
            data = torch.stack(data).permute(1, 0)
            self.data.append(data)
            '''
            data = [torch.zeros(96, device=torch.device('cuda'))]
            for k in range(motion.shape[1]-1):
                v = None
                for j in range(0, motion.shape[0], 4):
                    v_ = get_angvel_fd(normed[j:j+4, k], normed[j:j+4,k+1], 1)
                    if v is None:
                        v = v_
                    else:
                        v = torch.cat([v, v_], dim=0)
                data.append(v)
            print(time.time() - start)
            '''
            # data = torch.stack(data)
            # data = data.permute(1,0)
            # self.data.append(data)

    def get_angvel(self, index):
        return self.data[index]


class MotionNorm(Dataset):
    def __init__(self, subset_name, data_path=None, extra_data_dir=None):
        super(MotionNorm, self).__init__()
        self.is_trainset = subset_name == "train"
        # np.random.seed(2020)

        # 'self.skel' is unclear about its functionality
        self.skel = Skel()  # TD: add config

        # 'self.num_views' is the number of view_angles
        self.num_views = 10

        # each motion clip has 32 frames
        # each frame represents how many seconds is defined in '../utils/animation_data.py: class AnimationData'
        self.n_frames = 32

        dataset = np.load(data_path, allow_pickle=True)[subset_name].item()
        # print(type(dataset)) dict

        # there are 1452 motion clips; clip details can be found in './data/xia.info'
        motions, labels, metas = dataset["motion"], dataset["style"], dataset["meta"]
        # print(np.array(motions).shape) (1452, 32, 132)
        # print(np.array(labels).shape) (1452,)
        # print(type(metas))  # dict
        # print(metas.keys())  # dict_keys(['style', 'phase'])

        self.label_i = labels

        # thus, we get dataset length by directly use 'labels'; comment below and use its next line
        # self.len = len(self.label_i)
        self.len = len(labels)

        # 'self.metas' is a dict, keys are 'style', 'content', 'phase', each key has 1452 values
        # each key and its each value corresponds to a motion clip
        # what is phase?
        self.metas = [{key: metas[key][i] for key in metas.keys()} for i in range(self.len)]

        self.motion, self.foot = [], []

        content, style3d, style2d = [], [], []

        self.labels = []
        self.data_dict = {}
        self.diff_labels_dict = {}

        for i, motion in enumerate(motions):    # there are 1452 motion clips
            # each motion clip, denoted as 'motion' here, is a  32 * 132 ndarray
            # print(motion.shape) (32, 132)
            # print(type(motion)) # <class 'numpy.ndarray'>
            label = labels[i]   # get the 'motion' label

            # it seems 'AnimationData' converts the ndarray into positions and rotations?
            anim = AnimationData(motion, skel=self.skel)

            # we need 'self.labels' to be a set of labels
            # for each label in 'self.labels', we maintain a dictionary to record its 'motion' index
            if label not in self.labels:  # if it is a new label
                self.labels.append(label)  # add it to the set: 'self.labels'
                self.data_dict[label] = []  # create a list for this label in data_dict

            # data_dict is formed as {label:a list of motion's index}
            self.data_dict[label].append(i)
            # record the motion 'anim' into 'self.motion' list
            self.motion.append(anim)
            # record 'anim' foot into 'self.foot' list
            self.foot.append(anim.get_foot_contact(transpose=True))  # [4, T]
            # record 'anim' content into 'content' list
            # 'anim.get_content_input()' is [31 * 4 + 3 + 1, 32]
            # print(anim.get_content_input().shape)   # (128, 32)
            content.append(anim.get_content_input())
            # record 'anim' style in 3D to 'style3d' list
            # don't know what 'anim.get_style3d_input()' returns
            # print(anim.get_style3d_input().shape)   # (64, 32)
            style3d.append(anim.get_style3d_input())

            # the following code renders 3D style information to 2D when given a view_angle
            view_angles, scales = [], []
            for v in range(self.num_views):
                # just obtain a random view_angle
                view_angles.append(self.random_view_angle())
                # and a random scale for rendering
                scales.append(self.random_scale())
            # record 'anim.get_projections(view_angles, scales)' into 'style2d' list
            # don't know what is 'anim.get_projections(view_angles, scales)' actually
            style2d.append(anim.get_projections(view_angles, scales))

        # for each label in the set 'self.labels', we use 'self.diff_labels_dict' to keep the remaining other labels
        for x in self.labels:
            # for each label 'x', store all different other labels into 'diff_labels_dict[x]'
            self.diff_labels_dict[x] = [y for y in self.labels if y != x]

        # 'norm_cfg' is used for a path of saving mean/std
        norm_cfg = {  # specify the prefix of mean/std
            "train":
                {"content": None, "style3d": None, "style2d": None},
            # will be named automatically as "train_content", etc.
            "test":
                {"content": "train", "style3d": "train", "style2d": "train"},
            "trainfull":
                {"content": "train", "style3d": "train", "style2d": "train"}
        }

        # 'norm_data' is a list of normalized motion clips by their mean and std
        norm_data = []
        for key, raw in zip(["content", "style3d", "style2d"], [content, style3d, style2d]):
            '''
            key='content' (a string), raw=content (a list, 1452*128*32)
            key='style3d' (a string), raw=style3d (a list, 1452*64*32)
            key='style2d' (a string), raw=style2d (a list, 1452*10*42*32)
            '''
            prefix = norm_cfg[subset_name][key]
            pre_computed = prefix is not None
            if prefix is None:
                prefix = subset_name
            norm_data.append(
                NormData(prefix + "_" + key, pre_computed, raw, extra_data_dir, keep_raw=(key != "style2d"))
            )
        # self.content, self.style3d, self.style2d are all normalized by mean and std
        self.content, self.style3d, self.style2d = norm_data
        # self.angvel = VelocityData(self.content)
        self.mean, self.std = self.content.get_mean_std()
        self.angvel = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rand = random.SystemRandom()

    def concat_sequence(self, seqlen, data):
        """
        Concatenates a sequence of features to one.
        """
        nn, n_timesteps, n_feats = data.shape
        L = n_timesteps - (seqlen - 1)
        inds = np.zeros((L, seqlen)).astype(int)

        # create indices for the sequences we want
        rng = np.arange(0, n_timesteps)
        for ii in range(0, seqlen):
            inds[:, ii] = np.transpose(rng[ii:(n_timesteps - (seqlen - ii - 1))])

            # slice each sample into L sequences and store as new samples
        cc = data[:, inds, :].clone()

        # print ("cc: " + str(cc.shape))

        # reshape all timesteps and features into one dimention per sample
        dd = cc.reshape((nn, L, seqlen * n_feats))
        # print ("dd: " + str(dd.shape))
        return dd

    @staticmethod
    def random_view_angle():
        return (0, -np.pi / 2 + float(np.random.rand(1)) * np.pi, 0)

    @staticmethod
    def random_scale():
        return float(np.random.rand(1)) * 0.4 + 0.8

    def __len__(self):
        return self.len  # 1452

    def __getitem__(self, index):
        label = self.label_i[index]  # get label for the motion
        if len(self.diff_labels_dict[label]) == 0:
            l_diff = label  # get all different labels in a list
        else:
            l_diff = self.rand.choice(self.diff_labels_dict[label])
        index_same = self.rand.choice(
            self.data_dict[label])  # randomly choose an index of motions that have the same label
        index_diff = self.rand.choice(
            self.data_dict[l_diff])  # randomly choose an index of motions that have different label
        mean_pose, std_pose = self.content.get_mean_std()

        data = {"label": label,
                "meta": self.metas[index],
                "foot_contact": self.foot[index],  # for foot sliding problem
                "content": self.content.get_norm(index),
                "style3d": self.style3d.get_norm(index),
                "contentraw": self.content.get_raw(index),  # for visualization
                "style3draw": self.style3d.get_raw(index),  # positions are used as the recon target
                "same_style3d": self.style3d.get_norm(index_same),
                "diff_style3d": self.style3d.get_norm(index_diff),
                "mean": mean_pose,
                "std": std_pose,
                "content_label": content_types.index(self.metas[index]["content"]),  # comment this line if use bfa data
                "fake_label": len(content_types),
                # "angvel": None#self.angvel.get_angvel(index)
                # "cond": cond,
                }

        for idx, key in zip([index, index_same, index_diff], ["style2d", "same_style2d", "diff_style2d"]):
            raw = self.motion[idx].get_projections((self.random_view_angle(),), (self.random_scale(),))[0]
            raw = torch.tensor(raw, dtype=torch.float, device=self.device)
            if key == "style2d":
                data[key + "raw"] = raw
            data[key] = self.style2d.normalize(raw)

        return data
