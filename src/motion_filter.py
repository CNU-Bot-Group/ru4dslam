import torch
import lietorch

import src.geom.projective_ops as pops
from src.modules.droid_net import CorrBlock
from src.utils.mono_priors.metric_depth_estimators import get_metric_depth_estimator, predict_metric_depth
from src.utils.datasets import load_metric_depth, load_img_feature
from src.utils.mono_priors.img_feature_extractors import predict_img_features, get_feature_extractor

class MotionFilter:
    """ This class is used to filter incoming frames and extract features 
        mainly inherited from DROID-SLAM
    """

    def __init__(self, net, video, cfg, thresh=2.5, device="cuda:0"):
        self.cfg = cfg
        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update

        self.video = video
        self.thresh = thresh
        self.device = device

        self.count = 0

        # mean, std for image normalization
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        
        self.uncertainty_aware = cfg['tracking']["uncertainty_params"]['activate']
        self.save_dir = cfg['data']['output'] + '/' + cfg['scene']
        self.metric_depth_estimator = get_metric_depth_estimator(cfg)
        if cfg['mapping']["uncertainty_params"]['activate']:
            # If mapping needs dino features, we still need feature extractor
            self.feat_extractor = get_feature_extractor(cfg)

    @torch.amp.autocast('cuda',enabled=True)
    def __context_encoder(self, image):
        """ context features """
        net, inp = self.cnet(image).split([128,128], dim=2)
        return net.tanh().squeeze(0), inp.relu().squeeze(0)

    @torch.amp.autocast('cuda',enabled=True)
    def __feature_encoder(self, image):
        """ features for correlation volume """
        return self.fnet(image).squeeze(0)

    @torch.amp.autocast('cuda',enabled=True)
    @torch.no_grad()
    def track(self, tstamp, image, intrinsics=None):
        """ main update operation - run on every frame in video """

        Id = lietorch.SE3.Identity(1,).data.squeeze()
        ht = image.shape[-2] // self.video.down_scale
        wd = image.shape[-1] // self.video.down_scale

        # normalize images
        inputs = image[None, :, :].to(self.device)
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)

        # extract features
        gmap = self.__feature_encoder(inputs)

        force_to_add_keyframe = False

        ### always add first frame to the depth video ###
        if self.video.counter.value == 0:
            net, inp = self.__context_encoder(inputs[:,[0]])
            self.net, self.inp, self.fmap = net, inp, gmap
            mono_depth = predict_metric_depth(self.metric_depth_estimator,tstamp,image,self.cfg,self.device)
            if self.uncertainty_aware:
                dino_features = predict_img_features(self.feat_extractor,tstamp,image,self.cfg,self.device)
            else:
                dino_features = None
                if self.cfg['mapping']["uncertainty_params"]['activate']:
                    # If mapping needs dino features, we predict here and store the value in local disk
                    _ = predict_img_features(self.feat_extractor,tstamp,image,self.cfg,self.device)
            self.video.append(tstamp, image[0], Id, 1.0, mono_depth, intrinsics / float(self.video.down_scale), gmap, net[0,0], inp[0,0], dino_features)
        ### only add new frame if there is enough motion ###
        else:                
            # index correlation volume
            coords0 = pops.coords_grid(ht, wd, device=self.device)[None,None]
            corr = CorrBlock(self.fmap[None,[0]], gmap[None,[0]])(coords0)

            # approximate flow magnitude using 1 update iteration
            _, delta, weight = self.update(self.net[None], self.inp[None], corr)

            if self.cfg['tracking']['force_keyframe_every_n_frames'] > 0:
                # Actually, tstamp is the frame idx
                last_tstamp = self.video.timestamp[self.video.counter.value-1]
                force_to_add_keyframe = (tstamp - last_tstamp) >= self.cfg['tracking']['force_keyframe_every_n_frames']


            # check motion magnitue / add new frame to video
            if delta.norm(dim=-1).mean().item() > self.thresh or force_to_add_keyframe:
                self.count = 0
                net, inp = self.__context_encoder(inputs[:,[0]])
                self.net, self.inp, self.fmap = net, inp, gmap
                mono_depth = predict_metric_depth(self.metric_depth_estimator,tstamp,image,self.cfg,self.device)
                if self.uncertainty_aware:
                    dino_features = predict_img_features(self.feat_extractor,tstamp,image,self.cfg,self.device)
                else:
                    dino_features = None
                    if self.cfg['mapping']["uncertainty_params"]['activate']:
                        # if mapping needs dino features, we predict here and store the value in local disk
                        _ = predict_img_features(self.feat_extractor,tstamp,image,self.cfg,self.device)
                self.video.append(tstamp, image[0], None, None, mono_depth, intrinsics / float(self.video.down_scale), gmap, net[0], inp[0], dino_features)

            else:
                self.count += 1

        return force_to_add_keyframe

    @torch.no_grad()
    def get_img_feature(self, tstamp, image, suffix=''):
        dino_features = predict_img_features(self.feat_extractor,tstamp,image,self.cfg,self.device,suffix=suffix)
        return dino_features
