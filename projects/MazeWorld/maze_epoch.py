import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from src.dataloader import segment_iterator
from src.utils import Logger, log_progress, log_debug, log_warn, log_fatal
from src.utils import custom_load_model, noam_scheduler, LinearScheduler
from src.utils import Configure, DistStatistics, rewards2go
from src.utils import EpochManager, GeneratorBase
from src.utils import noam_scheduler, LinearScheduler
from src.utils import weighted_loss, img_pro, img_post
from src.dataloader import MazeDataSet, PrefetchDataLoader, MazeTaskDataSet, MazeDataSetShort

def string_mean_var(downsample_length, res):
    string=""
    for i, (xm,xb) in enumerate(zip(res["mean"], res["bound"])):
        string += f'{downsample_length * i}\t{xm}\t{xb}\n'
    return string

@EpochManager
class MazeEpochVAE:
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        if(self.is_training):
            self.logger_keys = ["learning_rate", 
                        "noise",
                        "kl_weight",
                        "reconstruction_error",
                        "kl_divergence"]
            self.stat = DistStatistics(*self.logger_keys[3:])
            self.lr = self.config.lr_vae
            self.lr_decay_interval = self.config.lr_vae_decay_interval
            self.lr_start_step = self.config.lr_vae_start_step
        else:
            self.logger_keys = ["reconstruction_error", 
                        "kl_divergence"]
            self.stat = DistStatistics(*self.logger_keys)

    def preprocess(self):
        if(self.is_training):
            self.sigma_scheduler = LinearScheduler(self.config.sigma_scheduler, 
                                                   self.config.sigma_value)
            self.lambda_scheduler = LinearScheduler(self.config.lambda_scheduler, 
                                                    self.config.lambda_value)
        # use customized dataloader
        self.dataloader = PrefetchDataLoader(
            MazeDataSet(self.config.data_path, self.config.seq_len_vae, verbose=self.main, max_maze=self.max_maze), 
            batch_size=self.config.batch_size_vae,
            rank=self.rank,
            world_size=self.world_size
            )
            
    def valid_epoch(self, epoch_id): # Add epoch control for VAE training
        if(self.config.has_attr('epoch_vae_stop')):
            if(epoch_id >= self.config.epoch_vae_stop):
                return False
        return True

    def compute(self, obs_arr, behavior_actid_arr, label_actid_arr, 
                behavior_act_arr, label_act_arr, 
                rew_arr, bev_arr,
                epoch_id=-1, 
                batch_id=-1):
        """
        Defining the computation function for each batch
        """
        if(self.is_training):
            assert self.optimizer is not None, "optimizer is required for training"

        losses = []
        seq_len = self.config.seq_len_vae
        for sub_idx, seg_obs in segment_iterator(
                            self.config.seq_len_vae, self.config.seg_len_vae,
                            self.device, obs_arr):
            # Permute (B, T, H, W, C) to (B, T, C, H, W)
            seg_obs = seg_obs.permute(0, 1, 4, 2, 3)
            seg_obs = seg_obs.contiguous()

            if(self.is_training):
                sigma = self.sigma_scheduler()
            else:
                sigma = 0
            loss = self.model.module.vae_loss(
                    seg_obs,
                    _sigma=sigma,
                    seq_len=seq_len)
            losses.append(loss)
            if(self.is_training):
                syn_loss = (loss["Reconstruction-Error"] + self.lambda_scheduler() * loss["KL-Divergence"]) / loss["count"]
                if(self.scaler is not None):
                    self.scaler.scale(syn_loss).backward()
                else:
                    syn_loss.backward()
                self.stat.gather(self.device,
                    reconstruction_error = loss["Reconstruction-Error"] / loss["count"],
                    kl_divergence = loss["KL-Divergence"] / loss["count"],
                    count = loss["count"])
        if(self.is_training):
            stat_res = self.stat()
            if(self.logger is not None):
                self.logger(self.optimizer.param_groups[0]['lr'],
                            self.sigma_scheduler(), 
                            self.lambda_scheduler(), 
                            stat_res["reconstruction_error"]["mean"], 
                            stat_res["kl_divergence"]["mean"],
                            epoch=epoch_id,
                            iteration=batch_id)
            # update the scheduler
            self.sigma_scheduler.step()
            self.lambda_scheduler.step()
        else:
            self.stat.gather(self.device,
                    reconstruction_error=loss["Reconstruction-Error"] / loss["count"], 
                    kl_divergence=loss["KL-Divergence"] / loss["count"], 
                    count=loss["count"])
            
        
    def epoch_end(self, epoch_id):
        if(not self.is_training):
            stat_res = self.stat()
            if(self.logger is not None):
                self.logger(stat_res["reconstruction_error"]["mean"], 
                        stat_res["kl_divergence"]["mean"], 
                        epoch=epoch_id)

@EpochManager
class MazeEpochCausal: # the computer
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.DataType=MazeDataSet
        if (self.config.has_attr("is_visualize")):
            self.is_visualize = self.config.is_visualize  
        else:
            self.is_visualize = False
        print(f"is_visualize: {self.is_visualize}") 
        
        if (self.config.has_attr("max_maze")):
            self.max_maze = self.config.max_maze
        else:
            self.max_maze = None
        if(self.is_training):
            self.logger_keys = ["learning_rate", 
                        "loss_worldmodel_raw",
                        "loss_worldmodel_latent",
                        "loss_policymodel"]
            self.stat = DistStatistics(*self.logger_keys[1:])
            self.lr = self.config.lr_causal
            self.lr_decay_interval = self.config.lr_causal_decay_interval
            self.lr_start_step = self.config.lr_causal_start_step
            self.reduce_dim = 1
            
        else:
            if not os.path.exists(self.config.output):
                os.makedirs(self.config.output)

            self.logger_keys = ["validate_worldmodel_raw",
                        "validate_worldmodel_latent",
                        "validate_policymodel"]
            self.stat = DistStatistics(*self.logger_keys)
            if(self.config.has_attr("downsample_length")):
                self.downsample_length = self.config.downsample_length
            else:
                self.downsample_length = 100
            self.reduce_dim = None
            
    def valid_epoch(self, epoch_id): # Add epoch control for VAE training
        if(self.config.has_attr('epoch_causal_start')):
            if(epoch_id < self.config.epoch_causal_start):
                return False
        return True

    def preprocess(self):
        # use customized dataloader
        self.dataloader = PrefetchDataLoader(
            MazeDataSet(self.config.data_path, self.config.seq_len_causal, verbose=self.main, max_maze=self.max_maze), #  
            batch_size=self.config.batch_size_causal,
            rank=self.rank,
            world_size=self.world_size
            )

    def compute(self, cmd_arr, obs_arr, behavior_actid_arr, label_actid_arr, 
                behavior_act_arr, label_act_arr, 
                rew_arr, # bev_arr,
                epoch_id=-1, 
                batch_id=-1):
        """
        Defining the computation function for each batch
        """
        if(self.is_training):
            assert self.optimizer is not None, "optimizer is required for training"

        losses = []
        current_prediction_observations = []
        for sub_idx, seg_cmd, seg_obs, seg_behavior_act, seg_label_act in segment_iterator(
                                self.config.seq_len_causal, self.config.seg_len_causal, self.device, 
                                cmd_arr, (obs_arr, 1), behavior_actid_arr, label_actid_arr):

            # Permute (B, T, H, W, C) to (B, T, C, H, W)
            seg_obs = seg_obs.permute(0, 1, 4, 2, 3)
            seg_obs = seg_obs.contiguous()

            loss, obs_pred, __, __ = self.model.module.sequential_loss(
                                    prompts = seg_cmd,
                                    observations = seg_obs,
                                    tags = None, 
                                    behavior_actions = seg_behavior_act,
                                    rewards = None,
                                    label_actions = seg_label_act, 
                                    state_dropout=0.20,
                                    use_loss_weight=self.is_training,
                                    is_training=self.is_training,
                                    reduce_dim=self.reduce_dim,) 
            
            if self.is_visualize and sub_idx % 20 == 0:
                current_prediction_observations.append(obs_pred)
            losses.append(loss)
            if(self.is_training):
                syn_loss = (self.config.lossweight_worldmodel_latent * loss["wm-latent"]
                        + self.config.lossweight_worldmodel_raw * loss["wm-raw"]
                        + self.config.lossweight_policymodel * loss["pm"]
                        + self.config.lossweight_l2 * loss["causal-l2"])
                if(self.scaler is not None):
                    self.scaler.scale(syn_loss).backward()
                else:
                    syn_loss.backward()
                self.stat.gather(self.device,
                                loss_worldmodel_raw = loss["wm-raw"] / loss["count_wm"],
                                loss_worldmodel_latent = loss["wm-latent"] / loss["count_wm"],
                                loss_policymodel = loss["pm"] / loss["count_pm"])
        
        if self.is_visualize:        
            current_prediction_observations = torch.cat(current_prediction_observations, dim=1)
        
        if(self.is_training):
            stat_res = self.stat()
            if(self.logger is not None):
                self.logger(self.optimizer.param_groups[0]['lr'],
                            stat_res["loss_worldmodel_raw"]["mean"], 
                            stat_res["loss_worldmodel_latent"]["mean"],
                            stat_res["loss_policymodel"]["mean"],
                            epoch=epoch_id,
                            iteration=batch_id)
        else:
            loss_wm_r = []
            loss_wm_l = []
            loss_pm = []
            counts = []

            loss_wm_r = torch.cat([loss["wm-raw"] / loss["count_wm"] for loss in losses], dim=1)
            loss_wm_l = torch.cat([loss["wm-latent"] / loss["count_wm"] for loss in losses], dim=1)
            loss_pm = torch.cat([loss["pm"] / loss["count_pm"] for loss in losses], dim=1)
            counts = torch.cat([loss["count_pm"] for loss in losses], dim=1)

            bsz = loss_wm_r.shape[0]
            seg_num = loss_wm_l.shape[1] // self.downsample_length
            valid_seq_len = seg_num * self.downsample_length

            loss_wm_r = torch.mean(loss_wm_r[:, :valid_seq_len].view(bsz, seg_num, -1), dim=-1)
            loss_wm_l = torch.mean(loss_wm_l[:, :valid_seq_len].view(bsz, seg_num, -1), dim=-1)
            loss_pm = torch.mean(loss_pm[:, :valid_seq_len].view(bsz, seg_num, -1), dim=-1)
            counts = torch.mean(counts[:, :valid_seq_len].view(bsz, seg_num, -1), dim=-1)
            for i in range(bsz):
                self.stat.gather(self.device,
                        validate_worldmodel_raw=loss_wm_r[i], 
                        validate_worldmodel_latent=loss_wm_l[i], 
                        validate_policymodel=loss_pm[i],
                        count=counts[i])



            if(self.is_visualize):
                import os 
                import cv2
                if(self.current_prediction_observations is not None):
                    img_folder = os.path.join(self.config.output, f"visualize{epoch_id}_{batch_id}")
                    if not os.path.exists(img_folder):
                        os.makedirs(img_folder)
                    video_filename = os.path.join(img_folder, f"visualize{epoch_id}_{batch_id}.avi")
                    Nt = self.current_prediction_observations.shape[1]
                    imgs = img_post(self.current_prediction_observations.cpu().numpy())
                    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
                    frame_height, frame_width = imgs[0][0].shape[-2:]
                    video_writer = cv2.VideoWriter(video_filename, fourcc, 10.0, (frame_width * 2, frame_height))
                    for i in range(Nt):
                        img = imgs[0][i]
                        pred_frame = img.transpose(1, 2, 0)
                        real_frame = seg_obs[0, i+1].cpu().numpy().transpose(1, 2, 0)
                        rotated_real = cv2.rotate(real_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        rotated_pred = cv2.rotate(pred_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        concatenated_img = np.hstack((rotated_real, rotated_pred))
                        concatenated_img = np.clip(concatenated_img, 0, 255).astype(np.uint8)
                        if i % 20 == 0:
                            cv2.imwrite(os.path.join(img_folder, f"T_{i}.png"), concatenated_img)
                        video_writer.write(concatenated_img)
                    video_writer.release()
                    print(f"Saved video with {len(imgs[0])} frames to {video_filename}")
                else:
                    print("current_prediction_observations is None")
                    
    def epoch_end(self, epoch_id):
        if(not self.is_training):
            stat_res = self.stat()
            if(self.logger is not None):
                self.logger(stat_res["validate_worldmodel_raw"]["mean"], 
                        stat_res["validate_worldmodel_latent"]["mean"], 
                        stat_res["validate_policymodel"]["mean"],
                        epoch=epoch_id)
            print(f"logger end epoch: {epoch_id}")
            if(self.extra_info is not None):
                if(self.extra_info.lower() == 'validate' and self.main):
                    if not os.path.exists(self.config.output):
                        os.makedirs(self.config.output)
                    print(f"Saving the validation results to {self.config.output}")
                    for key_name in stat_res:
                        print(f"key_name: {key_name}")
                        res_text = string_mean_var(self.downsample_length, stat_res[key_name])
                        file_path = f'{self.config.output}/result_{key_name}.txt'
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        with open(file_path, 'w') as f_model:
                            f_model.write(res_text)


@EpochManager
class MazeEpochCausalShort: # the computer
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.DataType=MazeDataSet
        if (self.config.has_attr("is_visualize")):
            self.is_visualize = self.config.is_visualize  
        else:
            self.is_visualize = False
        print(f"is_visualize: {self.is_visualize}") 
        
        if (self.config.has_attr("max_maze")):
            self.max_maze = self.config.max_maze
        else:
            self.max_maze = None
        if(self.is_training):
            self.logger_keys = ["learning_rate", 
                        "loss_worldmodel_raw",
                        "loss_worldmodel_latent",
                        "loss_policymodel"]
            self.stat = DistStatistics(*self.logger_keys[1:])
            self.lr = self.config.lr_causal
            self.lr_decay_interval = self.config.lr_causal_decay_interval
            self.lr_start_step = self.config.lr_causal_start_step
            self.reduce_dim = 1
            
        else:
            if not os.path.exists(self.config.output): #   线程冲突
                os.makedirs(self.config.output)


            self.logger_keys = ["validate_worldmodel_raw",
                        "validate_worldmodel_latent",
                        "validate_policymodel"]
            self.stat = DistStatistics(*self.logger_keys)
            if(self.config.has_attr("downsample_length")):
                self.downsample_length = self.config.downsample_length
            else:
                self.downsample_length = 100
            self.reduce_dim = None
            
    def valid_epoch(self, epoch_id): # Add epoch control for VAE training
        if(self.config.has_attr('epoch_causal_start')):
            if(epoch_id < self.config.epoch_causal_start):
                return False
        return True

    def preprocess(self):
        # use customized dataloader
        self.dataloader = PrefetchDataLoader(
            MazeDataSetShort(self.config.data_path, self.config.seq_len_causal, verbose=self.main, max_maze=self.max_maze), #  
            batch_size=self.config.batch_size_causal,
            rank=self.rank,
            world_size=self.world_size
            )

    def compute(self, cmd_arr, obs_arr, behavior_actid_arr, label_actid_arr, 
                behavior_act_arr, label_act_arr, 
                rew_arr, # bev_arr,
                epoch_id=-1, 
                batch_id=-1):
        """
        Defining the computation function for each batch
        """
        if(self.is_training):
            assert self.optimizer is not None, "optimizer is required for training"

        losses = []
        current_prediction_observations = []
        for sub_idx, seg_cmd, seg_obs, seg_behavior_act, seg_label_act in segment_iterator(
                                self.config.seq_len_causal, self.config.seg_len_causal, self.device, 
                                cmd_arr, (obs_arr, 1), behavior_actid_arr, label_actid_arr):

            seg_obs = seg_obs.permute(0, 1, 4, 2, 3)
            seg_obs = seg_obs.contiguous()

            loss, obs_pred, __, __ = self.model.module.sequential_loss(
                                    prompts = seg_cmd,
                                    observations = seg_obs,
                                    tags = None, 
                                    behavior_actions = seg_behavior_act,
                                    rewards = None,
                                    label_actions = seg_label_act, 
                                    state_dropout=0.20,
                                    use_loss_weight=self.is_training,
                                    is_training=self.is_training,
                                    reduce_dim=self.reduce_dim,) 
            
            if self.is_visualize and sub_idx % 20 == 0:
                current_prediction_observations.append(obs_pred)
            # self.current_prediction_actions = a_pred
            # self.current_cache = cache
            losses.append(loss)
            if(self.is_training):
                syn_loss = (self.config.lossweight_worldmodel_latent * loss["wm-latent"]
                        + self.config.lossweight_worldmodel_raw * loss["wm-raw"]
                        + self.config.lossweight_policymodel * loss["pm"]
                        + self.config.lossweight_l2 * loss["causal-l2"])
                if(self.scaler is not None):
                    self.scaler.scale(syn_loss).backward()
                else:
                    syn_loss.backward()
                self.stat.gather(self.device,
                                loss_worldmodel_raw = loss["wm-raw"] / loss["count_wm"],
                                loss_worldmodel_latent = loss["wm-latent"] / loss["count_wm"],
                                loss_policymodel = loss["pm"] / loss["count_pm"])
        
        if self.is_visualize:        
            current_prediction_observations = torch.cat(current_prediction_observations, dim=1)
        
        if(self.is_training):
            stat_res = self.stat()
            if(self.logger is not None):
                self.logger(self.optimizer.param_groups[0]['lr'],
                            stat_res["loss_worldmodel_raw"]["mean"], 
                            stat_res["loss_worldmodel_latent"]["mean"],
                            stat_res["loss_policymodel"]["mean"],
                            epoch=epoch_id,
                            iteration=batch_id)
        else:
            loss_wm_r = []
            loss_wm_l = []
            loss_pm = []
            counts = []

            loss_wm_r = torch.cat([loss["wm-raw"] / loss["count_wm"] for loss in losses], dim=1)
            loss_wm_l = torch.cat([loss["wm-latent"] / loss["count_wm"] for loss in losses], dim=1)
            loss_pm = torch.cat([loss["pm"] / loss["count_pm"] for loss in losses], dim=1) 
            counts = torch.cat([loss["count_pm"] for loss in losses], dim=1)

            bsz = loss_wm_r.shape[0]
            seg_num = loss_wm_l.shape[1] // self.downsample_length
            valid_seq_len = seg_num * self.downsample_length

            loss_wm_r = torch.mean(loss_wm_r[:, :valid_seq_len].view(bsz, seg_num, -1), dim=-1)
            loss_wm_l = torch.mean(loss_wm_l[:, :valid_seq_len].view(bsz, seg_num, -1), dim=-1)
            loss_pm = torch.mean(loss_pm[:, :valid_seq_len].view(bsz, seg_num, -1), dim=-1)
            counts = torch.mean(counts[:, :valid_seq_len].view(bsz, seg_num, -1), dim=-1)
            for i in range(bsz):
                self.stat.gather(self.device,
                        validate_worldmodel_raw=loss_wm_r[i], 
                        validate_worldmodel_latent=loss_wm_l[i], 
                        validate_policymodel=loss_pm[i],
                        count=counts[i])



            if(self.is_visualize):
                import os 
                import cv2
                if(self.current_prediction_observations is not None):
                    img_folder = os.path.join(self.config.output, f"visualize{epoch_id}_{batch_id}")
                    if not os.path.exists(img_folder):
                        os.makedirs(img_folder)
                    video_filename = os.path.join(img_folder, f"visualize{epoch_id}_{batch_id}.avi")
                    Nt = self.current_prediction_observations.shape[1]
                    imgs = img_post(self.current_prediction_observations.cpu().numpy())
                    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
                    frame_height, frame_width = imgs[0][0].shape[-2:]
                    video_writer = cv2.VideoWriter(video_filename, fourcc, 10.0, (frame_width * 2, frame_height))
                    for i in range(Nt):
                        img = imgs[0][i]
                        pred_frame = img.transpose(1, 2, 0)
                        real_frame = seg_obs[0, i+1].cpu().numpy().transpose(1, 2, 0)
                        rotated_real = cv2.rotate(real_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        rotated_pred = cv2.rotate(pred_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        concatenated_img = np.hstack((rotated_real, rotated_pred))
                        concatenated_img = np.clip(concatenated_img, 0, 255).astype(np.uint8)
                        if i % 20 == 0:
                            cv2.imwrite(os.path.join(img_folder, f"T_{i}.png"), concatenated_img)
                        video_writer.write(concatenated_img)
                    video_writer.release()
                    print(f"Saved video with {len(imgs[0])} frames to {video_filename}")
                else:
                    print("current_prediction_observations is None")
                    
    def epoch_end(self, epoch_id):
        if(not self.is_training):
            stat_res = self.stat()
            if(self.logger is not None):
                self.logger(stat_res["validate_worldmodel_raw"]["mean"], 
                        stat_res["validate_worldmodel_latent"]["mean"], 
                        stat_res["validate_policymodel"]["mean"],
                        epoch=epoch_id)
            print(f"logger end epoch: {epoch_id}")
            if(self.extra_info is not None):
                if(self.extra_info.lower() == 'validate' and self.main):
                    if not os.path.exists(self.config.output):
                        os.makedirs(self.config.output)
                    print(f"Saving the validation results to {self.config.output}")
                    for key_name in stat_res:
                        print(f"key_name: {key_name}")
                        res_text = string_mean_var(self.downsample_length, stat_res[key_name])
                        file_path = f'{self.config.output}/result_{key_name}.txt'
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        with open(file_path, 'w') as f_model:
                            f_model.write(res_text)


            


class interactive_trajectory(GeneratorBase):
    def epoch_end(self, epoch_id):
        # MazeTaskDataSet
        pass
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key in kwargs:
            setattr(self, key, kwargs[key])
            print(f"{key}: {kwargs[key]}")
        self.output_root = self.config.output_root
        self.in_context_len = self.config.in_context_len
        self.end_position = self.config.end_position
        self.start_position = self.config.start_position
        self.record_interval = self.config.record_interval
        self.record_points = [i for i in range(self.start_position, self.end_position, self.record_interval)]
        if self.config.has_attr("max_maze"):
            self.max_maze = self.config.max_maze
        else:
            self.max_maze = None
        if self.end_position > self.config.seq_len_causal:
            assert False, "end_position should be smaller than seq_len_causal"
        


    def preprocess(self):
        self.dataloader = PrefetchDataLoader(
            MazeTaskDataSet(self.config.data_path, self.config.seq_len_causal, verbose=self.main, max_maze = self.max_maze, folder_verbose=True),
            batch_size=1, #   
            rank=self.rank,
            world_size=self.world_size
            )
        if self.output_root is not None:
            if not os.path.exists(self.output_root):
                os.makedirs(self.output_root)
                print(f"Created output root {self.output_root}")
            if self.config.data_path[-1] == "/":
                output_folder_path = os.path.join(self.output_root, self.config.data_path.split("/")[-2])
            else:
                output_folder_path = os.path.join(self.output_root, self.config.data_path.split("/")[-1])
            print(f"output folder path: {output_folder_path}")
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)
                print(f"Created output folder {output_folder_path}")
            self.output_folder_path = output_folder_path
        print(f"Preprocessed dataloader with {len(self.dataloader)} batches")

    def __call__(self, epoch_id, rank):
        import gym
        import pickle
        import cv2
        import maze_generator.mazeworld
        from maze_generator.mazeworld import MazeTaskSampler, Resampler, MazeStaticSampler
        from maze_generator.mazeworld.agents import OracleAgent
        for batch_id, (batch_data, folder_name) in enumerate(self.dataloader):
            max_steps = 10000
            n_range = (15,16)
            maze_env = gym.make("mazeworld-v2", enable_render=False, max_steps=max_steps, resolution=(128, 128))
        
            folder_name = folder_name[0] # batch size is 1
            new_task_path = batch_data[0]
            new_task = pickle.load(open(new_task_path, 'rb'))

            print(f"task: {new_task}")
            print("-----------------------------\n\n")  
            maze_env.set_task(new_task)

            done = False
            observation_list = []
            reward_list = []
            bev_list = []
            cmd_list = []
            sum_reward = 0
            
            observation, information = maze_env.reset()
            observation = np.array(observation, dtype=np.uint8)
            # (H, W, C) to (C, H, W)
            observation = np.transpose(observation, (2, 0, 1))
            command = information["command"]
            command = np.repeat(command, 256, axis=0)
            last_observation = None 
            last_action = None 
            inference_record = []
            action_record = []
            loss_record = []
            output_root = self.output_folder_path
            maze_output_folder = os.path.join(output_root, folder_name)
            if not os.path.exists(maze_output_folder):
                os.makedirs(maze_output_folder)
            output_folder = os.path.join(maze_output_folder, self.config.model_name)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            print(f"output folder: {output_folder}")
            print("-----------------------------")
            import tqdm
            K_step = 1
            cache = None
            self.model.module.reset()
            for step in range(max_steps):
                if done:
                    print(f"done at step {step}")
                    break
                cmd_string = information["command"]
                pred_obs_list, action, cache = self.model.module.generate_states_and_action(
                                command,
                                observation, 
                                future_steps=K_step,
                                history_observation=None,
                                history_action=None, 
                                history_update_memory=True, 
                                autoregression_update_memory=True,
                                cache=cache,
                                single_batch=True,
                                history_single_step=True,
                                raw_images=True,
                                need_predict_states=True,
                                need_numpy=True)
                action = action[0,0]
                inference_record.append(pred_obs_list)
                action_record.append(action)
                obs, reward, done, information = maze_env.step(action)
                command = information["command"]
                command = np.repeat(command, 256, axis=0)
                maze_env.render()
                obs = np.array(obs, dtype=np.uint8)
                obs = np.transpose(obs, (2, 0, 1))
                observation_list.append(obs)
                mse_loss = np.mean((obs - pred_obs_list[0])**2/(255*255))
                loss_record.append(mse_loss)
                last_observation = observation
                last_action = action
                observation = obs
                reward_list.append(reward)
                bev_list.append(maze_env.get_local_map()[1])
                cmd_list.append(information["command"])
                sum_reward += reward

            inference_record = np.array(inference_record)
            observation_list = np.array(observation_list)
            reward_list = np.array(reward_list)
            # save reward record to npy
            np.save(os.path.join(output_folder, "reward.npy"), reward_list)
            print(f"Saved reward to {os.path.join(output_folder, 'reward.npy')}")
            print("------------------------------")
            print(f"sum reward: {sum_reward}")
            print("------------------------------")
            import matplotlib.pyplot as plt
            mean_loss_record = []
            downsample_length = 50
            loss_record = np.array(loss_record)
            # save loss to npy
            np.save(os.path.join(output_folder, "loss.npy"), loss_record)
            print(f"Saved loss to {os.path.join(output_folder, 'loss.npy')}")
            for i in range(0, len(loss_record)):
                mean_loss_record.append(np.mean(loss_record[max(i - downsample_length, 0):min(i + downsample_length, len(loss_record))]))
            plt.plot(range(0, len(loss_record)), mean_loss_record, label="mse loss")
            plt.legend()
            plt.savefig(os.path.join(output_folder, "mse_loss.png"))
            plt.close()
            print(f"Saved mse loss plot to {os.path.join(output_folder, 'mse_loss.png')}")
            maze_env.save_trajectory(os.path.join(output_folder, "trajectory.png"))
            print(f"Saved trajectory to {os.path.join(output_folder, 'trajectory.png')}")
            maze_env.save_trajectory_npy(os.path.join(output_folder, f"trajectory.npy"))
            pickle.dump(new_task, open(os.path.join(output_folder, "task.pkl"), "wb"))
            
            video_folder = os.path.join(output_folder, f'video')
            if not os.path.exists(video_folder):
                os.makedirs(video_folder)
            video_filename = os.path.join(video_folder, f"pred_obs_video{0}.avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID') 
            frame_height, frame_width = inference_record[0].shape[-2:]
            video_writer = cv2.VideoWriter(video_filename, fourcc, 10.0, (frame_width * 2, frame_height))
            frame_count = 0
            T = 1
            for real_frames, pred_frames in zip(observation_list, inference_record):
                # (B, T, C, H, W) to (H, W, C) just pick up the first frame of T, and we default B=1
                real_frame = real_frames.transpose(1, 2, 0)
                pred_frame = pred_frames[0, 0].transpose(1, 2, 0)
                rotated_real = cv2.rotate(real_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                rotated_pred = cv2.rotate(pred_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                concatenated_img = np.hstack((rotated_real, rotated_pred))
                # save the concatenated image
                img = np.clip(concatenated_img, 0, 255).astype(np.uint8)
                if frame_count % 100 == 0:
                    cv2.imwrite(os.path.join(video_folder, f"frame_{frame_count}.png"), img)
                    if T > 1:
                        ARimageFolder = os.path.join(video_folder, f"ARimage_{frame_count}")
                        if not os.path.exists(ARimageFolder):
                            os.makedirs(ARimageFolder)
                        whole_ARimage = None
                        for i in range(T):
                            ARimage = pred_frames[0,i].transpose(1, 2, 0)
                            rotated_ARimage = cv2.rotate(ARimage, cv2.ROTATE_90_COUNTERCLOCKWISE)
                            ARimage = np.clip(rotated_ARimage, 0, 255).astype(np.uint8)
                            ARreal = real_frames[0,i].transpose(1, 2, 0)
                            rotated_ARreal = cv2.rotate(ARreal, cv2.ROTATE_90_COUNTERCLOCKWISE)
                            ARimage = np.clip(rotated_ARreal, 0, 255).astype(np.uint8)
                            # concatenate the ARimage and ARreal up and down
                            ARconcatenated_img = np.vstack((rotated_ARreal, rotated_ARimage))
                            if i == 0:
                                whole_ARimage = ARconcatenated_img
                            else:
                                whole_ARimage = np.hstack((whole_ARimage, ARconcatenated_img))
                            ARimage = np.clip(ARconcatenated_img, 0, 255).astype(np.uint8)
                            cv2.imwrite(os.path.join(ARimageFolder, f"ARframe_{i}.png"), ARimage)
                        cv2.imwrite(os.path.join(ARimageFolder, f"whole_ARimage.png"), whole_ARimage)
                        
                    cv2.imwrite(os.path.join(video_folder, f"frame_{frame_count}.png"), img)
                frame_count += 1
                video_writer.write(img)
            video_writer.release() 
            print(f"Saved video with {len(observation_list)} frames to {video_filename}")




def flatten_memory(caches):
    N_mem_layer = 18
    None_count = 0
    flat_memorys = []
    for cache in caches:
        flat_layers = []
        if cache is None:
            continue
        for n_mem_layer in range(N_mem_layer):
            flat_memory = np.append(cache[n_mem_layer]['recurrent_state'][0].flatten().cpu().numpy().T, cache[n_mem_layer]['recurrent_state'][1].flatten().cpu().numpy().T)
            flat_layers.append(flat_memory)
        flat_layers = np.array(flat_layers)
        flat_memorys.append(flat_layers)
    flat_memorys = np.array(flat_memorys)
    return flat_memorys


def process_into_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, list):
        return [process_into_numpy(d) for d in data]
    elif isinstance(data, dict):
        return {k: process_into_numpy(v) for k, v in data.items()}
    elif isinstance(data, tuple):
        return tuple(process_into_numpy(d) for d in data)
    elif isinstance(data, np.ndarray):
        return data
    else:
        assert False, f"Unsupported data type {type(data)}"
    return data
        

def video_visualization(pred_obs_list_with_initial, real, output_folder):
    # (B, T, C, H, W)
    import cv2
    video_folder = os.path.join(output_folder, f'video')
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

            
    assert len(pred_obs_list_with_initial) == len(real), f"Length mismatch: {len(pred_obs_list_with_initial)} vs {len(real)}"
    
    pred_obs_list_with_initial = process_into_numpy(pred_obs_list_with_initial)
    real = process_into_numpy(real)
    assert len(pred_obs_list_with_initial.shape) == 6, f"Invalid shape: {pred_obs_list_with_initial.shape}"
    assert len(real.shape) == 6, f"Invalid shape: {real.shape}" # (N, B, T, C, H, W)

    N = pred_obs_list_with_initial.shape[0]
    B = pred_obs_list_with_initial.shape[1]
    T = pred_obs_list_with_initial.shape[2]
    assert real.shape == pred_obs_list_with_initial.shape, f"Shape mismatch: {real.shape} vs {pred_obs_list_with_initial.shape}"
    assert B == 1, f"Batch size should be 1, we don't handle mutiple batch by now for generator, but got {B}"
    video_filename = os.path.join(video_folder, f"pred_obs_video{0}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    frame_height, frame_width = pred_obs_list_with_initial[0].shape[-2:]
    video_writer = cv2.VideoWriter(video_filename, fourcc, 10.0, (frame_width * 2, frame_height))
    frame_count = 0

    for real_frames, pred_frames in zip(real, pred_obs_list_with_initial):
        # (B, T, C, H, W) to (H, W, C) just pick up the first frame of T, and we default B=1

        real_frame = real_frames[0,0].transpose(1, 2, 0)
        pred_frame = pred_frames[0,0].transpose(1, 2, 0)
        rotated_real = cv2.rotate(real_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rotated_pred = cv2.rotate(pred_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        concatenated_img = np.hstack((rotated_real, rotated_pred))
        # save the concatenated image

        img = np.clip(concatenated_img, 0, 255).astype(np.uint8)
        if frame_count % 100 == 0:
            if T > 1:
                ARimageFolder = os.path.join(video_folder, f"ARimage_{frame_count}")
                if not os.path.exists(ARimageFolder):
                    os.makedirs(ARimageFolder)
                whole_ARimage = None
                for i in range(T):
                    ARimage = pred_frames[0,i].transpose(1, 2, 0)
                    rotated_ARimage = cv2.rotate(ARimage, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    ARimage = np.clip(rotated_ARimage, 0, 255).astype(np.uint8)
                    ARreal = real_frames[0,i].transpose(1, 2, 0)
                    rotated_ARreal = cv2.rotate(ARreal, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    ARimage = np.clip(rotated_ARreal, 0, 255).astype(np.uint8)
                    # concatenate the ARimage and ARreal up and down
                    ARconcatenated_img = np.vstack((rotated_ARreal, rotated_ARimage))
                    if i == 0:
                        whole_ARimage = ARconcatenated_img
                    else:
                        whole_ARimage = np.hstack((whole_ARimage, ARconcatenated_img))
                    ARimage = np.clip(ARconcatenated_img, 0, 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(ARimageFolder, f"ARframe_{i}.png"), ARimage)
                cv2.imwrite(os.path.join(ARimageFolder, f"whole_ARimage.png"), whole_ARimage)
                
            cv2.imwrite(os.path.join(video_folder, f"frame_{frame_count}.png"), img)
        frame_count += 1
        video_writer.write(img)

    
    video_writer.release() 

    print(f"Saved video with {len(real)} frames to {video_filename}")


class prediction_coding_generator(GeneratorBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key in kwargs:
            setattr(self, key, kwargs[key])
            print(f"{key}: {kwargs[key]}")
        self.output_root = self.config.output_root
        self.data_root = self.config.data_path
        self.pred_len = self.config.pred_len
        self.in_context_len = self.config.in_context_len
        self.end_position = self.config.end_position
        self.start_position = self.config.start_position
        self.record_interval = self.config.record_interval
        self.record_points = [i for i in range(self.start_position, self.end_position, self.record_interval)]
        if self.config.has_attr("max_maze"):
            self.max_maze = self.config.max_maze
        else:
            self.max_maze = None
        self.K_step = self.config.K_step
        if self.output_root is not None:
            if not os.path.exists(self.output_root):
                os.makedirs(self.output_root)
                print(f"Created output folder {self.output_root}")
        else:
            assert False, "output_root is required for general_generator"
        if self.end_position > self.config.seq_len_causal:
            assert False, "end_position should be smaller than seq_len_causal"


    def preprocess(self):
        self.dataloader = PrefetchDataLoader(
            MazeDataSet(self.config.data_path, self.config.seq_len_causal, verbose=self.main, max_maze = self.max_maze, folder_verbose=True),
            batch_size=1, #   
            rank=self.rank,
            world_size=self.world_size
            )
        print(f"Preprocessed dataloader with {len(self.dataloader)} batches")
    def __call__(self, epoch_id, rank):
        import cv2
        batch_size = 1 #  
        pred_len = self.pred_len
        for batch_id, (batch_data, folder_name) in enumerate(self.dataloader):
            folder_name = folder_name[0] # batch size is 1
            if len(folder_name.split("/")) > 1:
                parent_folder = folder_name.split("/")[0]
                sub_name = folder_name.split("/")[1]
                if not os.path.exists(os.path.join(self.output_root, parent_folder)):
                    os.makedirs(os.path.join(self.output_root, parent_folder))

            print(f"batch_id: {batch_id} processing {folder_name} with {len(batch_data)} data of shape ")
            output_folder_path = os.path.join(self.output_root, folder_name)
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)

            cmd_arr, obs_arr, behavior_actid_arr, label_actid_arr, behavior_act_arr, label_act_arr, rew_arr = batch_data
            obs_arr = obs_arr.permute(0, 1, 4, 2, 3) # (B, T, H, W, C) to (B, T, C, H, W)
            states = obs_arr.contiguous()
            commands = cmd_arr.contiguous()
            actions = behavior_actid_arr.contiguous()

            print(f"batch_id: {batch_id} processing {folder_name} with {len(batch_data)} data of shape of {states.shape}")
            assert states.shape[1] == actions.shape[1] + 1, f"states shape: {states.shape}, actions shape: {actions.shape}"
            history_cache = None
            loss_records = []
            pred_records = []
            real_records = []

            for in_context_len in [1, 10, 100, 1000]:
                pred_len = 1
                effect_len = 2
                print(f"pred_len: {pred_len}")
                print(f"in_context_len: {in_context_len}")
                mask_points = range(in_context_len + 1, min(in_context_len + self.end_position, states.shape[1] - 1), 10)
                print(f"record points: {mask_points}")
                # folder_count = 0
                output_folder_pred = os.path.join(output_folder_path, f"context_{in_context_len}")
                if not os.path.exists(output_folder_pred):
                    os.makedirs(output_folder_pred)
                
                map_loss_record = []
                
                for check_point in mask_points: # the check point will be masked by the prediction of check_point - 1
                    history_cache = None #  
                    history_before_cache = None
                    last_cache = None
                    start_point = check_point - in_context_len
                    end_point = min(check_point + effect_len, states.shape[1] - 1)
                    loss_record = {}
                    inference_record = {}
                    pred_len = 1
                    print(f"check_point: {check_point}, start_point: {start_point}, end_point: {end_point}")
                    for i in range(start_point, end_point):
                        if i == check_point - 1:
                            pred_len = self.K_step # To change the K when predicting the check point
                        end = min(i, states.shape[1] - 1)
                        pred_obs_list, history_cache = self.model.module.generate_states_only(
                                prompts=commands[:, end:end+pred_len],
                                current_observation=states[:, end:end+1], 
                                action_trajectory=actions[:, end:end+pred_len],
                                history_observation=None,
                                history_action=None, 
                                history_update_memory=False, 
                                autoregression_update_memory=False, 
                                cache=history_cache,
                                single_batch=True,
                                history_single_step=False,
                                future_single_step=False,
                                raw_images=True,
                                need_numpy=False)
                        real = states[:, end+1:end+1+pred_len]
                        print(f"check_point {i} with pred_obs_list shape: {pred_obs_list.shape}")
                        print(f"sum of real: {torch.sum(real)}")
                        mse_loss, cnt = weighted_loss(pred_obs_list.cpu(), 
                                                loss_type="mse",
                                                gt=real, 
                                                need_cnt=True,
                                                )
                        mse_loss = mse_loss/255/255
                        print(f"check_point {i} with mse_loss: {mse_loss/cnt}, cnt: {cnt}")
                            
                        
                        if i == check_point - 1: # the check point will be masked by the prediction of check_point - 1
                            print("record the history cache")
                            history_before_cache = history_cache.copy()
                            

                        if i >= check_point - 1:
                            real = states[i+1:i+1+pred_len]
                            # mse loss for every state
                            loss_record[i] = mse_loss.detach().cpu().numpy()
                            inference_record[i] = pred_obs_list[:, 0]

                    # the check point will be masked by the prediction of check_point - 1
                    masked_loss_record = {}
                    state_copy = states.clone() 
                    state_copy[:, check_point:check_point+1] = inference_record[check_point - 1]
                    history_cache = history_before_cache
                    effect_loss_sum = 0
                    masked_loss_sum = 0
                    for i in range(check_point, end_point):
                        end = i
                        pred_obs_list, history_cache = self.model.module.generate_states_only(
                                prompts=commands[:, end:end+pred_len],
                                current_observation=state_copy[:, end:end+1], 
                                action_trajectory=actions[:, end:end+pred_len],
                                history_observation=None, 
                                history_action=None, 
                                history_update_memory=False, 
                                autoregression_update_memory=False, 
                                cache=history_cache,
                                single_batch=True,
                                history_single_step=False,
                                future_single_step=False,
                                raw_images=True,
                                need_numpy=False)
                        real = state_copy[:, end+1:end+1+pred_len]
                        print(f"sum of real: {torch.sum(real)}")
                        mse_loss, cnt = weighted_loss(pred_obs_list.cpu(), 
                                                loss_type="mse",
                                                gt=real, 
                                                need_cnt=True,
                                                )
                        mse_loss = mse_loss/255/255
                        masked_loss_record[i] = mse_loss
                        masked_loss_sum += mse_loss
                        effect_loss_sum += loss_record[i]
                    print(f"masked_loss_sum: {masked_loss_sum}, effect_loss_sum: {effect_loss_sum}")
                    relative_loss_diff = (masked_loss_sum - effect_loss_sum) / effect_loss_sum
                    relative_loss_diff = relative_loss_diff.detach().cpu().numpy()
                    data_pair = (loss_record[check_point - 1], relative_loss_diff)
                    np.save(os.path.join(output_folder_pred, f"point_{check_point}.npy"), data_pair)
                    print(f"Saved point to {os.path.join(output_folder_pred, f'point_{check_point}.npy')}")
                    

    def epoch_end(self, epoch_id):
        pass


class general_generator(GeneratorBase): 

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key in kwargs:
            setattr(self, key, kwargs[key])
            print(f"{key}: {kwargs[key]}")
        self.output_root = self.config.output_root
        self.data_root = self.config.data_path
        self.pred_len = self.config.pred_len
        self.in_context_len = self.config.in_context_len
        self.end_position = self.config.end_position
        self.start_position = self.config.start_position
        self.record_interval = self.config.record_interval
        self.record_points = [i for i in range(self.start_position, self.end_position, self.record_interval)]
        
        print(f"record points: {self.record_points}")
        if self.config.has_attr("max_maze"):
            self.max_maze = self.config.max_maze
        else:
            self.max_maze = None

        if self.output_root is not None:
            if not os.path.exists(self.output_root):
                os.makedirs(self.output_root)
                print(f"Created output folder {self.output_root}")
        else:
            assert False, "output_root is required for general_generator"
        if self.end_position > self.config.seq_len_causal:
            assert False, "end_position should be smaller than seq_len_causal"
        
        self.logger_keys = ["validate_worldmodel_raw"]
        self.stat = DistStatistics(*self.logger_keys)
        if(self.config.has_attr("downsample_length")):
            self.downsample_length = self.config.downsample_length
        else:
            self.downsample_length = 10

    def preprocess(self):
        self.dataloader = PrefetchDataLoader(
            MazeDataSet(self.config.data_path, self.config.seq_len_causal, verbose=self.main, max_maze = self.max_maze, folder_verbose=True),
            batch_size=1,
            rank=self.rank,
            world_size=self.world_size
            )
        self.init_logger()
        print(f"Preprocessed dataloader with {len(self.dataloader)} batches")
    def init_logger(self):
        if not hasattr(self, 'logger'):
            self.logger = None
        if(self.logger is None):
            if(self.logger_keys is not None and len(self.logger_keys)!=0):
                assert type(self.logger_keys) == list, \
                    f"The logger_keys must be a list of string."
                process_name = f"Generation-{self.__class__.__name__}"
                max_iter = -1
                log_file = self.log_config.log_file
                self.logger = Logger(
                        *self.logger_keys,
                        on=self.main, 
                        max_iter=max_iter,
                        use_tensorboard=self.log_config.use_tensorboard,
                        log_file=log_file,
                        prefix=f"{self.run_name}-{process_name}",
                        field=f"{self.log_config.tensorboard_log}/{self.run_name}-{process_name}")

    def __call__(self, epoch_id, rank):
        import cv2
        batch_size = 1 
        pred_len = self.pred_len
        loss_batch = []
        cache_generate = True
        o_generate = False
        video_generate = False
        for batch_id, (batch_data, folder_name) in enumerate(self.dataloader):
            folder_name = folder_name[0] # batch size is 1
            if len(folder_name.split("/")) > 1:
                parent_folder = folder_name.split("/")[0]
                sub_name = folder_name.split("/")[1]
                if not os.path.exists(os.path.join(self.output_root, parent_folder)):
                    os.makedirs(os.path.join(self.output_root, parent_folder))

            print(f"batch_id: {batch_id} processing {folder_name} with {len(batch_data)} data of shape")
            output_folder_path = os.path.join(self.output_root, folder_name)
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)
            cmd_arr, obs_arr, behavior_actid_arr, label_actid_arr, behavior_act_arr, label_act_arr, rew_arr = batch_data
            obs_arr = obs_arr.permute(0, 1, 4, 2, 3) # (B, T, H, W, C) to (B, T, C, H, W)
            states = obs_arr.contiguous()
            commands = cmd_arr.contiguous()
            actions = behavior_actid_arr.contiguous()

            print(f"batch_id: {batch_id} processing {folder_name} with {len(batch_data)} data of shape of {states.shape}")
            assert states.shape[1] == actions.shape[1] + 1, f"states shape: {states.shape}, actions shape: {actions.shape}"
            history_cache = None
            self.model.module.reset()
            loss_records = []
            pred_records = []
            real_records = []
            for checkpoint_id in range(0, self.end_position):
                end = min(checkpoint_id, states.shape[1] - 1)
                pred_obs_list, history_cache = self.model.module.generate_states_only(
                        prompts=commands[:, end:end+pred_len],
                        current_observation=states[:, end:end+1], 
                        action_trajectory=actions[:, end:end+pred_len],
                        history_observation=None, #states[start:end],
                        history_action=None, #actions[start:end],
                        history_update_memory=False, 
                        autoregression_update_memory=False, # TOTEST
                        cache=history_cache,
                        single_batch=True,
                        history_single_step=False,
                        future_single_step=False,
                        raw_images=True,
                        need_numpy=False)

                real = states[:, end+1:end+1+pred_len]
                print(f"sum of real: {torch.sum(real)}")
                mse_loss, cnt = weighted_loss(pred_obs_list.cpu(), 
                                        loss_type="mse",
                                        gt=real, 
                                        need_cnt=True,
                                        )
                mse_loss = mse_loss/255/255
                print(f"check_point {checkpoint_id} with mse_loss: {mse_loss/cnt}, cnt: {cnt}")
                loss_records.append(mse_loss.detach().numpy()/cnt)  
                import copy
                if checkpoint_id in self.record_points:
                    if cache_generate == True:
                        np.save(os.path.join(output_folder_path, f"cache_{checkpoint_id}.npy"), history_cache)
                        print(f"Saved cache to {os.path.join(output_folder_path, f'cache_{checkpoint_id}.npy')}")
                    if o_generate == True:
                        o_list = self.model.module.get_o_list()
                        o_list = copy.deepcopy(o_list)
                        o_list = o_list.cpu().detach().numpy()
                        np.save(os.path.join(output_folder_path, f"o_list_{checkpoint_id}.npy"), o_list)
                        print(f"Saved o_list to {os.path.join(output_folder_path, f'o_list_{checkpoint_id}.npy')}")
                pred_records.append(pred_obs_list.cpu().detach().numpy())
                
                real = real.clone().cpu().detach().numpy()
                real_records.append(real)
            loss_records = np.array(loss_records)

            # save the loss record to npy 
            np.save(os.path.join(output_folder_path, f"losses.npy"), loss_records)
            print(f"Saved losses to {os.path.join(output_folder_path, f'losses.npy')}")
            
            loss_batch.append(loss_records)
            real_records = np.array(real_records)
            pred_records = np.array(pred_records)
            if video_generate == True:
                video_visualization(pred_records, real_records, output_folder_path)

        loss_batch = np.array(loss_batch)
        bsz = loss_batch.shape[0]
        seg_num = loss_batch.shape[1] // self.downsample_length
        valid_seq_len = seg_num * self.downsample_length
        loss_batch = np.mean(loss_batch[:, :valid_seq_len].reshape(bsz, seg_num, -1), axis=-1)
        self.stat.gather(self.device,
                validate_worldmodel_raw=loss_batch[0], 
                count=cnt)

    def epoch_end(self, epoch_id):
        stat_res = self.stat()
        if not hasattr(self, 'logger'):
            self.logger = None
        if(self.logger is not None):
            self.logger(stat_res["validate_worldmodel_raw"]["mean"],
                    epoch=epoch_id)
        if(self.extra_info is not None):
            if(self.extra_info.lower() == 'validate' and self.main):
                if not os.path.exists(self.config.output):
                    os.makedirs(self.config.output)
                for key_name in stat_res:
                    res_text = string_mean_var(self.downsample_length, stat_res[key_name])
                    file_path = f'{self.config.output}/result_{key_name}.txt'
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    with open(file_path, 'w') as f_model:
                        f_model.write(res_text)
            