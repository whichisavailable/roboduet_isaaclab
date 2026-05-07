# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import pathlib
import statistics
import time
from collections import deque

import git
import torch

import rsl_rl


class Logger:
    """Compatibility logger with the same console/TensorBoard style as RSL-RL."""

    def __init__(
        self,
        log_dir: str | None,
        cfg: dict,
        env_cfg: dict | object,
        num_envs: int,
        is_distributed: bool,
        gpu_world_size: int,
        gpu_global_rank: int,
        device: str,
    ) -> None:
        self.log_dir = log_dir
        self.cfg = cfg
        self.env_cfg = env_cfg
        self.num_envs = num_envs
        self.gpu_world_size = gpu_world_size
        self.device = device
        self.git_status_repos = [rsl_rl.__file__]
        self.tot_timesteps = 0
        self.tot_time = 0

        self.ep_extras = []
        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)
        self.cur_reward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.cur_episode_length = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        if self.cfg["algorithm"].get("rnd_cfg"):
            self.erewbuffer = deque(maxlen=100)
            self.irewbuffer = deque(maxlen=100)
            self.cur_ereward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.cur_ireward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.disable_logs = is_distributed and gpu_global_rank != 0

    def init_logging_writer(self) -> None:
        if self.log_dir is not None and not self.disable_logs:
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()
            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'wandb', 'neptune', or 'tensorboard'.")
        else:
            self.writer = None

        files_to_upload = self._store_code_state()
        if self.writer is not None and getattr(self, "logger_type", "tensorboard") in ["wandb", "neptune"]:
            self.writer.store_config(self.env_cfg, self.cfg)  # type: ignore[attr-defined]
            for path in files_to_upload:
                self.writer.save_file(path)  # type: ignore[attr-defined]

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict,
        intrinsic_rewards: torch.Tensor | None = None,
    ) -> None:
        if self.writer is None:
            return
        if "episode" in extras:
            self.ep_extras.append(extras["episode"])
        elif "log" in extras:
            self.ep_extras.append(extras["log"])

        if intrinsic_rewards is not None:
            self.cur_ereward_sum += rewards
            self.cur_ireward_sum += intrinsic_rewards
            self.cur_reward_sum += rewards + intrinsic_rewards
        else:
            self.cur_reward_sum += rewards
        self.cur_episode_length += 1

        new_ids = (dones > 0).nonzero(as_tuple=False)
        self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
        self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
        self.cur_reward_sum[new_ids] = 0
        self.cur_episode_length[new_ids] = 0
        if intrinsic_rewards is not None:
            self.erewbuffer.extend(self.cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
            self.irewbuffer.extend(self.cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
            self.cur_ereward_sum[new_ids] = 0
            self.cur_ireward_sum[new_ids] = 0

    def log(
        self,
        it: int,
        start_it: int,
        total_it: int,
        collect_time: float,
        learn_time: float,
        loss_dict: dict,
        learning_rate: float,
        action_std: torch.Tensor,
        rnd_weight: float | None,
        print_minimal: bool = False,
        width: int = 80,
        pad: int = 40,
    ) -> None:
        if self.writer is None:
            return

        collection_size = self.cfg["num_steps_per_env"] * self.num_envs * self.gpu_world_size
        iteration_time = collect_time + learn_time
        self.tot_timesteps += collection_size
        self.tot_time += iteration_time

        extras_string = ""
        if self.ep_extras:
            for key in self.ep_extras[0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in self.ep_extras:
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                if "/" in key:
                    self.writer.add_scalar(key, value, it)
                    extras_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, it)
                    extras_string += f"""{f"Mean episode {key}:":>{pad}} {value:.4f}\n"""

        for key, value in loss_dict.items():
            self.writer.add_scalar(f"Loss/{key}", value, it)
        self.writer.add_scalar("Loss/learning_rate", learning_rate, it)
        self.writer.add_scalar("Policy/mean_std", action_std.mean().item(), it)

        fps = int(collection_size / (collect_time + learn_time))
        self.writer.add_scalar("Perf/total_fps", fps, it)
        self.writer.add_scalar("Perf/collection_time", collect_time, it)
        self.writer.add_scalar("Perf/learning_time", learn_time, it)

        if len(self.rewbuffer) > 0:
            if self.cfg["algorithm"].get("rnd_cfg"):
                self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(self.erewbuffer), it)
                self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(self.irewbuffer), it)
                self.writer.add_scalar("Rnd/weight", rnd_weight, it)
            self.writer.add_scalar("Train/mean_reward", statistics.mean(self.rewbuffer), it)
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(self.lenbuffer), it)
            if getattr(self, "logger_type", "tensorboard") != "wandb":
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(self.rewbuffer), int(self.tot_time))
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(self.lenbuffer), int(self.tot_time)
                )

        log_string = f"""{"#" * width}\n"""
        log_string += f"""\033[1m{f" Learning iteration {it}/{total_it} ".center(width)}\033[0m \n\n"""
        run_name = self.cfg.get("run_name")
        log_string += f"""{"Run name:":>{pad}} {run_name}\n""" if run_name else ""
        log_string += (
            f"""{"Total steps:":>{pad}} {self.tot_timesteps} \n"""
            f"""{"Steps per second:":>{pad}} {fps:.0f} \n"""
            f"""{"Collection time:":>{pad}} {collect_time:.3f}s \n"""
            f"""{"Learning time:":>{pad}} {learn_time:.3f}s \n"""
        )
        for key, value in loss_dict.items():
            log_string += f"""{f"Mean {key} loss:":>{pad}} {value:.4f}\n"""
        if len(self.rewbuffer) > 0:
            if self.cfg["algorithm"].get("rnd_cfg"):
                log_string += f"""{"Mean extrinsic reward:":>{pad}} {statistics.mean(self.erewbuffer):.2f}\n"""
                log_string += f"""{"Mean intrinsic reward:":>{pad}} {statistics.mean(self.irewbuffer):.2f}\n"""
            log_string += f"""{"Mean reward:":>{pad}} {statistics.mean(self.rewbuffer):.2f}\n"""
            log_string += f"""{"Mean episode length:":>{pad}} {statistics.mean(self.lenbuffer):.2f}\n"""
        log_string += f"""{"Mean action std:":>{pad}} {action_std.mean().item():.2f}\n"""
        if not print_minimal:
            log_string += extras_string
        done_it = it + 1 - start_it
        remaining_it = total_it - start_it - done_it
        eta = self.tot_time / done_it * remaining_it
        log_string += (
            f"""{"-" * width}\n"""
            f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
            f"""{"Time elapsed:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{"ETA:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(eta))}\n"""
        )
        print(log_string)

        if getattr(self, "logger_type", "tensorboard") == "wandb":
            for video in pathlib.Path(self.log_dir).rglob("*.mp4"):  # type: ignore[arg-type]
                self.writer.save_video(video, it)  # type: ignore[attr-defined]
        self.ep_extras.clear()

    def save_model(self, path: str, it: int) -> None:
        if self.writer is not None and getattr(self, "logger_type", "tensorboard") in ["neptune", "wandb"]:
            self.writer.save_model(path, it)  # type: ignore[attr-defined]

    def stop_logging_writer(self) -> None:
        if self.writer is not None and getattr(self, "logger_type", "tensorboard") in ["neptune", "wandb"]:
            self.writer.stop()  # type: ignore[attr-defined]

    def _store_code_state(self) -> list[str]:
        files_to_upload = []
        if self.log_dir is not None and not self.disable_logs:
            git_log_dir = os.path.join(self.log_dir, "git")
            os.makedirs(git_log_dir, exist_ok=True)
            for repository_file_path in self.git_status_repos:
                try:
                    repo = git.Repo(repository_file_path, search_parent_directories=True)
                    tree = repo.head.commit.tree
                    commit_hash = repo.head.commit.hexsha
                except Exception:
                    print(f"Could not find git repository in {repository_file_path}. Skipping.")
                    continue
                repo_name = pathlib.Path(repo.working_dir).name
                diff_file_name = os.path.join(git_log_dir, f"{repo_name}.diff")
                if os.path.isfile(diff_file_name):
                    continue
                print(f"Storing git diff for '{repo_name}' in: {diff_file_name}")
                with open(diff_file_name, "x", encoding="utf-8") as handle:
                    content = (
                        f"--- git commit ---\n{commit_hash}\n\n\n"
                        f"--- git status ---\n{repo.git.status()} \n\n\n"
                        f"--- git diff ---\n{repo.git.diff(tree)}"
                    )
                    handle.write(content)
                files_to_upload.append(diff_file_name)
        return files_to_upload
