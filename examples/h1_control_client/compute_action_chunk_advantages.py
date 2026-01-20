"""
Compute action chunk advantages using visual similarity and future rewards.

This script implements fine-grained advantage labeling at the action chunk level,
rather than labeling entire episodes. It:

1. Pre-computes rewards for all frames using Qwen VLM
2. Pre-computes visual embeddings using DINOv3
3. For each action chunk (starting frame), determines if it has "advantage" by:
   - Finding similar states across all episodes
   - Looking at future rewards from those similar states
   - Labeling as advantage=True if in top 1/3 of outcomes

Based on notebooks:
- pre_compute_reward_observation_embd.ipynb
- Qwen_action_chunck_advantage_inference_test.ipynb

Usage:
    python compute_action_chunk_advantages.py \
        --data-dir path/to/parquet/files \
        --task-instruction "Fold the towel into a small square" \
        --checkpoint-path path/to/qwen/checkpoint \
        --output-dir path/to/output

Author: Adapted from embodied reward notebooks
"""

import os
import sys
import glob
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import io
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "third_party" / "emboided_reward"))

try:
    from embodied_reward_util import sample_even_frames_from_images
    from Qwen_reward_inference_util import (
        load_model_and_tokenizer,
        build_qwen3vl_sft_sample,
        run_inference,
        parse_first_float
    )
except ImportError as e:
    print(f"Error importing reward utilities: {e}")
    print("Make sure third_party/emboided_reward/ modules are available")
    sys.exit(1)


def parse_epoch_episode(p: str) -> Tuple[int, int]:
    """Extract epoch and episode numbers from file path."""
    path = Path(p)
    
    import re
    m_ep = re.search(r"episode_(\d+)\.parquet$", path.name)
    if not m_ep:
        raise ValueError(f"Cannot find episode_XXXXXX.parquet in: {p}")
    episode = int(m_ep.group(1))
    
    epoch = None
    for part in path.parts:
        if part.startswith("epoch_"):
            epoch = int(part.split("_", 1)[1])
            break
    if epoch is None:
        # No epoch in path, default to 0
        epoch = 0
    
    return epoch, episode


def topk_similar(q: torch.Tensor, db: torch.Tensor, k: int = 10) -> Tuple[List[int], List[float]]:
    """Find top-k most similar embeddings using cosine similarity."""
    sims = db @ q
    vals, idxs = torch.topk(sims, k=min(k, sims.numel()))
    return idxs.tolist(), vals.tolist()


class ActionChunkAdvantageComputer:
    """Computes action chunk advantages using visual similarity and future rewards."""
    
    def __init__(
        self,
        task_instruction: str,
        reward_method: str = "Ours",
        qwen_checkpoint_path: str = "",
        dinov3_model_id: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        max_frames: int = 30,
        look_ahead_window: int = 80,
        advantage_threshold: float = 1/3,
        device: str = "cuda",
    ):
        """
        Args:
            task_instruction: Task description for reward model
            reward_method: Reward model method ('Ours' or 'GVL')
            qwen_checkpoint_path: Path to fine-tuned Qwen checkpoint (for 'Ours' method)
            dinov3_model_id: HuggingFace model ID for DINOv3
            max_frames: Max frames to sample for reward prediction
            look_ahead_window: Number of future frames to consider for advantage
            advantage_threshold: Top percentile threshold (e.g., 1/3 for top 33%)
            device: cuda or cpu
        """
        self.task_instruction = task_instruction
        self.reward_method = reward_method
        self.max_frames = max_frames
        self.look_ahead_window = look_ahead_window
        self.advantage_threshold = advantage_threshold
        self.device = device
        
        print("Loading models...")
        print(f"  Device: {device}")
        print(f"  Reward Method: {reward_method}")
        
        # Load reward model based on method
        if reward_method == "Ours":
            # Load Qwen reward model
            print(f"  Loading Qwen from: {qwen_checkpoint_path}")
            self.qwen_model, self.qwen_tokenizer = load_model_and_tokenizer(qwen_checkpoint_path)
            self.openai_client = None
        elif reward_method == "GVL":
            # Load OpenAI client
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set in environment")
            print("  Using OpenAI GPT-5.2 for GVL method")
            self.openai_client = OpenAI(api_key=api_key)
            self.qwen_model = None
            self.qwen_tokenizer = None
        else:
            raise ValueError(f"Unknown reward method: {reward_method}")
        
        # Load DINOv3 for visual embeddings
        print(f"  Loading DINOv3: {dinov3_model_id}")
        from transformers import AutoImageProcessor, AutoModel
        
        hf_token = os.environ.get("HF_TOKEN")
        self.dinov3_processor = AutoImageProcessor.from_pretrained(dinov3_model_id, token=hf_token)
        self.dinov3_model = AutoModel.from_pretrained(dinov3_model_id, token=hf_token).to(device).eval()
        
        print("Models loaded successfully!")
    
    @torch.inference_mode()
    def embed_images(self, pils: List[Image.Image], batch_size: int = 64) -> torch.Tensor:
        """Compute normalized DINOv3 embeddings for images."""
        outs = []
        for i in range(0, len(pils), batch_size):
            batch = pils[i:i+batch_size]
            inputs = self.dinov3_processor(images=batch, return_tensors="pt").to(self.device)
            out = self.dinov3_model(**inputs)
            x = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out.last_hidden_state[:, 0, :]
            outs.append(F.normalize(x, dim=-1).cpu())
        return torch.cat(outs, 0)
    
    def compute_episode_rewards(self, ego_images: List[Image.Image]) -> np.ndarray:
        """Compute dense reward predictions for all frames in an episode."""
        # Sample frames for reward prediction
        test_video_frames = sample_even_frames_from_images(ego_images, max_frames=self.max_frames, rotate_angle=0)
        
        # Add fake GT reward (required by build_qwen3vl_sft_sample)
        for i, fr in enumerate(test_video_frames):
            fr.gt_reward = 0
        
        if self.reward_method == "Ours":
            # Use Qwen model
            # Build inference samples (one per frame)
            test_video_data = [
                build_qwen3vl_sft_sample(
                    frames=test_video_frames,
                    task=self.task_instruction,
                    anchor_gt_idx=1,
                    shuffle_frames=False,
                    target_frame_idx=i + 1,
                )
                for i in range(len(test_video_frames))
            ]
            
            # Run inference with batch_size=1 to avoid batching issues
            reward_pred = run_inference(self.qwen_model, self.qwen_tokenizer, test_video_data, inference_batch_size=1)
            
        elif self.reward_method == "GVL":
            # Use OpenAI GVL model
            from embodied_reward_util import (
                shuffle_frames,
                build_gemini_parts,
                build_openai_responses_input_from_gemini_parts
            )
            
            # Shuffle frames
            shuffle_frames(test_video_frames)
            
            # Build input for OpenAI
            gemini_input = build_gemini_parts(test_video_frames, self.task_instruction)
            openai_input = build_openai_responses_input_from_gemini_parts(gemini_input)
            
            # Get reward predictions
            resp = self.openai_client.responses.create(
                model="gpt-5.2",
                input=openai_input,
            )
            reward_pred_result_raw = resp.output_text
            reward_pred = self._parse_reward_from_result(test_video_frames, reward_pred_result_raw)
        
        else:
            raise ValueError(f"Unknown reward method: {self.reward_method}")
        
        # Interpolate to get dense rewards for all frames
        anchor_idx = np.array([fr.src_idx for fr in test_video_frames], dtype=int)
        anchor_r = np.array(reward_pred, dtype=float)
        dense_r = np.interp(np.arange(len(ego_images)), anchor_idx, anchor_r)
        
        return dense_r
    
    def _parse_reward_from_result(self, rollout_frames: List, raw_result: str) -> List[int]:
        """Parse reward predictions from GVL response."""
        import re
        import json
        
        arr_text = raw_result
        try:
            data = json.loads(arr_text)
        except json.JSONDecodeError:
            # Try to repair JSON
            repaired = re.sub(r",\s*([}\]])", r"\1", arr_text)
            try:
                data = json.loads(repaired)
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse GVL reward result, using zeros")
                return [0] * len(rollout_frames)
        
        # Sort by shuffled order and extract percentages
        by_shuf = sorted(rollout_frames, key=lambda f: f.shuf_idx)
        reward_pred = []
        
        for i, item in enumerate(data):
            if i >= len(by_shuf):
                break
            percent = int(max(0, min(100, int(item.get("task_completion_percentage", 0)))))
            reward_pred.append(percent)
        
        # Ensure we have the right number of predictions
        while len(reward_pred) < len(rollout_frames):
            reward_pred.append(0)
        
        return reward_pred
    
    def query_action_chunk_advantage(
        self,
        test_episode_idx: int,
        test_start_frame: int,
        all_episode_embeddings: List[torch.Tensor],
        all_episode_rewards: np.ndarray,
    ) -> bool:
        """
        Determine if an action chunk has advantage.
        
        Args:
            test_episode_idx: Index of the test episode
            test_start_frame: Starting frame of the action chunk
            all_episode_embeddings: List of embedding tensors, one per episode
            all_episode_rewards: Padded rewards array (N_episodes x T_max)
        
        Returns:
            True if action chunk has advantage (in top 1/3), False otherwise
        """
        N_episodes = len(all_episode_embeddings)
        T_max = all_episode_rewards.shape[1]
        
        test_embeddings = all_episode_embeddings[test_episode_idx]
        T_test = test_embeddings.shape[0]
        
        # Check bounds
        if test_start_frame >= T_test:
            return False
        
        test_embedding = test_embeddings[test_start_frame]
        
        # Find most similar frame in each episode
        episode_to_similar_idx = {}
        for ep_idx in range(N_episodes):
            ep_embeddings = all_episode_embeddings[ep_idx]
            similar_idx, _ = topk_similar(test_embedding, ep_embeddings, k=1)
            episode_to_similar_idx[ep_idx] = similar_idx[0]
        
        # Compute mean future rewards for each episode
        future_rewards_per_episode = []
        for ep_idx in range(N_episodes):
            similar_idx = episode_to_similar_idx[ep_idx]
            end_idx = min(similar_idx + self.look_ahead_window, T_max)
            future_rewards = all_episode_rewards[ep_idx, similar_idx:end_idx]
            
            # Skip if all NaN
            if np.isnan(future_rewards).all():
                continue
            
            mean_reward = np.nanmean(future_rewards)
            future_rewards_per_episode.append([ep_idx, mean_reward])
        
        # Rank episodes by future reward
        future_rewards_per_episode.sort(key=lambda x: x[1], reverse=True)
        
        # Check if test episode is in top threshold%
        threshold_count = int(self.advantage_threshold * len(future_rewards_per_episode))
        threshold_count = max(1, threshold_count)  # At least 1
        
        top_episode_indices = [x[0] for x in future_rewards_per_episode[:threshold_count]]
        
        return test_episode_idx in top_episode_indices


def load_episode_images(parquet_path: str) -> List[Image.Image]:
    """Load ego camera images from a parquet episode file."""
    df = pd.read_parquet(parquet_path)
    df_sorted = df.sort_values('frame_index', kind='stable')
    ego_images = [
        Image.open(io.BytesIO(x['bytes'])).convert('RGB')
        for x in df_sorted['ego_cam'].tolist()
    ]
    return ego_images


def main():
    parser = argparse.ArgumentParser(description="Compute action chunk advantages")
    parser.add_argument("--data-dir", required=True, help="Directory containing episode parquet files")
    parser.add_argument("--task-instruction", required=True, help="Task instruction for reward model")
    parser.add_argument("--checkpoint-path", default="", help="Path to Qwen checkpoint (required for 'Ours' method)")
    parser.add_argument("--reward-method", default="Ours", choices=["Ours", "GVL"], help="Reward model method: 'Ours' or 'GVL'")
    parser.add_argument("--output-dir", default=None, help="Output directory (defaults to data-dir)")
    parser.add_argument("--max-frames", type=int, default=30, help="Max frames for reward sampling")
    parser.add_argument("--look-ahead-window", type=int, default=80, help="Frames to look ahead for advantage")
    parser.add_argument("--advantage-threshold", type=float, default=1/3, help="Top percentile for advantage (e.g., 0.33)")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--force-recompute", action="store_true", help="Recompute even if cache files exist")
    
    args = parser.parse_args()
    
    # Validate arguments based on reward method
    if args.reward_method == "Ours":
        if not args.checkpoint_path:
            print("Error: --checkpoint-path is required for reward method 'Ours'")
            sys.exit(1)
        if not os.path.exists(args.checkpoint_path):
            print(f"Error: Checkpoint path does not exist: {args.checkpoint_path}")
            sys.exit(1)
    elif args.reward_method == "GVL":
        if not os.environ.get("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY environment variable is required for reward method 'GVL'")
            sys.exit(1)
    
    # Setup
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all episode files
    episode_files = sorted(glob.glob(str(data_dir / "episode_*.parquet")))
    if not episode_files:
        print(f"Error: No episode files found in {data_dir}")
        sys.exit(1)
    
    print(f"\nFound {len(episode_files)} episodes in {data_dir}")
    
    # Initialize computer
    computer = ActionChunkAdvantageComputer(
        task_instruction=args.task_instruction,
        reward_method=args.reward_method,
        qwen_checkpoint_path=args.checkpoint_path,
        max_frames=args.max_frames,
        look_ahead_window=args.look_ahead_window,
        advantage_threshold=args.advantage_threshold,
        device=args.device,
    )
    
    # Step 1: Compute rewards for all episodes
    print("\n" + "="*80)
    print("STEP 1: Computing rewards for all episodes")
    print("="*80)
    
    all_episode_rewards = []
    for ep_file in tqdm(episode_files, desc="Computing rewards"):
        # Include reward method in cache filename
        reward_cache = ep_file.replace('.parquet', f'_{args.reward_method}_reward.pkl')
        
        if os.path.exists(reward_cache) and not args.force_recompute:
            rewards = pickle.load(open(reward_cache, 'rb'))
        else:
            images = load_episode_images(ep_file)
            rewards = computer.compute_episode_rewards(images)
            pickle.dump(rewards, open(reward_cache, 'wb'))
        
        all_episode_rewards.append(rewards)
    
    # Pad rewards to same length
    T_max = max(len(r) for r in all_episode_rewards)
    N_episodes = len(all_episode_rewards)
    all_episode_rewards_padded = np.full((N_episodes, T_max), np.nan, dtype=float)
    for i, r in enumerate(all_episode_rewards):
        all_episode_rewards_padded[i, :len(r)] = r
    
    print(f"Rewards computed: {N_episodes} episodes, max length {T_max}")
    
    # Step 2: Compute visual embeddings for all episodes
    print("\n" + "="*80)
    print("STEP 2: Computing visual embeddings")
    print("="*80)
    
    all_episode_embeddings = []
    for ep_file in tqdm(episode_files, desc="Computing embeddings"):
        embd_cache = ep_file.replace('.parquet', '_ego_image_embeddings.pkl')
        
        if os.path.exists(embd_cache) and not args.force_recompute:
            embeddings = pickle.load(open(embd_cache, 'rb'))
        else:
            images = load_episode_images(ep_file)
            embeddings = computer.embed_images(images, batch_size=64)
            pickle.dump(embeddings, open(embd_cache, 'wb'))
        
        all_episode_embeddings.append(embeddings)
    
    print(f"Embeddings computed: {N_episodes} episodes")
    
    # Step 3: Compute action chunk advantages
    print("\n" + "="*80)
    print("STEP 3: Computing action chunk advantages")
    print("="*80)
    
    all_stats = []
    for ep_idx, ep_file in enumerate(tqdm(episode_files, desc="Computing advantages")):
        advantage_cache = ep_file.replace('.parquet', '_action_chunk_advantages.pkl')
        
        if os.path.exists(advantage_cache) and not args.force_recompute:
            # Load existing and show stats
            advantages = pickle.load(open(advantage_cache, 'rb'))
            true_count = advantages.sum()
            print(f"  [CACHED] {Path(ep_file).name}: {true_count}/{len(advantages)} frames with advantage ({100*true_count/len(advantages):.1f}%)")
            all_stats.append((Path(ep_file).name, true_count, len(advantages)))
            continue
        
        T_episode = all_episode_embeddings[ep_idx].shape[0]
        
        # Compute advantage for each potential action chunk start
        advantages = np.zeros(T_episode, dtype=bool)
        for start_frame in range(T_episode):
            advantages[start_frame] = computer.query_action_chunk_advantage(
                test_episode_idx=ep_idx,
                test_start_frame=start_frame,
                all_episode_embeddings=all_episode_embeddings,
                all_episode_rewards=all_episode_rewards_padded,
            )
        
        # Save advantages
        pickle.dump(advantages, open(advantage_cache, 'wb'))
        
        # Stats
        true_count = advantages.sum()
        print(f"  {Path(ep_file).name}: {true_count}/{T_episode} frames with advantage ({100*true_count/T_episode:.1f}%)")
        all_stats.append((Path(ep_file).name, true_count, T_episode))
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    total_true = sum(s[1] for s in all_stats)
    total_frames = sum(s[2] for s in all_stats)
    print(f"Total episodes: {len(all_stats)}")
    print(f"Total frames: {total_frames}")
    print(f"Frames with Advantage=True: {total_true} ({100*total_true/total_frames:.1f}%)")
    print(f"Frames with Advantage=False: {total_frames - total_true} ({100*(total_frames-total_true)/total_frames:.1f}%)")
    print(f"\nPer-episode breakdown:")
    for name, true_cnt, total in sorted(all_stats):
        print(f"  {name}: {true_cnt:4d}/{total:4d} ({100*true_cnt/total:5.1f}%)")
    
    print("\n" + "="*80)
    print("DONE! Action chunk advantages computed.")
    print("="*80)
    print(f"Output files: {output_dir}/*_action_chunk_advantages.pkl")
    print("\nNext steps:")
    print("  1. Run convert script to create LeRobot dataset with per-chunk advantages")
    print("  2. Train with advantage-augmented prompts")


if __name__ == "__main__":
    main()

