python slam_loop_closure.py \
  --images_path /home/ivm/pose_graph/pgSlam/scenario/imgs/ \
  --fps 10 \
  --start_detection_frame 50 \
  --temporal_distance 50 \
  --temporal_consistency_window 2 \
  --similarity_threshold 0.55 \
  --top_k 3 \
  --device cuda
