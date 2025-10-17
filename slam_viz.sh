python slam_loop_closure_realtime.py \
    --images_path /home/ivm/pose_graph/pgSlam/scenario/imgs/ \
    --poses_file /home/ivm/pose_graph/pgSlam/scenario/vertices_stan.txt \
    --fps 30 \
    --start_detection_frame 50 \
    --temporal_distance 50 \
    --similarity_threshold 0.55 \
    --temporal_consistency_window 2 \
    --max_frames 1100 \
    --update_every 5

