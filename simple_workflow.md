python generate_robot_data.py
python train.py --experiment_name universal_robots_ur5e_experiment --dataset_path data/universal_robots_ur5e
python video_api.py --model_path output/gsplat_full --dataset_path data/universal_robots_ur5e