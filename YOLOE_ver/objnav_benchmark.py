import habitat
import os
import argparse
import csv
import cv2
import imageio
import numpy as np
from tqdm import tqdm
from constants import *
from config_utils import hm3d_config
from gpt4v_planner import GPT4V_Planner
from policy_agent import Policy_Agent
from habitat.utils.visualizations.maps import colorize_draw_agent_and_fit_to_height

# 🔁 YOLOE 전용 유틸 (세그 전용)
from cv_utils.yoloe_tools import (
    initialize_yoloe_model,
    yoloe_detection,
)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

def write_metrics(metrics, path="objnav_hm3d.csv"):
    with open(path, mode="w", newline="") as csv_file:
        fieldnames = metrics[0].keys()
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)

def adjust_topdown(metrics):
    return cv2.cvtColor(colorize_draw_agent_and_fit_to_height(metrics['top_down_map'], 1024), cv2.COLOR_BGR2RGB)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_episodes", type=int, default=200)
    return parser.parse_known_args()[0]

# ✅ bbox 없이 세그먼트만 쓰는 감지 함수
def detect_mask(image, category, yoloe_model):
    """
    YOLOE 세그만 사용해 category 마스크를 얻는다.
    - 성공 시: (True, image, mask(H,W){0,1})
    - 실패 시: (False, [], [])
    """
    det = yoloe_detection(
        image=image,
        target_classes=[category],   # 텍스트 프롬프트 1개
        model=yoloe_model,
        box_threshold=0.25,
        iou_threshold=0.50,
        run_extra_nms=False,
        use_text_prompt=True,
        retina_masks=True,
    )
    if det.masks is None or det.class_id.size == 0:
        return False, [], []

    # 프롬프트가 [category] 하나이므로 class_id==0 인 것들 중 '면적 최대' 선택
    idxs = np.where(det.class_id == 0)[0]
    if len(idxs) == 0:
        return False, [], []
    areas = [det.masks[i].sum() for i in idxs]
    top = int(idxs[int(np.argmax(areas))])
    mask = det.masks[top].astype(np.uint8)
    return True, image, mask


args = get_args()
habitat_config = hm3d_config(stage='val', episodes=args.eval_episodes)
print("scene_dataset =", habitat_config.habitat.simulator.scene_dataset)
print("scenes_dir    =", habitat_config.habitat.dataset.scenes_dir)
print("data_path     =", habitat_config.habitat.dataset.data_path)
habitat_env = habitat.Env(habitat_config)

# ✅ YOLOE 초기화 (세그 가중치 필수)
#    detect_objects는 플래너에서 쓰는 클래스 리스트와 일치시켜 두면 좋아.
DETECT_OBJECTS = ['bed', 'sofa', 'chair', 'plant', 'tv', 'toilet', 'floor']
yoloe_model = initialize_yoloe_model(
    weights=YOLOE_CHECKPOINT_PATH,   # 세그 지원 가중치
    device="cuda:0",
    classes=DETECT_OBJECTS,       # 텍스트 프롬프트 기본 세팅
    prompt_mode="text",
    conf_threshold=0.25,
    iou_threshold=0.50,
)

# ✅ 플래너/에이전트
#   - 플래너가 YOLOE 한 개만 받도록 바꿨다면 ↓ 그대로 사용
#   - 여전히 (dino_model, sam_model) 2개 인자를 받는 옛 버전이라면 같은 모델을 두 번 넣는다.
try:
    nav_planner = GPT4V_Planner(yoloe_model)
except TypeError:
    # fallback: 옛 시그니처 호환
    nav_planner = GPT4V_Planner(yoloe_model, yoloe_model)

nav_executor = Policy_Agent(model_path=POLICY_CHECKPOINT)
evaluation_metrics = []

for i in tqdm(range(args.eval_episodes)):
    find_goal = False
    obs = habitat_env.reset()
    dir = "./tmp/trajectory_%d" % i
    os.makedirs(dir, exist_ok=False)
    fps_writer = imageio.get_writer("%s/fps.mp4" % dir, fps=4)
    topdown_writer = imageio.get_writer("%s/metric.mp4" % dir, fps=4)
    heading_offset = 0
    start_geodesic_m = float(habitat_env.get_metrics()['distance_to_goal'])  

    nav_planner.reset(habitat_env.current_episode.object_category)
    episode_images = [obs['rgb']]
    episode_topdowns = [adjust_topdown(habitat_env.get_metrics())]

    # a whole round planning process
    for _ in range(11):
        obs = habitat_env.step(3)
        episode_images.append(obs['rgb'])
        episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
    goal_image, goal_mask, debug_image, goal_rotate, goal_flag = nav_planner.make_plan(episode_images[-12:])
    for j in range(min(11 - goal_rotate, 1 + goal_rotate)):
        if goal_rotate <= 6:
            obs = habitat_env.step(3)
            episode_images.append(obs['rgb'])
            episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
        else:
            obs = habitat_env.step(2)
            episode_images.append(obs['rgb'])
            episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
    nav_executor.reset(goal_image, goal_mask)

    while not habitat_env.episode_over:
        action, skill_image = nav_executor.step(obs['rgb'], habitat_env.sim.previous_step_collided)
        if action != 0 or goal_flag:
            if action == 4:
                heading_offset += 1
            elif action == 5:
                heading_offset -= 1
            obs = habitat_env.step(action)
            episode_images.append(obs['rgb'])
            episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
        else:
            if habitat_env.episode_over:
                break

            for _ in range(0, abs(heading_offset)):
                if habitat_env.episode_over:
                    break
                if heading_offset > 0:
                    obs = habitat_env.step(5)
                    episode_images.append(obs['rgb'])
                    episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
                    heading_offset -= 1
                elif heading_offset < 0:
                    obs = habitat_env.step(4)
                    episode_images.append(obs['rgb'])
                    episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
                    heading_offset += 1

            # a whole round planning process
            for _ in range(11):
                if habitat_env.episode_over:
                    break
                obs = habitat_env.step(3)
                episode_images.append(obs['rgb'])
                episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
            goal_image, goal_mask, debug_image, goal_rotate, goal_flag = nav_planner.make_plan(episode_images[-12:])
            for j in range(min(11 - goal_rotate, goal_rotate + 1)):
                if habitat_env.episode_over:
                    break
                if goal_rotate <= 6:
                    obs = habitat_env.step(3)
                    episode_images.append(obs['rgb'])
                    episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
                else:
                    obs = habitat_env.step(2)
                    episode_images.append(obs['rgb'])
                    episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
            nav_executor.reset(goal_image, goal_mask)

    for image, topdown in zip(episode_images, episode_topdowns):
        fps_writer.append_data(image)
        topdown_writer.append_data(topdown)
    fps_writer.close()
    topdown_writer.close()

    evaluation_metrics.append({
        'episode': i,
        'object_goal': habitat_env.current_episode.object_category,
        'success': habitat_env.get_metrics()['success'],
        'spl': habitat_env.get_metrics()['spl'],
        'start_distance_to_goal': start_geodesic_m, 
        'final_distance_to_goal': habitat_env.get_metrics()['distance_to_goal'],
        'llm_calls': int(nav_planner.llm_call_count),
        'llm_avg_time_sec': float(np.mean(nav_planner.llm_durations)) if len(nav_planner.llm_durations) > 0 else 0.0,
        'planner_avg_time_sec': float(np.mean(nav_planner.planner_durations)) if len(nav_planner.planner_durations) > 0 else 0.0,
    })

    write_metrics(evaluation_metrics)
