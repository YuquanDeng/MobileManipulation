import torch
from torchvision import transforms
import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import cv2
import numpy as np
import os
import sys
sys.path.append('../mobile-manipulation')
from policies.MLP_policy import MLP_policy
from infrastructure.utils import pretrained_ResNet50
from infrastructure.training_utils import load_model_params
import datetime

FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"

def example2():
    # Initialize simulation environment.
    env = habitat.Env(
        config=habitat.get_config("benchmark/nav/pointnav/pointnav_habitat_test.yaml")
    )
    print("Environment creation successful")
    observations = env.reset()
    log_observation(observations)
    print("Agent stepping around inside environment.")

    # Load pretrained ResNet50 and control Policy.
    device = torch.device("cpu")
    resnet50 = pretrained_ResNet50(device)

    curr_dir = os.getcwd()
    MLP_policy = load_model_params(os.path.join(curr_dir, "../mobile-manipulation/conf_outputs/2023-04-07/16-35/MLP_policy.pt"))
    MLP_policy.to(device)


    # Get goal image and origin image.
    goal_img = np.load("./goal_images/goal_img_2023-04-04_08-09-24.npy")
    print("Loaded goal image successful.")

    curr_img = transform_rgb_bgr(observations["rgb"])
    cv2.imshow("RGB", curr_img)

    print("curr_img: ", curr_img)
    action = get_policy(MLP_policy, resnet50, curr_img, goal_img, device)

    # 1 episode.
    count_steps = 0
    while not env.episode_over:
        keystroke = cv2.waitKey(0)

        if keystroke == ord(FORWARD_KEY):
            action = HabitatSimActions.move_forward
            print("action: FORWARD")
        elif keystroke == ord(LEFT_KEY):
            action = HabitatSimActions.turn_left
            print("action: LEFT")
        elif keystroke == ord(RIGHT_KEY):
            action = HabitatSimActions.turn_right
            print("action: RIGHT")
        elif keystroke == ord(FINISH):
            action = HabitatSimActions.stop
            print("action: FINISH")
        else:
            print("INVALID KEY")
            continue

        observations = env.step(action)
        curr_img = transform_rgb_bgr(observations["rgb"])
        count_steps += 1
        log_observation(observations)
        cv2.imshow("RGB", curr_img)

        print(curr_img)
        action = get_policy(MLP_policy, resnet50, curr_img, goal_img, device)

    print("Episode finished after {} steps.".format(count_steps))
    reach_goal(action, observations)

def reach_goal(action, observations):
    if (
        action == HabitatSimActions.stop
        and observations["pointgoal_with_gps_compass"][0] < 0.2
    ):
        print("you successfully navigated to destination point")
    else:
        print("your navigation was unsuccessful")

def example():
    env = habitat.Env(
        config=habitat.get_config("benchmark/nav/pointnav/pointnav_habitat_test.yaml")
    )

    print("Environment creation successful")
    observations = env.reset()
    print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
        observations["pointgoal_with_gps_compass"][0],
        observations["pointgoal_with_gps_compass"][1]))
    cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))

    print("Agent stepping around inside environment.")

    count_steps = 0
    while not env.episode_over:
        keystroke = cv2.waitKey(0)

        if keystroke == ord(FORWARD_KEY):
            action = HabitatSimActions.move_forward
            print("action: FORWARD")
        elif keystroke == ord(LEFT_KEY):
            action = HabitatSimActions.turn_left
            print("action: LEFT")
        elif keystroke == ord(RIGHT_KEY):
            action = HabitatSimActions.turn_right
            print("action: RIGHT")
        elif keystroke == ord(FINISH):
            action = HabitatSimActions.stop
            print("action: FINISH")
        else:
            print("INVALID KEY")
            continue

        observations = env.step(action)
        count_steps += 1

        print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
            observations["pointgoal_with_gps_compass"][0],
            observations["pointgoal_with_gps_compass"][1]))
        cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))

    print("Episode finished after {} steps.".format(count_steps))

    if (
        action == HabitatSimActions.stop
        and observations["pointgoal_with_gps_compass"][0] < 0.2
    ):
        print("you successfully navigated to destination point")
    else:
        print("your navigation was unsuccessful")
    
    # Part 1: Save the current images(maybe goal images or intermediate 
    # goal images) with current times.
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    np.save(f'./goal_images/goal_img_{timestamp}', transform_rgb_bgr(observations["rgb"]))
    print("Successfully saved goal image.")

#########
# utils
#########
def get_policy(MLP_policy, resnet50, curr_img, goal_img, device):
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    resnet50.eval()
    MLP_policy.eval()
    with torch.no_grad():
        # Extract features.
        # Change into HxWxC
        C, H, W = curr_img.shape
        curr_img = curr_img.reshape((H, W, C))
        goal_img = goal_img.reshape((H, W, C))

        curr_img = val_transform(curr_img).unsqueeze(0).to(device)
        goal_img = val_transform(goal_img).unsqueeze(0).to(device)
        curr_features = resnet50(curr_img)
        goal_features = resnet50(goal_img)
        print(f'curr_features size: {curr_features.size()}')
        combined_features = torch.cat((curr_features.reshape((1, 2048)), goal_features.reshape((1, 2048))), -1)

        # Output actions.
        combined_features.to(device)
        action = MLP_policy(combined_features)

    print(f"action: {action}")
    return action.detach().cpu().numpy()

def load_model(path: str):
    print('-'*100)
    print(f"Successfully loaded model from {path}")
    return torch.load(path)

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def log_observation(observations):
    print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
        observations["pointgoal_with_gps_compass"][0],
        observations["pointgoal_with_gps_compass"][1]))

def a1_action2discrete_action(a1_action: np.array, vel_threshold: float=0.01, theta_threshold: float=0.01):
    # Randomly takes 5 actions
    count_steps = 0
    init_actions = np.random.randint(0, 3, size=5)
    number2action = {
        0: HabitatSimActions.move_forward,
        1: HabitatSimActions.turn_left,
        2: HabitatSimActions.turn_right
    }

    number2string = {
        0: "action: FORWARD",
        1: "action: LEFT",
        2: "action: RIGHT"
    }
    
    for action in init_actions:
        print(number2string[action])
        curr_img = env_logger(env, number2action[action])
        count_steps += 1

    return None


if __name__ == "__main__":
    example2()
