import numpy as np
from PIL import Image
from envs.base import GymEnv
from envs.alfred.thor_env import ThorEnv
import utils.thor_constants as constants

from utils.logger_utils import get_logger

logger = get_logger("alfred env")


class ALFREDEnv(GymEnv):
    def __init__(self, config, **kwargs):
        self.hparams = config

        self._env = ThorEnv(x_display=self.hparams.x_display)

        self.num_fails = 0
        self.non_interact_actions = [
            "MoveAhead",
            "Rotate",
            "Look",
            "<<stop>>",
            "<<pad>>",
            "<<seg>>",
        ]

    @property
    def last_event(self):
        return self._env.last_event

    def has_interaction(self, action):
        """
        check if low-level action is interactive
        """
        if any(a in action for a in self.non_interact_actions):
            return False
        else:
            return True

    def reset(self):
        self.num_fails = 0
        return self._env.last_event.frame

    def setup_scene(self, traj_data, reward_type="dense", test_split=False):
        """
        intialize the scene and agent from the task info
        """
        # scene setup
        scene_num = traj_data["scene"]["scene_num"]
        object_poses = traj_data["scene"]["object_poses"]
        dirty_and_empty = traj_data["scene"]["dirty_and_empty"]
        object_toggles = traj_data["scene"]["object_toggles"]
        scene_name = "FloorPlan%d" % scene_num

        logger.info(f"restoring scene: {scene_name} with {reward_type} rewards")

        self._env.reset(scene_name, silent=True)
        self._env.restore_scene(object_poses, object_toggles, dirty_and_empty)

        # initialize to start position
        self._env.step(dict(traj_data["scene"]["init_action"]))

        # setup task for reward
        if not test_split:
            self._env.set_task(traj_data, reward_type=reward_type)

    def read_task_data(self, task, subgoal_idx=None):
        """
        read data from the traj_json
        """

        # read general task info
        repeat_idx = task["repeat_idx"]
        task_dict = {
            "repeat_idx": repeat_idx,
            "type": task["task_type"],
            "task": "/".join(task["root"].split("/")[-3:-1]),
        }
        # read subgoal info
        if subgoal_idx is not None:
            task_dict["subgoal_idx"] = subgoal_idx
            task_dict["subgoal_action"] = task["plan"]["high_pddl"][subgoal_idx][
                "discrete_action"
            ]["action"]
        return task_dict

    def get_observation(self, extractor):
        """
        get environment observation
        """

        frames = extractor.featurize(
            [Image.fromarray(self._env.last_event.frame)], batch=1
        )
        return frames

    def extract_rcnn_pred(self, class_idx, obj_predictor):
        """
        extract a pixel mask using a pre-trained MaskRCNN
        """

        rcnn_pred = obj_predictor.predict_objects(
            Image.fromarray(self._env.last_event.frame)
        )
        class_name = obj_predictor.vocab_obj.index2word(class_idx)
        candidates = list(filter(lambda p: p.label == class_name, rcnn_pred))

        if self.hparams.debug:
            visible_objs = [
                obj
                for obj in self._env.last_event.metadata["objects"]
                if obj["visible"] and obj["objectId"].startswith(class_name + "|")
            ]
            print(
                "Agent prediction = {}, detected {} objects (visible {})".format(
                    class_name, len(candidates), len(visible_objs)
                )
            )

        if len(candidates) > 0:
            if self._env.last_interaction[0] == class_idx:
                # last_obj['id'] and class_name + '|' in self._env.last_obj['id']:
                # do the association based selection
                last_center = np.array(self._env.last_interaction[1].nonzero()).mean(
                    axis=1
                )
                cur_centers = np.array(
                    [np.array(c.mask[0].nonzero()).mean(axis=1) for c in candidates]
                )
                distances = ((cur_centers - last_center) ** 2).sum(axis=1)
                index = np.argmin(distances)
                mask = candidates[index].mask[0]
            else:
                # do the confidence based selection
                index = np.argmax([p.score for p in candidates])
                mask = candidates[index].mask[0]
        else:
            mask = None

        return mask

    def obstruction_detection(self, action, action_dist):
        """
        change 'MoveAhead' action to a turn in case if it has failed previously
        """
        if action != "MoveAhead_25":
            return action

        if self._env.last_event.metadata["lastActionSuccess"]:
            return action

        idx_rotateR = self.vocab_action.word2index("RotateRight_90")
        idx_rotateL = self.vocab_action.word2index("RotateLeft_90")

        action = (
            "RotateLeft_90"
            if action_dist[idx_rotateL] > action_dist[idx_rotateR]
            else "RotateRight_90"
        )

        if self.hparams.debug:
            print("Blocking action is changed to: {}".format(action))

        return action

    def step(self, action_dict, obj_predictor=None, extractor=None):
        """
        environment step based on model prediction
        """
        info = {}

        mask = None
        action_str, obj_str = action_dict["action_str"], action_dict["obj_str"]
        action_idx, obj_idx = action_dict["action_idx"], action_dict["obj_idx"]

        # forward model
        if self.hparams.debug:
            print(f"Predicted action: {action_str}, obj: {obj_str}")

        action_dist, obj_dist = (
            action_dict["action_dist"],
            action_dict["target_obj_dist"],
        )

        obj = obj_str if self.has_interaction(action_str) else None

        if obj is not None:
            # get mask from a pre-trained RCNN
            assert obj_predictor is not None
            mask = self.extract_rcnn_pred(obj_idx, obj_predictor)

        # remove blocking actions
        action = self.obstruction_detection(action_str, action_dist.probs[-1])

        # use the predicted action
        episode_end = action == constants.STOP_TOKEN

        api_action = None

        # constants.TERMINAL_TOKENS was originally used for subgoal evaluation
        target_instance_id = ""

        if not episode_end:
            (
                step_success,
                _,
                target_instance_id,
                err,
                api_action,
            ) = self._env.va_interact(
                action,
                interact_mask=mask,
                smooth_nav=self.hparams.smooth_nav,
                debug=self.hparams.debug,
            )

            self._env.last_interaction = (obj, mask)
            if not step_success:
                self.num_fails += 1
                if self.num_fails >= self.hparams.max_fails:
                    if self.hparams.debug:
                        print(
                            "Interact API failed {} times; latest error '{}'".format(
                                self.num_fails, err
                            )
                        )
                    episode_end = True

            info.update(
                {
                    "step_success": step_success,
                    "num_fails": self.num_fails,
                    "target_instance_id": target_instance_id,
                    "api_action": api_action,
                    "err": err,
                }
            )

        next_obs = self.get_observation(extractor)
        reward = self._env.get_transition_reward()[0]
        done = episode_end

        return next_obs, reward, done, info
