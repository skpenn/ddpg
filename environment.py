import numpy as np
import cv2
import time
import os

replay_place = ((1500, 1660), (440, 730))
score_start = (199, 118)
score_shape = (96, 80)
state_place = ((600, 1624), (28, 1052))
start_tap = (550, 550)
restart_tap = (550, 1550)
boundage = (30, 10)
tap_str = 'adb shell input tap {} {}'
swipe_str = 'adb shell input swipe {0} {1} {0} {1} {2}'
threshold = 0.8
score_range = 30


class Environment(object):
    def __init__(self):
        self.state_shape = [256, 256, 3]
        self.action_size = 1
        self._img = None
        self.inGame = True
        self.score = 0
        self._finish_template = cv2.imread("data/replay.png")
        self._score_template = cv2.imread("data/all.png")

    def __call__(self, action: list = None):
        if not self.inGame:
            x, y = self._rand_tap(restart_tap[0], restart_tap[1], boundage[0], boundage[1], bound=True)
            print("Tapped restart")
            os.system(tap_str.format(x, y))
            time.sleep(2)
            self._fetch_img()
            while self._isFinished():
                time.sleep(5)
                x, y = self._rand_tap(restart_tap[0], restart_tap[1], boundage[0], boundage[1], bound=True)
                print("Tapped restart")
                os.system(tap_str.format(x, y))
                self._fetch_img()
            score = self._get_score()
            if score != 0:
                self.reset()
            else:
                self.score = 0
                self.inGame = True
        rewards = 0
        _done = False
        if action is not None:
            act = action[0]
            if act > 1:
                act = 1
            if act > 0.2:
                x, y = self._rand_tap(start_tap[0], start_tap[1], boundage[0], boundage[1])
                os.system(swipe_str.format(int(x), int(y), int(action[0] * 1000)))
                time.sleep(action[0])
                self._fetch_img()
                score = self._get_score()
                if score == self.score:
                    time.sleep(1)
                    self._fetch_img()
                    if self._isFinished():
                        rewards = -1
                        self.inGame = False
                        _done = True
                    else:
                        rewards = 0
                elif score > self.score:
                    rewards = (score - self.score) / score_range
                    self.score = score
                else:
                    raise Exception()
            else:
                time.sleep(2)
                self._fetch_img()
                score = self._get_score()
                rewards = (score - self.score) / score_range
                self.score = score
        return cv2.resize(self._img[state_place[0][0]:state_place[0][1], state_place[1][0]:state_place[1][1]],
                           tuple(self.state_shape[:2])), rewards, _done

    def reset(self) -> bool:
        print("Tapped start")
        os.system('adb shell input keyevent 4')
        x, y = self._rand_tap(start_tap[0], start_tap[1], boundage[0], boundage[1], bound=True)
        os.system(tap_str.format(x, y))
        time.sleep(1)
        self._fetch_img()
        while self._isFinished():
            time.sleep(5)
            x, y = self._rand_tap(restart_tap[0], restart_tap[1], boundage[0], boundage[1], bound=True)
            print("Tapped restart")
            os.system(tap_str.format(x, y))
            self._fetch_img()
        score = self._get_score()
        if score >= 0:
            self.score = score
            self.inGame = True
            return True
        return False

    def _isFinished(self) -> bool:
        sub_img = self._img[replay_place[0][0]:replay_place[0][1], replay_place[1][0]:replay_place[1][1]]
        res = cv2.matchTemplate(sub_img, self._finish_template, cv2.TM_CCOEFF_NORMED)
        _, max, _, _ = cv2.minMaxLoc(res)
        return max > threshold

    @staticmethod
    def _rand_tap(x, y, x_stddev, y_stddev, bound=False):
        x_n = x + np.random.normal(0, x_stddev, 1)[0]
        y_n = y + np.random.normal(0, y_stddev, 1)[0]
        if bound:
            if not x - x_stddev * 3 <= x_n <= x + x_stddev * 3:
                x_n = x
            if not y - y_stddev * 3 <= y_n <= y + y_stddev * 3:
                y_n = y
        return x_n, y_n

    def _fetch_img(self):
        os.system('adb shell screencap -p /sdcard/1.png')
        os.system('adb pull /sdcard/1.png data/autoplay.png')
        time.sleep(0.15)
        self._img = cv2.imread("data/autoplay.png")

    def _get_score(self) -> int:
        score = -1
        start = list(score_start)
        while True:
            sub_img = self._img[start[0]:start[0] + score_shape[0], start[1]:start[1] + score_shape[1]]
            res = cv2.matchTemplate(self._score_template, sub_img, cv2.TM_CCOEFF_NORMED)
            _, maxv, _, maxl = cv2.minMaxLoc(res)
            if maxv < threshold:
                break
            number = (maxl[0] + score_shape[1] / 2) // score_shape[1]
            if score == -1:
                score = number
            else:
                score = score * 10 + number
            start[1] += score_shape[1]
        return score


if __name__ == "__main__":
    def rand_agent():
        return np.random.normal(0.6, 0.4 / 3, 1)


    env = Environment()
    env.reset()
    for _ in range(100):
        print("start!")
        _, reward, done = env()
        while not done:
            print("reward: {}".format(reward))
            action = rand_agent()
            print("action: {}".format(action))
            _, reward, done = env(action)
        print("failed!, reward: {}".format(reward))
