import gym
import gym_deepmimic
import argparse
import torch
import sys
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

###Global variables needed for GLUT
win_width = 800
win_height = int(win_width * 9.0 / 16.0)
reshaping = False
display_anim_time = int(1000 * 1 / 30)
gym_env = None
policy = None
animating = False


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # ==============General Arguments==============#

    parser.add_argument('--deepmimic', type=str, help='arguments for creating DeepMimicEnv', default='amp_args.txt')

    parser.add_argument('--model_path', type=str, default='expert_test.pt')
    # ==============Reset args==============#
    parser.add_argument('--custom_time', action='store_true',
                        help='When true, resets to random time between time_min and time_max')
    parser.add_argument('--time_min', default=0, type=float, help='lower bound on reset time')
    parser.add_argument('--time_max', default=0, type=float, help='upper bound on reset time')
    parser.add_argument('--no_resolve', action='store_true',
                        help='Resolve flag tells DeepMimicCore whether to resolve ground interesections when resolving. In most cases, should resolve')

    parser.add_argument('--noise_bef_rot', action='store_true',
                        help='There are two methods used to add randomness to state when resetting. This flag controls the order. Check DeepMimicCore KinCharacter.cpp for info on the two methods')
    parser.add_argument('--noise_min', type=float, default=0, help='Lower bound on amount of noise added to state')
    parser.add_argument('--noise_max', type=float, default=0, help='Upper bound on amount of noise added to state')
    parser.add_argument('--radian', type=float, default=0,
                        help='Amount of allowed rotation (in both positive and negative direction) when adding noise')

    parser.add_argument('--rot_vel_w_pose', action='store_true',
                        help='When true, any random roation applied during reset to pose is also applied to velocity')
    parser.add_argument('--vel_noise', action='store_true', help='When true, random noise added to velocity as well')
    parser.add_argument('--interp', type=float, default=1,
                        help='A float between [0,1] indicating how to initialize velocity during reset. 1 means to reset to expert. 0 is zero velocity')
    parser.add_argument('--knee_rot', action='store_true', help='When true, random noise added to knees in state')
    return parser.parse_args()


def load_policy(args):
    # TODO: Load model here
    return torch.load(args.model_path)


def step_policy(policy, state):
    # TODO: Get action from policy
    a, agent_info = policy.get_action(state)
    return agent_info['evaluation']


def init_draw():
    glutInit()

    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(win_width, win_height)
    glutCreateWindow(b'DeepMimic')

    return


def update_intermediate_buffer():
    global gym_env
    if not (reshaping):
        if (win_width != gym_env.deepmimic.get_win_width() or win_height != gym_env.deepmimic.get_win_height()):
            gym_env.deepmimic.reshape(win_width, win_height)
    return


def draw():
    global reshaping
    global gym_env
    update_intermediate_buffer()
    gym_env.deepmimic.draw()

    glutSwapBuffers()
    reshaping = False

    return


def reshape(w, h):
    global reshaping
    global win_width
    global win_height

    reshaping = True
    win_width = w
    win_height = h

    return


def setup_draw():
    global gym_env
    glutDisplayFunc(draw)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutTimerFunc(display_anim_time, animate, 0)

    reshape(win_width, win_height)
    gym_env.deepmimic.reshape(win_width, win_height)

    return


def shutdown():
    global gym_env

    Logger.print('Shutting down...')
    gym_env.deepmimic.shutdown()
    sys.exit(0)
    return


def reset():
    global gym_env
    gym_env.reset()
    return


def step_forward():
    global gym_env
    global policy
    ####Put step code here
    action = step_policy(policy, gym_env.ob)

    next_state, reward, done, infos = gym_env.step(action)
    if done:
        reset()
    glutPostRedisplay()


def keyboard(key, x, y):
    global gym_env
    key_val = int.from_bytes(key, byteorder='big')
    gym_env.deepmimic.keyboard(key_val, x, y)

    if (key == b'\x1b'):  # escape
        shutdown()
    elif (key == b'r'):
        reset();
    elif (key == b't'):
        toggle_animate()
    elif (key == b' '):
        step_forward()

    glutPostRedisplay()
    return


def animate(callback_val):
    if animating:
        step_forward()
        glutTimerFunc(display_anim_time, animate, 0)
    return


def toggle_animate():
    global animating
    animating = not animating
    if animating:
        glutTimerFunc(display_anim_time, animate, 0)
    return


def main():
    global gym_env
    global policy
    args = get_args()
    init_draw()
    reset_args = dict(custom_time=args.custom_time, time_min=args.time_min, time_max=args.time_max,
                      resolve=not args.no_resolve,
                      noise_bef_rot=args.noise_bef_rot, noise_min=args.noise_min, noise_max=args.noise_max,
                      radian=args.radian, rot_vel_w_pose=args.rot_vel_w_pose, vel_noise=args.vel_noise,
                      interp=args.interp, knee_rot=args.knee_rot)
    gym_env = gym.make('deepmimic-v0', deepmimic_args=args.deepmimic, enable_draw=True, reset_args=reset_args)
    reset()

    policy = load_policy(args)
    setup_draw()

    glutMainLoop()


if __name__ == '__main__':
    main()
