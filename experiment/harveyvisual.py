# import self-written scripts to get random positions and manage exp
import exp_utils


LOCATION = '7tscanner'

if __name__ == '__main__':
    exp_utils.run_exp(LOCATION, exp_name='harveyvisual', train=False)
