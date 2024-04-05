# import self-written scripts to get random positions and manage exp
import exp_utils


LOCATION = '7tscanner'
EXPNAME = 'priming'

# progress through the 'waiting for scanner...' screen by pressing 5
if __name__ == '__main__':
    exp_utils.run_exp(LOCATION, exp_name=EXPNAME)
