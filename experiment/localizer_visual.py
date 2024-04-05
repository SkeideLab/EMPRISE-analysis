# Import Packages, Libraries and Files

from psychopy import visual, core, event, __version__, data, gui, logging, monitors,constants
from psychopy.hardware import keyboard
from psychopy.hardware.emulator import launchScan
import os
import numpy as np
import random
import show_movie

"""
The active condition consisted of two complex figures (Rey-Osterrieth figure; Rey, 1941) 
that were shown on the screen and children had to decide whether they were completely identical 
or whether a small part was missing in one of them. If so, a button press was required. Children 
were specifically instructed to “try to find the errors”, but not to worry should they not have 
been done in time before the next images appeared. The control condition again showed two versions 
of the complex figure, but in this case, the orientation (identical or different) needed to be assessed; 
if the orientation was different, a button press was required. In order to make sure the children knew 
when to do which part of the task, the background was shown in blue in the control condition. As in 
the VIT the screen turned green for 0.5 s between conditions to indicate the beginning of the other 
condition. Again a block design was used with 6 blocks of the control and 5 blocks of the active 
condition. Each block contained 5 images that were presented in randomized order for 5.5 s each 
with a pause of 0.5 s (when only the fixation cross and the reminders were shown).
"""


# TODO test on linux!
def prepare_visual_stim(exp_duration, stim_duration, win):
    pictures = {}
    pictures2 = []
    dir = os.path.dirname(os.path.abspath(__file__))
    print('Entering directory', dir)
    variation_num = [0, 1, 2, 3, 4]
    
    # 1, 2, 3, 4 are the variations where parts of the image are deleted
    # this corresponds to the active condition
    # 60, 90, 120, 270 are the degrees by which the original image is rotated
    # for the control condition
    visual_dir = os.path.join(dir, 'localizer_stims')
    original_filepath = os.path.join(
        visual_dir, f'localizer_VST_variation_0.png')

    pictures['orig1'] = visual.ImageStim(win, image=original_filepath,
                                ori=0, pos=(2.25, 0), colorSpace='rgb', opacity=1, flipHoriz=False, flipVert=False, 
                                interpolate=True, color=[1, 1, 1], size=6, name='/'.join(original_filepath.split(os.path.sep)[-2:]))

    pictures['orig2'] = visual.ImageStim(win, image=original_filepath,
                                ori=0, pos=(-2.25, 0), colorSpace='rgb', opacity=1, flipHoriz=False, flipVert=False, 
                                interpolate=True, color=[1, 1, 1], size=6, name='/'.join(original_filepath.split(os.path.sep)[-2:]))
    pictures['changed_vers']=[]
    pictures['blue1']=visual.ImageStim(win, image=os.path.join(visual_dir,'localizer_VST_variation_blue.png'),
                                ori=0, pos=(-2.25, 0), colorSpace='rgb', opacity=1, flipHoriz=False, flipVert=False, 
                                interpolate=True, color=[1, 1, 1], size=6, name=os.path.join(visual_dir,'localizer_VST_variation_blue.png'))
    pictures['blue2']=visual.ImageStim(win, image=os.path.join(visual_dir,'localizer_VST_variation_blue.png'),
                                ori=0, pos=(-2.25, 0), colorSpace='rgb', opacity=1, flipHoriz=False, flipVert=False, 
                                interpolate=True, color=[1, 1, 1], size=6, name= os.path.join(visual_dir,'localizer_VST_variation_blue.png'))
    #original.autoDraw = False
    for var in variation_num:
        filepath = os.path.join(
            visual_dir, f'localizer_VST_variation_{var}.png')

        stim = visual.ImageStim(win, image=filepath, ori=0, pos=(-2.25, 0), colorSpace='rgb',
                                opacity=1, flipHoriz=False, flipVert=False, interpolate=True, color=[1, 1, 1],
                                size=6, name='/'.join(filepath.split(os.path.sep)[-2:]))
        # stim.autoDraw = False

        # filepath2 = os.path.join(
        #     visual_dir, f'localizer_VST_variation_{var1}.png')
        # stim2 = visual.ImageStim(win, image=filepath2, ori=0, pos=(-2.25, 0), colorSpace='rgb',
        #                          opacity=1, flipHoriz=False, flipVert=False, texRes=128, interpolate=True, color=[1, 1, 1],
        #                          size=(4, 4), name='/'.join(filepath2.split(os.path.sep)[-2:]))
        # stim2.autoDraw = False

        pictures['changed_vers'].append( stim)
        #pictures.append({'stim': stim2, 'condition': 'control'})
    # num_unique_stim = len(pictures)
    # dur_unique_stim = num_unique_stim * stim_duration
    # num_stim = int(exp_duration / stim_duration)
    # n_reps, rest_stims = divmod(num_stim, num_unique_stim)
    # print(n_reps)
    # print(exp_duration - n_reps*num_unique_stim * stim_duration)
    # # if the duration of the rest time is >1
    # if exp_duration - n_reps*num_unique_stim * stim_duration > 1:
    #     n_reps += 1

    # trials = data.TrialHandler(
    #     trialList=pictures, nReps=n_reps, method='random', name='trials')

    return pictures

def FixationCross(win):
    """Draws a small white fixation cross on the window win. This will autodrawn forever even if win.flip is called.

    Args:
        win (psychopy.visual.window): window on screen
    """
    Line_1 = visual.Line(
        win=win, name='Line_1',
        start=(-0.2, -0.2), end=(0.2, .2),
        lineWidth=1.5, lineColor='white',
        depth=0)
    Line_2 = visual.Line(
        win=win, name='Line_2',
        start=(-.2, .2), end=(.2, -.2),
        lineWidth=1.5, lineColor='white', colorSpace='rgb',
        depth=0)

    return [Line_1,Line_2]


def present(trials,pictures,trial_clock,exp_params,win,fix_cross):

    times={}
    pos=[[3.5,0],[-3.5,0]]
    random.shuffle(pos)
    trial_clock.add(exp_params['stim_duration'])

        
    if trials.thisTrial['condition']=='active':
        print('active, ',end='')
        p1=pictures['orig1']
        exp_params['rec'].color='gray'
        if trials.thisTrial['truth']=='same':
            p2=pictures['orig2']
            print('same')
        else:
            p2=random.choice(pictures['changed_vers'])
            print('different')
        
    else:
        p1=pictures['blue1']
        p2=pictures['blue2']
        if trials.thisTrial['truth']=='same':
            p2.ori=0
        else:
            p2.ori=random.choice([60, 80, 180, 270])

        exp_params['rec'].color='lightskyblue'

    p1.pos=pos[0]
    p2.pos=pos[1] 
    p1.autoDraw=True
    p2.autoDraw=True
    win.timeOnFlip(times,'t_start')
    win.callOnFlip(exp_params['stim_clock'].reset)
    win.callOnFlip(exp_params['stim_keyboard'].getKeys, keyList=
            exp_params['button'])
    win.flip()
    answered=False
    while trial_clock.getTime() > 0:
        # it is ok to not check response buttons during this time because
        # there is no immediate feedback and the response time 
        # of any button presses is saved
        if not answered:
            ans = exp_params['stim_keyboard'].getKeys(
                keyList=exp_params['button'])
            if ans: 
                print(f'keypress, {ans[0].rt}')
                trials.addData('RT',ans[0].rt)
                answered=True
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
            

    p1.autoDraw=False
    p2.autoDraw=False
    [c.setAutoDraw(True) for c in fix_cross]

    if trials.nRemaining >0:
        if trials.trialList[trials.thisTrialN+1]['condition']!=trials.thisTrial['condition']:
            exp_params['rec'].color='green'
   
    win.timeOnFlip(times,'t_stop')
    trial_clock.add(exp_params['fix_duration'])
    win.flip()
    
    while trial_clock.getTime() > 0:
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()


    # end of this sample and fix period
    [c.setAutoDraw(False) for c in fix_cross]
    [trials.addData(key,value) for key,value in times.items()]
    [print(f'{key}: {value}') for key,value in times.items()]
    

if __name__ == '__main__':
    # Ensure that relative paths start from the same directory as this script
    _thisDir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(_thisDir)
    # Store info about the experiment session
    # right hand button is non-match like in main exp, should press 'b'
    expName = 'localizer-visual'
    expInfo = {'participant': '', 'session': '','button':['b'], 'show movie': True,
               'movie name': 1}

    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName,fixed=['button'])
    if dlg.OK == False:
        core.quit()

     #---MRI-Scanner-Interface---#
    # settings for launchScan:
    MR_settings = {
        'TR': 2.000,     # duration (sec) per whole-brain volume
        # number of whole-brain 3D volumes per scanning run (for simulation mode!)
        'volumes': 5,
        # character to use as the sync timing event; assumed to come at start of a volume
        'sync': 't',
        # number of volumes lacking a sync pulse at start of scan (for T1 stabilization)
        'skip': 0,
        'sound': False    # in test mode: play a tone as a reminder of scanner noise
    }

    expInfo['date'] = data.getDateStr(format='%Y_%m_%d_%H%M')
    expInfo['psychopy version'] = __version__

    # Clocks
    global_clock = core.Clock()
    trial_clock = core.CountdownTimer()

    Monitor_data = [1024, 768, 40, 118]

    # this saves the monitor config to a psychopy settings file so we can use it later
    mon = monitors.Monitor(
        'stimulus_screen', width=Monitor_data[2], distance=Monitor_data[3])
    mon.setSizePix((Monitor_data[0], Monitor_data[1]))
    mon.save()

    # LOGGING -------------------------------------------------------------
    out_path = os.path.join(_thisDir, 'logfiles')
    # if not os.path.exists(out_path):
    #     os.mkdir(out_path)
    # filename = os.path.join(out_path, u'sub-%s_ses-%s_%s_%s' %
    #                         (expInfo['participant'], expInfo['session'], expName, expInfo['date']))
    for identifier in ['participant', 'session']:
            if not expInfo[identifier]:
                expInfo[identifier] = '_'
    out_path = os.path.join(
            out_path, expInfo['session'], f"{expInfo['participant']}_d-{data.getDateStr(format='%Y%m%d')}")

    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    run_id = u'sub-%s_ses-%s_%s_%s' % (
                    expInfo['participant'], expInfo['session'], expName, data.getDateStr(format='%Y_%m_%d_%H%M'))
    print(f"Now running: {run_id}")
    filename = os.path.join(out_path, run_id)
    logFile = logging.LogFile(filename+'.log', level=logging.EXP)
    logging.console.setLevel(logging.WARNING)
    logging.setDefaultClock(global_clock)
    logging.exp(expInfo)
    logging.exp(MR_settings)
    defaultKeyboard = keyboard.Keyboard()
    win = visual.Window(
        size=(Monitor_data[0], Monitor_data[1]), fullscr=True, screen=0,
        winType='pyglet', allowGUI=False,
        monitor='stimulus_screen', color='grey', colorSpace='rgb',
        blendMode='avg', useFBO=True,
        units='deg', name='localizer_window')
    fix_cross=FixationCross(win)
    # timing in seconds
    exp_params={'exp_duration' : 120, #30, #120
    'stim_duration' : 5.5,
    'fix_duration' : 0.5,
    'block_size' : 5} #1} #5
    exp_params['block_duration'] = (exp_params['stim_duration'] + exp_params['fix_duration']) * exp_params['block_size']
    exp_params['num_blocks'] = exp_params['exp_duration']//exp_params['block_duration']
    whole, rest = divmod(exp_params['num_blocks'], 2)
    exp_params['num_control'] = exp_params['num_active'] = int(whole)

    blocks=[cond for z_tuple in zip(
        ['active'] * exp_params['num_active'], ['control'] * exp_params['num_control']) for cond in z_tuple]
    if rest != 0:
        exp_params['num_active'] += 1
        blocks.append('active')
    # expand blocks in such a way that a condition block now is multiple samples of the condition
    blocks_exp=[[cond]*exp_params['block_size'] for cond in blocks]

    # this gives the condition for every sample
    samples_cond=[item for sublist in blocks_exp for item in sublist]
    # this gives the truth, i. e. same or different imgs for every sample
    samples_truth=random.choices(['same','different'],k=len(samples_cond))
    # make a dict to be read in by trialhandler
    samples_dict=[{'condition':cond,'truth':truth} for cond,truth in zip(samples_cond,samples_truth)]

    trials = data.TrialHandler(
        trialList=samples_dict, nReps=1, method='sequential', name='trials')

    exp_params['stim_clock'] = core.Clock()
    exp_params['stim_keyboard'] = keyboard.Keyboard(
        clock=global_clock)#exp_params['stim_clock'])
    exp_params['button']=expInfo['button']

    exp_params['rec'] = visual.Rect(win,size=[2,2],units='norm')#visual.Circle(win, radius=7.5, edges=64, lineWidth=0, 
                         #fillColor='black', colorSpace='rgb', autoDraw=False, name='background_circle')
    pictures = prepare_visual_stim(
        exp_params['exp_duration'], exp_params['stim_duration'], win)
    #sounds = prepare_audio_stim()
    ############################################################################
    vol = launchScan(win, MR_settings, globalClock=global_clock,mode='Scan')
    ############################################################################
    exp_params['rec'].autoDraw=True
    # cir.draw()
    # FixationCross(win)
    # win.flip()
    print('Trials ******')
    # print(trials)
    print(trials.nTotal)
    print(trials.trialList)
    start_time = global_clock.getTime()
    trial_clock.reset()
    #trials.next()
    while trials.nRemaining>0:
        trials.next()
        present(trials,pictures,trial_clock,exp_params,win,fix_cross)
        
    #distribute_trials(trials, original_picture, cir)
    end_time = global_clock.getTime()
    run_time = end_time - start_time
    print('full run time: '+str(run_time))
    print(end_time)
    logging.exp('full run time: '+str(run_time))
    trials.saveAsWideText(fileName=filename+'_events.tsv')

    exp_params['rec'].autoDraw=False
    win.flip()
    while global_clock.getTime() < exp_params['exp_duration']+2:
        keys = event.getKeys(keyList=['escape'])
        # print(keys)
        if keys:
            print('goodbye')
            break

    if expInfo['show movie']:
        # if expInfo['movie name']=='':
        #     mov_name=r'C:\Program Files\PsychoPy\Lib\site-packages\psychopy\tests\data\testMovie.mp4'
        # else:
        
        [item.setAutoDraw(False) for item in fix_cross]
        show_movie.start_movie(win,expInfo['movie name'])
        # videopath = r'./jwpIntro.mov'
        # videopath = os.path.join(
        #     '/afs/cbs.mpg.de/software/pythonext/psychopy/2021.1.3/ubuntu-bionic-amd64/lib/python3.6/site-packages/psychopy/demos/coder/stimuli/', videopath)
        # # videopath=r'C:\Program Files\PsychoPy\Lib\site-packages\psychopy\tests\data\testMovie.mp4'
        # movies = {
        #     '': videopath
        # }

        # mov_name = movies[expInfo['movie name']]
        # mov = visual.MovieStim3(win, mov_name, size=(320, 240),
        #                         flipVert=False, flipHoriz=False, loop=False)
        # print('orig movie size=%s' % mov.size)
        # print('duration=%.2fs' % mov.duration)

        # while mov.status != constants.FINISHED:
        #     mov.draw()
        #     # print('frame')
        #     win.flip()
        #     keys = event.getKeys(keyList=['escape'])
        #     # print(keys)
        #     if keys:
        #         print('goodbye')
        #         break

    core.quit()