from psychopy import prefs  # noqa
prefs.hardware['audioLib'] = ['PTB', 'sounddevice', 'pyo', 'pygame']  # noqa
from psychopy import visual, core, event, sound, __version__, monitors
import os
from psychopy.hardware import keyboard
import random

# stop the script by pressing escape


def prepare_audio_stim(categories=['female_1', 'female_2', 'female_3',
                                   'male_1', 'male_2', 'male_3']):
    sounds = []

    directory = os.path.dirname(os.path.abspath(__file__))
    for category in categories:

        audio_dir = os.path.join(directory, 'audio', category)
        for filename in os.listdir(audio_dir):
            if ".wav" in filename:
                filepath = os.path.join(audio_dir, filename)
                sounds.append(sound.Sound(filepath))

    return sounds


def show_face(win):
    visual.Circle(win, radius=3, edges=32, lineWidth=0, lineColor=None,
                  lineColorSpace='rgb', fillColor='grey', fillColorSpace='rgb', autoDraw=True, name='feedback_circle')
    face = visual.ImageStim(
        win=win, image='localizer_stims/speaking_face.png', name='speaking_face', size=4)
    face.autoDraw = True
    win.flip()
    return face


def FixationCross(win):
    """Draws a small white fixation cross on the window win. This will autodrawn forever even if win.flip is called.

    Args:
        win (psychopy.visual.window): window on screen
    """
    Line_1 = visual.Line(
        win=win, name='Line_1',
        start=(-0.05, -0.05), end=(0.05, .05),
        units='height',
        lineWidth=1.5, lineColor='white',
        depth=0,
        autoDraw=True)
    Line_2 = visual.Line(
        win=win, name='Line_2',
        start=(-.05, .05), end=(.05, -.05),
        units='height',
        lineWidth=1.5, lineColor='white', colorSpace='rgb',
        depth=0,  # -1.0,
        autoDraw=True)

    win.flip()


if __name__ == '__main__':
    # Ensure that relative paths start from the same directory as this script
    _thisDir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(_thisDir)

    Monitor_data = [1024, 768, 33, 130]

    # this saves the monitor config to a psychopy settings file so we can use it later
    mon = monitors.Monitor(
        'stimulus_screen', width=Monitor_data[2], distance=Monitor_data[3])
    mon.setSizePix((Monitor_data[0], Monitor_data[1]))
    mon.save()

    win = visual.Window(
        size=(Monitor_data[0], Monitor_data[1]), fullscr=True, screen=0,
        winType='pyglet', allowGUI=False,
        monitor='stimulus_screen', color='black', colorSpace='rgb',
        blendMode='avg', useFBO=True,
        units='deg', name='audiotest_window')

    default_keyboard = keyboard.Keyboard()

    # win = visual.Window(
    #     fullscr=True, screen=0,
    #     winType='pyglet', allowGUI=False,
    #     color='black', colorSpace='rgb',
    #     blendMode='avg', useFBO=True,
    #     name='audio_test')

    sounds = prepare_audio_stim()

    # FixationCross(win)
    show_face(win)

while not event.getKeys(['escape']):

    audio = random.choice(sounds)
    audio.play()
    core.wait(1)

core.quit()
