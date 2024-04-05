# Import Packages, Libraries and Files

from psychopy import visual, core, event, __version__, gui, monitors,constants
import os
import glob


def start_movie(win,moviename):
    
    videos=glob.glob(f"{os.path.dirname(os.path.abspath(__file__))}/videos/**/*.mp4",recursive=True)
    if len(videos)<1:
        print('ERROR! No videos found! Terminating...')
        core.quit()
    # this works
    videos.sort(key=lambda path: path.split(os.path.sep)[-1].split('_')[0])

    mov_name = videos[moviename-1]
    mov = visual.MovieStim3(win, mov_name,
                            flipVert=False, flipHoriz=False, loop=False)
    print('orig movie size=%s' % mov.size)
    print('duration=%.2fs' % mov.duration)

    #TODO make sure volume of video is ok
    while mov.status != constants.FINISHED:
        mov.draw()
        # print('frame')
        win.flip()
        keys = event.getKeys(keyList=['escape'])
        # print(keys)
        if keys:
            print('goodbye')
            break




def main():
    Monitor_data = [1024, 768, 40, 118]

    # this saves the monitor config to a psychopy settings file so we can use it later
    mon = monitors.Monitor(
        'stimulus_screen', width=Monitor_data[2], distance=Monitor_data[3])
    mon.setSizePix((Monitor_data[0], Monitor_data[1]))
    mon.save()

    expInfo = {
               'movie name': 1}

    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title='Welcher Film soll abgespielt werden?')
    if dlg.OK == False:
        core.quit()

    
    win = visual.Window(
        size=(Monitor_data[0], Monitor_data[1]), fullscr=True, screen=0,
        winType='pyglet', allowGUI=False,
        monitor='stimulus_screen', color='grey', colorSpace='rgb',
        blendMode='avg', useFBO=True,
        units='deg', name='localizer_window')

    start_movie(win,expInfo['movie name'])
    
    core.quit()
    



if __name__ == '__main__':
    main()
    