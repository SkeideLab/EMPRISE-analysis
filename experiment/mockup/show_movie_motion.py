from psychopy import visual, constants, event, core, gui
import os
import serial
import glob

expInfo = {'movie name': 1, 'port number':''}

dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False,
                      title='Movie with Motion Detection')
if dlg.OK == False:
    core.quit()

moviename = expInfo['movie name']
videos = glob.glob(
    f"{os.path.dirname(os.path.abspath(__file__))}/../videos/**/*.mp4", recursive=True)
if len(videos) < 1:
    print('ERROR! No videos found! Terminating...')
    core.quit()
# this works
videos.sort(key=lambda path: path.split(os.path.sep)[-1].split('_')[0])

videopath = videos[moviename-1]

if not os.path.exists(videopath):
    raise RuntimeError("Video File could not be found:" + videopath)

print('path found')


# # -------------------serial port------------------------------------------

# print('**********************************************************')
# print('ACHTUNG: Bitte Knöpfe nicht drücken!')
# print('ACHTUNG: Bitte Motion Sensor bewegen!')
# print('***********************************************************')
# # make list with one dict per port. Dict contains port and boolean that controls if it will be used
# sers = [{'ser': serial.Serial(
#     f"/dev/ttyUSB{ser_num}", timeout=0), 'use': False} for ser_num in [0, 1]]

# # while we have no response from either port
# while not any([ser['use'] for ser in sers]):
#     for ser in sers:
#         resp = list(ser['ser'].read())
#         if len(resp) > 0:
#             ser['use'] = True

# # close port that will not be used
# [ser.close() for ser in sers if ser['use'] == False]
# # assign port that will be used to ser
# ser = [ser for ser in sers if ser['use'] == True][0]

ser=serial.Serial(f"/dev/ttyUSB{expInfo['port number']}", timeout=0)
win = visual.Window(fullscr=True)  # (800, 600))
mov = visual.MovieStim2(win, videopath)
print('orig movie size=%s' % mov.size)
print('duration=%.2fs' % mov.duration)
globalClock = core.Clock()
print(f"Time: {globalClock.getTime()}")

while mov.status != constants.FINISHED:
    mov.draw()
    win.flip()
    if event.getKeys(keyList=['escape']):
        break
    if len(list(ser.read())) > 0:
        mov.pause()
        core.wait(1)
        ser.reset_input_buffer()
        mov.play()

ser.close()


# ---------------------end-------------------------------------------
win.close()
core.quit()
