import requests

resp = requests.get('http://vision.cs.unc.edu/jielei/tvqa/frames/', auth=('jielei', 'ZreAZ2GrwfEUj9Zr'))

s = requests.Session()

s.get('http://vision.cs.unc.edu/jielei/tvqa/frames/')

s.get('http://vision.cs.unc.edu/jielei/tvqa/frames/tvqa_video_frames_fps3.tar.gz.ab')
