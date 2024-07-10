import sys
CENTERTRACK_PATH = r'C:\Users\Admin\Desktop\forkedProject\CenterTrack\src\lib'
sys.path.insert(0, CENTERTRACK_PATH)

from detector import Detector
from opts import opts

MODEL_PATH = r'C:\Users\Admin\Desktop\forkedProject\CenterTrack\models\coco_tracking.pth'
TASK = 'tracking' # or 'tracking,multi_pose' for pose tracking and 'tracking,ddd' for monocular 3d tracking
opt = opts().init('{} --load_model {}'.format(TASK, MODEL_PATH).split(' '))

detector = Detector(opt)

images = [r"C:\Users\Admin\Desktop\forkedProject\CenterTrack\videos\nuscenes_mini.mp4"]
for img in images:
  ret = detector.run(img)['results']