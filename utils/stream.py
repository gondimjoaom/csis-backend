import cv2
import subprocess as sp

class Streamer:
    def __init__(self, frame):
        rtsp_server = 'rtsp://0.0.0.0:8080/csis' # push server (output server)
        
        #pull rtsp data, or your cv cap.  (input server)
        # cap = cv2.VideoCapture(
        #     'rtsp://example.org:554/pull from me ')
        
        h,w = frame.shape[:2]
        sizeStr = str(int(w)) + \
            'x' + str(int(h))
        fps = int(15)
        
        command = ['ffmpeg',
                '-re',
                '-s', sizeStr,
                '-r', str(fps),  # rtsp fps (from input server)
                '-i', '-',
                
                # You can change ffmpeg parameter after this item.
                '-pix_fmt', 'yuv420p',
                '-r', '30',  # output fps
                '-g', '50',
                '-c:v', 'libx264',
                '-b:v', '2M',
                '-bufsize', '64M',
                '-maxrate', "4M",
                '-preset', 'veryfast',
                '-rtsp_transport', 'tcp',
                '-segment_times', '5',
                '-f', 'rtsp',
                rtsp_server]

        self.process = sp.Popen(command, stdin=sp.PIPE)
    def sendFrame(self, frame):
    # while(cap.isOpened()):
        # ret, frame = cap.read()
        ret2, frame2 = cv2.imencode('.png', frame)
        self.process.stdin.write(frame2.tobytes())