from utils.video_utils import (read_video, 
                   save_video)
from trackers.player_tracker import PlayerTracker
from trackers.ball_tracker import BallTracker


def main():
    input_video_path = "inputs/input_video.mp4"
    video_frames = read_video(input_video_path)
    # detect players and ball
    player_tracker = PlayerTracker(model_path='yolov8x')
    
    ball_tracker = BallTracker(model_path='models\yolo_last .pt')
    
    
    player_detections = player_tracker.detect_frames(video_frames, 
                                                     read_from_stub=True,
                                                     stub_path='tracker_stubs/player_detections.pkl')
    ball_detections = ball_tracker.detect_frames(video_frames, 
                                                     read_from_stub=True,
                                                     stub_path='tracker_stubs/ball_detections.pkl')
    
    
    
    ## Draw player Bounding boxes
    
   
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(video_frames, ball_detections)
    
    save_video(video_frames, "output_videos/output_video.avi")
    
if __name__ == "__main__":
    main() 