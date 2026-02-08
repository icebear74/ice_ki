# Assuming the videos array exists in the create_full_config.py file

def add_priority_to_videos(videos):
    for video in videos:
        video['priority'] = 255
    return videos

# Example usage
videos = [{'name': 'example_video.mp4'}, {'name': 'another_video.mp4'}]
videos_with_priority = add_priority_to_videos(videos)

# Now videos_with_priority will have priority=255 for each video