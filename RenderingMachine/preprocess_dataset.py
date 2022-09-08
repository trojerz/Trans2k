from PIL import Image
import os

video_dirs = list(sorted([os.path.join('GOT10k', p) for p in os.listdir('GOT10k')]))
print(video_dirs)

for vid_ in video_dirs:
    vid_dir = vid_ + '/'
    images = list(sorted([os.path.join(vid_dir, p) for p in os.listdir(vid_dir) if p.lower().endswith('jpg')]))

    for vid_path in images:
        img = Image.open(vid_path)
        if img.size != (1280, 720):
            img = img.resize((1280, 720), Image.ANTIALIAS)
            img.save(vid_path)