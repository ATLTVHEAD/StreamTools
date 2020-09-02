call C:\ProgramData\Anaconda3\Scripts\activate.bat
call conda activate yolact
call cd C:\Users\nated\Documents\Python Scripts\Cloned Repo\yolact
call python eval.py --trained_model=weights/yolact_resnet50_54_800000.pth --score_threshold=0.15 --top_k=15 --video_multiframe=18 --display_scores=False --display_text=False --greenscreen=True --video="rtmp://192.168.1.193/live/test"