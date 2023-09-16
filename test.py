import os.path as osp
from operator import itemgetter
from typing import Optional, Tuple

from mmengine import Config
from mmaction.apis import inference_recognizer, init_recognizer
from mmaction.visualization import ActionVisualizer

def perform_action_recognition(
    config: str,
    checkpoint: str,
    video: str,
    label: str,
    out_filename: str,
    fps: int = 30,
    font_scale: Optional[float] = None,
    font_color: str = 'white',
    target_resolution: Optional[Tuple[int]] = None,
) -> None:
    """Perform action recognition on a video.

    Args:
        config (str): Path to the model configuration file.
        checkpoint (str): URL or local path to the model checkpoint.
        video (str): Path to the input video file.
        label (str): Path to the label file.
        out_filename (str): Output filename for the generated file.
        fps (int): Number of picture frames to read per second. Defaults to 30.
        font_scale (float): Font scale of the text. Defaults to None.
        font_color (str): Font color of the text. Defaults to 'white'.
        target_resolution (Tuple[int], optional): Set to
            (desired_width, desired_height) to have resized frames.
            If either dimension is None, the frames are resized by keeping
            the existing aspect ratio. Defaults to None.
    """

    if video.startswith(('http://', 'https://')):
        raise NotImplementedError

    # Load the model configuration
    cfg = Config.fromfile(config)

    # Build the recognizer from the configuration and checkpoint
    model = init_recognizer(cfg, checkpoint, device='cuda:0')
    pred_result = inference_recognizer(model, video)

    pred_scores = pred_result.pred_scores.item.tolist()
    score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
    score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
    top5_label = score_sorted[:5]

    labels = open(label).readlines()
    labels = [x.strip() for x in labels]
    results = [(labels[k[0]], k[1]) for k in top5_label]

    print('The top-5 labels with corresponding scores are:')
    for result in results:
        print(f'{result[0]}: ', result[1])

    if out_filename is not None:

        if target_resolution is not None:
            if target_resolution[0] == -1:
                assert isinstance(target_resolution[1], int)
                assert target_resolution[1] > 0
            if target_resolution[1] == -1:
                assert isinstance(target_resolution[0], int)
                assert target_resolution[0] > 0
            target_resolution = tuple(target_resolution)

        # Initialize the visualizer
        out_type = 'gif' if osp.splitext(out_filename)[1] == '.gif' else 'video'
        visualizer = ActionVisualizer()
        visualizer.dataset_meta = dict(classes=labels)

        text_cfg = {'colors': font_color}
        if font_scale is not None:
            text_cfg.update({'font_sizes': font_scale})

        visualizer.add_datasample(
            out_filename,
            video,
            pred_result,
            draw_pred=True,
            draw_gt=False,
            text_cfg=text_cfg,
            fps=fps,
            out_type=out_type,
            out_path=osp.join('demo', out_filename),
            target_resolution=target_resolution)

        print(f'輸出檔案：{out_filename}')
