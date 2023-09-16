from mmaction.apis import inference_recognizer, init_recognizer
from test import perform_action_recognition

def getData(filename):
    config_path = 'mmaction2/configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py'
    checkpoint_path = 'https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth' # 可以是本地路径
    img_path = './videos/' + filename   # 您可以指定自己的图片路径
    label_path='kinetics_400_labels.csv'
    out_filename=filename

    perform_action_recognition(config_path,checkpoint_path,img_path,label_path,out_filename)

    return 'import success'

if __name__ == '__main__':
    getData('demo.mp4')