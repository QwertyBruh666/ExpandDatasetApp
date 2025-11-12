import sys
from ultralytics.data.annotator import auto_annotate


def coco_annotate_it(input_dir: str, det_model: str = 'yolo11x.pt', sam_model: str = 'sam2.1_b.pt', output_dir: str = 'texts_dir', min_conf: str = '0.35'):
    auto_annotate(det_model=det_model,
                  sam_model=sam_model,
                  output_dir=output_dir,
                  data=input_dir,
                  conf=float(min_conf))



def main():
    argv = sys.argv
    coco_annotate_it()
    pass


if __name__ == '__main__':
    main()