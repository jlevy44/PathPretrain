import fire
from .train_model import train_model

def predict(inputs_dir='inputs_training',
                    crop_size=224,
                    resize=256,
                    mean=[0.5, 0.5, 0.5],
                    std=[0.1, 0.1, 0.1],
                    num_classes=2,
                    architecture='resnet50',
                    batch_size=32,
                    model_save_loc='saved_model.pkl',
                    predictions_save_path='predictions.pkl',
                    predict_set='test',
                    gpu_id=-1,
                    tensor_dataset=False,
                    pickle_dataset=False,
                    label_map=dict(),
                    semantic_segmentation=False
                    ):
    train_model(inputs_dir=inputs_dir,
                        crop_size=crop_size,
                        resize=resize,
                        mean=mean,
                        std=std,
                        num_classes=num_classes,
                        architecture=architecture,
                        batch_size=batch_size,
                        predict=True,
                        model_save_loc=model_save_loc,
                        predictions_save_path=predictions_save_path,
                        predict_set=predict_set,
                        verbose=True,
                        gpu_id=gpu_id,
                        tensor_dataset=tensor_dataset,
                        pickle_dataset=pickle_dataset,
                        label_map=label_map,
                        semantic_segmentation=semantic_segmentation)

def main():
    fire.Fire(predict)

if __name__=="__main__":
    main()
