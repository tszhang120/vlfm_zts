# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Optional

import numpy as np
import torch #张量运算和训练与推理
import torchvision.transforms.functional as F # 图像预处理的工具集，用于图像转换、归一化

from vlfm.vlm.detections import ObjectDetections # 存储目标检测的结果，边界框、置信度、类别信息

from .server_wrapper import ServerMixin, host_model, send_request, str_to_image

try:
    from groundingdino.util.inference import load_model, predict #加载模型、对图像进行预测
except ModuleNotFoundError:
    print("Could not import groundingdino. This is OK if you are only using the client.")

GROUNDING_DINO_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_WEIGHTS = "data/groundingdino_swint_ogc.pth"
CLASSES = "chair . person . dog ."  # Default classes. Can be overridden at inference.


class GroundingDINO:
    def __init__(
            self,
            config_path: str = GROUNDING_DINO_CONFIG, # 配置文件路径
            weights_path: str = GROUNDING_DINO_WEIGHTS, # 权重文件路径
            caption: str = CLASSES, # 目标识别
            box_threshold: float = 0.35, # 边界框的置信度阈值，默认0.35
            text_threshold: float = 0.25, # 文本匹配的置信度阈值，默认0.25
            device: torch.device = torch.device("cuda"),
    ):
        self.model = load_model(model_config_path=config_path, model_checkpoint_path=weights_path).to(device) #将模型移动到指定的设备
        self.caption = caption # 保存传入的caption
        self.box_threshold = box_threshold #
        self.text_threshold = text_threshold

    def predict(self, image: np.ndarray, caption: Optional[str] = None) -> ObjectDetections: # 接收一张np.ndarray格式的图像，和一个可选的caption参数，返回ObjectDetections对象
        """
        This function makes predictions on an input image tensor or numpy array using a
        pretrained model.

        Arguments:
            image (np.ndarray): An image in the form of a numpy array.
            caption (Optional[str]): A string containing the possible classes
                separated by periods. If not provided, the default classes will be used.

        Returns:
            ObjectDetections: An instance of the ObjectDetections class containing the
                object detections.
        """
        # Convert image to tensor and normalize from 0-255 to 0-1
        image_tensor = F.to_tensor(image) # 将图像转为pytorch张量
        image_transformed = F.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 归一化，这样就符合常见的ImageNet数据集的均值和标准差
        if caption is None: # 如果没有传入caption参数，就用类的默认caption，
            caption_to_use = self.caption
        else:
            caption_to_use = caption
        print("Caption:", caption_to_use) # 打印出来，方便调试
        with torch.inference_mode(): #使用torch.inference_mode()进行推理，关闭梯度计算来节省内存？？？这里是这句话就会关闭吗
            boxes, logits, phrases = predict( # 使用predict函数对图像进行推理，返回边界框、置信度、类别
                model=self.model,
                image=image_transformed,
                caption=caption_to_use,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
            )
        detections = ObjectDetections(boxes, logits, phrases, image_source=image) # 创建ObjectDetections保存推理结果

        # Remove detections whose class names do not exactly match the provided classes
        classes = caption_to_use[: -len(" .")].split(" . ") # 提取类别，然后调用filter_by_class方法？？？这里是怎么做的
        detections.filter_by_class(classes) # 过滤调不在指定类别中的检测结果

        return detections


class GroundingDINOClient:
    def __init__(self, port: int = 12181): # 连接到本地12181端口
        self.url = f"http://localhost:{port}/gdino"

    def predict(self, image_numpy: np.ndarray, caption: Optional[str] = "") -> ObjectDetections: # 将"图像"和"描述"通过send_request发送到服务器，服务器返回推理结果（JSON格式），
        response = send_request(self.url, image=image_numpy, caption=caption)
        detections = ObjectDetections.from_json(response, image_source=image_numpy) # 将JSON格式的推理结果转为ObjectDetections对象

        return detections


if __name__ == "__main__":
    import argparse  # 处理命令行参数

    parser = argparse.ArgumentParser()  # 创建ArgumentParser对象，argparse根据这个对象来解析命令行参数
    parser.add_argument("--port", type=int, default=12181)  # 给对象添加一个命令行参数、
    args = parser.parse_args()  # 将命令行传入的参数储存在args变量中

    print("Loading model...")


    class GroundingDINOServer(ServerMixin, GroundingDINO):  # 定义类，继承了ServerMixin（混入类，提供与服务器相关的功能，如http服务）和GroundingDINO（负责加载模型和推理）两个类
        def process_payload(self, payload: dict) -> dict: # 如何处理请求的有效负载，接收一个字典作为参数返回一个字典
            image = str_to_image(payload["image"]) # 从pyload中取出image的value，然后通过函数将其转换为图像数据
            return self.predict(image, caption=payload["caption"]).to_json() # 返回一个ObjectDetections对象，然后转换为JSON格式，


    gdino = GroundingDINOServer() # 创建一个实例，复制给gdino这个名字，
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(gdino, name="gdino", port=args.port) # 调用host_model参数，启动http服务器并将gdino绑定到服务器上
