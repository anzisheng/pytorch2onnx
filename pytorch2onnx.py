import torch
import torchvision


def convert_torch2onnx():
    # 设置pretrained=True，将自动下载resnet18模型权重
    # 自动下载的权重文件来自于pytorch，在imagenet数据集上完成训练并release出来
    resnet18 = torchvision.models.resnet18(pretrained=True)
    # 指定resnet18.onnx模型的input shape，onnx模型的input shape需固定
    dummy_input = torch.randn(1, 3, 224, 224, device="cpu")
    torch.onnx.export(resnet18, dummy_input, "resnet18.onnx")


if __name__ == "__main__":
    convert_torch2onnx()
