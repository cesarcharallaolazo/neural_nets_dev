# from utils_cifar import get_train_data_loader, get_test_data_loader, imshow, classes
# import numpy as np
# from sagemaker.predictor import Predictor
# import torch
# import sagemaker, boto3
#
# testloader = get_test_data_loader()
# dataiter = iter(testloader)
# images, labels = dataiter.next()
#
# feat = images.numpy()
#
# # import io, json
# # memfile = io.BytesIO()
# # np.save(memfile, feat)
# # memfile.seek(0)
# # serialized = json.dumps(memfile.read().decode('latin-1'))
#
# # predictor = Predictor(endpoint_name="762e3xn3ir-algo-1-ckcqt",
# #                       instance_type="local")
# # print(type(images.numpy()))
# # outputs = predictor.predict(feat.tobytes())
# #
# # _, predicted = torch.max(torch.from_numpy(np.array(outputs)), 1)
# #
# # print('Predicted: ', ' '.join('%4s' % classes[predicted[j]]
# #                               for j in range(4)))
#
# import requests
# import pickle
# # with open("./data/cifar10/cat/image_9.png", 'rb') as f:
# #  payload = f.read()
# #  payload = payload
# #
# url = "http://localhost:8080/invocations"
# #
# # if isinstance(Body, str):
# #     Body = Body.encode("utf-8")
#
# # cifar10_estimator.delete_endpoint()
# # print(Body)
#
# r = requests.post(url, data=feat, headers={})
# print(r.text.split("\n"))
#
# # response = predictor.predict(data=payload)
# # print(response)


###########

from utils_cifar import get_train_data_loader, get_test_data_loader, imshow, classes


# testloader = get_test_data_loader()
# dataiter = iter(testloader)
# images, labels = dataiter.next()
#
# feat = images.numpy()
# print(feat)

import torch
import torchvision
import torchvision.models
import torchvision.transforms as transforms
transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root="./data", train=True,
                                            download=False, transform=transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                               shuffle=True)

dataiter = iter(train_loader)
images, labels = dataiter.next()
# print(images)
images = images.numpy()
# print(images)

import numpy
params = {'param0': 'param0', 'param1': 'param1'}
# arr = np.random.rand(10, 10)
data = {'images': images.tolist()}

payload = data

import json
import  sagemaker
response = sagemaker.local.LocalSagemakerRuntimeClient().invoke_endpoint(
    EndpointName="pytorch-cifar-classification",
    ContentType="application/json",
    Accept="application/json",
    Body=json.dumps(payload)
)

import urllib
# thepage = urllib.request.urlopen(theurl).read()

# data = json.loads(response["Body"].raw)
print(response['Body'].read())
# print("Response=",response)
# response_body = json.loads(response['Body'].read())
# print(json.dumps(response_body, indent=4))



# import boto3
# import json
# import sagemaker
# # from deploy_env import DeployEnv
#
# # env = DeployEnv()
#
# print("Attempting to invoke model_name=%s / env=%s..." % (env.setting('model_name'), env.current_env()))
#
# payload = [["The Wimbledon tennis tournament starts next week!"],["The Canadian President signed in the new federal law."]]
#
# # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker-runtime.html#SageMakerRuntime.Client.invoke_endpoint
# response = env.runtime_client().invoke_endpoint(
#     EndpointName="pytorch-text-classification",
#     ContentType="application/json",
#     Accept="application/json",
#     Body=json.dumps(payload)
# )
#
# print("Response=",response)
# response_body = json.loads(response['Body'].read())
# print(json.dumps(response_body, indent=4))