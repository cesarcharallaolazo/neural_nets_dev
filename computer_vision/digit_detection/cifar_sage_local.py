import sagemaker
import boto3
from dotenv import load_dotenv
import numpy as np
import torchvision, torch

#######

sagemaker_session = sagemaker.Session(
    boto3.session.Session(region_name="us-east-1"))
role = "arn:aws:iam::428881646170:role/service-role/AmazonSageMaker-ExecutionRole-20210618T123631"

########
import subprocess

instance_type = "local"

try:
    if subprocess.call("nvidia-smi") == 0:
        ## Set type to GPU if one is present
        instance_type = "local_gpu"
except:
    pass

print("Instance type = " + instance_type)

#######

from utils_cifar import get_train_data_loader, get_test_data_loader, imshow, classes

trainloader = get_train_data_loader()
testloader = get_test_data_loader()

#######

import numpy as np
import torchvision, torch

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
# imshow(torchvision.utils.make_grid(images))

# print labels
print(' '.join('%9s' % classes[labels[j]] for j in range(4)))

#########

from sagemaker.pytorch import PyTorch, PyTorchModel

# inputs = sagemaker_session.upload_data(path='data', bucket=bucket, key_prefix='data/cifar10')
training_input_path = "file://data/cifar-10-python.tar.gz"
validation_input_path = "file://data"
output_path = "file:///tmp/model/sagemaker/torch/cifar"

cifar10_estimator = PyTorch(
    # name="pytorch-cifar-classification",
    py_version='py3',
    entry_point='cifar10.py',
    role=role,
    framework_version='1.7.1',
    instance_count=1,
    instance_type=instance_type,
    output_path=output_path
)


cifar10_estimator.fit({"training": training_input_path})

######

cifar10_predictor = cifar10_estimator.deploy(initial_instance_count=1,
                                             instance_type=instance_type)
print(cifar10_predictor.endpoint_name)

#########

# get some test images
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%4s' % classes[labels[j]] for j in range(4)))

outputs = cifar10_predictor.predict(images.numpy())

_, predicted = torch.max(torch.from_numpy(np.array(outputs)), 1)

print('Predicted: ', ' '.join('%4s' % classes[predicted[j]]
                              for j in range(4)))

#########

# cifar10_predictor.delete_endpoint()
