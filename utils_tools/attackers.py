import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescent, FastGradientMethod, CarliniL2Method, DeepFool, SquareAttack, SpatialTransformationAttack

# Load a pre-trained model
model = models.resnet18(pretrained=True)
model.eval()

# Define a loss function and optimizer (required for PyTorchClassifier)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Define transformations for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load an example image
image_path = 'path/to/your/image.jpg'
image = Image.open(image_path)
image = transform(image).unsqueeze(0).numpy()

# Create a PyTorch classifier
classifier = PyTorchClassifier(
    model=model,
    loss=loss_fn,
    optimizer=optimizer,
    input_shape=(3, 224, 224),
    nb_classes=1000,
)

# Example target label
target_label = np.array([7])  # Specify a target label for targeted attacks

# FGSM attack
fgsm = FastGradientMethod(estimator=classifier, eps=0.1)
image_adv_fgsm = fgsm.generate(x=image)

# PGD attack
pgd = ProjectedGradientDescent(estimator=classifier, eps=0.1, eps_step=0.01, max_iter=40)
image_adv_pgd = pgd.generate(x=image)

# C&W attack
cw = CarliniL2Method(classifier=classifier, confidence=0.5)
image_adv_cw = cw.generate(x=image)

# DeepFool attack
deepfool = DeepFool(classifier=classifier)
image_adv_deepfool = deepfool.generate(x=image)

# SquareAttack
square_attack = SquareAttack(estimator=classifier)
image_adv_square = square_attack.generate(x=image)

# PatchAttack
patch_attack = SpatialTransformationAttack(estimator=classifier)
image_adv_patch = patch_attack.generate(x=image)

# TransferableAttack (example using PGD as base)
transferable_attack = ProjectedGradientDescent(estimator=classifier, eps=0.1, eps_step=0.01, max_iter=40)
image_adv_transferable = transferable_attack.generate(x=image)

# Function to convert numpy array to PIL image and display
def show_image(image_np, title):
    image_pil = Image.fromarray((image_np * 255).astype(np.uint8).squeeze().transpose(1, 2, 0))
    image_pil.show(title=title)

# Show original and adversarial images
show_image(image, 'Original Image')
show_image(image_adv_fgsm, 'FGSM Adversarial Image')
show_image(image_adv_pgd, 'PGD Adversarial Image')
show_image(image_adv_cw, 'C&W Adversarial Image')
show_image(image_adv_deepfool, 'DeepFool Adversarial Image')
show_image(image_adv_square, 'Square Adversarial Image')
show_image(image_adv_patch, 'Patch Adversarial Image')
show_image(image_adv_transferable, 'Transferable Adversarial Image')
