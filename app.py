import gradio as gr
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os
import cv2
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# Sınıf isimleri (sadece iki sınıf)
classes = ['NORMAL', 'PNEUMONIA']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Modeli yükle
# model = models.resnet18()
# num_ftrs = model.fc.in_features
# model.fc = torch.nn.Linear(num_ftrs, len(classes))
# model_path = os.path.join('results', 'model_resnet18.pth')

model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_ftrs, len(classes))
model_path = os.path.join('results', 'model_efficientnet_b0.pth')
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model = model.to(device)

# Görüntü dönüşümleri
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

THRESHOLD = 0.7  # Zatürre için güven eşiği

# Grad-CAM fonksiyonu
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        loss = output[0, class_idx]
        loss.backward()
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        weights = gradients.mean(dim=(1, 2))  # [C]
        cam = (weights[:, None, None] * activations).sum(0)
        cam = torch.relu(cam)
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def overlay_cam_on_image(img, cam):
    img = np.array(img.resize((224, 224)).convert('RGB'))
    heatmap = (cam * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return Image.fromarray(overlay)

def predict_with_cam(img):
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_idx = probs.argmax()
        pred_prob = probs[pred_idx]
        label = classes[pred_idx]
        # Threshold uygulaması
        if label == 'PNEUMONIA' and pred_prob < THRESHOLD:
            label = 'NORMAL'
            pred_prob = 1 - probs[1]  # Normal olasılığı
            pred_idx = 0
    # Emin değilse uyarı ver
    if pred_prob < THRESHOLD:
        uyarı = f"⚠️ Model bu tahminde emin değil!\nTahmin edilen sınıf: {label}\nGüven: %{pred_prob*100:.1f}\nLütfen bir uzmana danışın."
        if label == 'PNEUMONIA':
            grad_cam = GradCAM(model, model.features[-1][0])
            cam = grad_cam(img_tensor, class_idx=1)
            grad_cam.remove_hooks()
            overlay = overlay_cam_on_image(img, cam)
            return uyarı, overlay
        else:
            return uyarı, img
    if label == 'NORMAL':
        metin = f"🟢 Tahmin: {label}\n\nBu akciğer röntgeninde zatürre bulgusu YOK.\nGüven: %{pred_prob*100:.1f}"
        return metin, img
    else:
        # Grad-CAM ile işaretli görsel oluştur
        grad_cam = GradCAM(model, model.features[-1][0])
        cam = grad_cam(img_tensor, class_idx=1)  # PNEUMONIA için
        grad_cam.remove_hooks()
        overlay = overlay_cam_on_image(img, cam)
        metin = f"🔴 Tahmin: {label}\n\nBu akciğer röntgeninde ZATÜRRE tespit edildi!\nGüven: %{pred_prob*100:.1f}"
        return metin, overlay

output_text = gr.Textbox(
    label="Sonuç",
    lines=3,
    interactive=False,
    elem_id="output-box",
)
output_image = gr.Image(
    label="Modelin dikkat ettiği bölge (Grad-CAM)",
    type="pil",
)

gr.Interface(
    fn=predict_with_cam,
    inputs=gr.Image(type="pil"),
    outputs=[output_text, output_image],
    title="Akciğer Röntgeni Sınıflandırıcı (ResNet18 + Grad-CAM)",
    description="Bir akciğer röntgeni yükleyin, model NORMAL mı PNEUMONIA mı olduğunu tahmin etsin ve zatürre tespitinde dikkat ettiği bölgeyi işaretlesin.\n\n<small>Bu uygulama tıbbi teşhis amacıyla kullanılmamalıdır. Sonuçlar yalnızca eğitim ve demo amaçlıdır.</small>",
    allow_flagging="never"
).launch() 