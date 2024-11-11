import torch
import torch.nn as nn
import numpy as np
import base64
import cv2
from io import BytesIO
from PIL import Image

class ColorClockVAEHandler:
    def __init__(self, model_path, size=140, latent_dim=20, device=None):
        self.size = size
        self.input_dim = size ** 2 * 3  # RGB 3채널
        self.condition_dim = 2
        self.latent_dim = latent_dim
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = self.load_model(model_path)

    class ConditionalVAE(nn.Module):
        def __init__(self, input_dim, condition_dim, latent_dim):
            super(ColorClockVAEHandler.ConditionalVAE, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim + condition_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
            )

            self.fc_mu = nn.Linear(128, latent_dim)
            self.fc_logvar = nn.Linear(128, latent_dim)
            
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim + condition_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, input_dim),
                nn.Sigmoid()
            )

        def encode(self, x, condition):
            x = x.view(x.size(0), -1)
            condition = condition.view(condition.size(0), -1)
            x_cond = torch.cat([x, condition], dim=1)
            h = self.encoder(x_cond)
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            return mu, logvar

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z, condition):
            z_cond = torch.cat([z, condition], dim=1)
            return self.decoder(z_cond)

        def forward(self, x, condition):
            mu, logvar = self.encode(x, condition)
            z = self.reparameterize(mu, logvar)
            return self.decode(z, condition), mu, logvar


    def load_model(self, model_path):
        model = self.ConditionalVAE(self.input_dim, self.condition_dim, self.latent_dim).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("Model loaded successfully.")
        return model

    def generate_image(self, hour, minute):
        with torch.no_grad():
            hour_norm = torch.tensor([hour / 12.0], dtype=torch.float32).to(self.device)
            minute_norm = torch.tensor([minute / 60.0], dtype=torch.float32).to(self.device)
            condition = torch.stack([hour_norm, minute_norm], dim=1)

            z = torch.randn(1, self.latent_dim).to(self.device)
            generated_img = self.model.decode(z, condition)
            generated_img = generated_img.view(3, self.size, self.size).permute(1, 2, 0).cpu().numpy()

            transformed_img = self.apply_perspective_transform(generated_img)
            return (transformed_img * 255).astype(np.uint8)

    @staticmethod
    def apply_perspective_transform(image):
        h, w, c = image.shape
        src_points = np.float32([
            [0, 0],
            [w - 1, 0],
            [0, h - 1],
            [w - 1, h - 1]
        ])
        dst_points = np.float32([
            [np.random.uniform(0, w * 0.2), np.random.uniform(0, h * 0.2)],
            [np.random.uniform(w * 0.8, w), np.random.uniform(0, h * 0.2)],
            [np.random.uniform(0, w * 0.2), np.random.uniform(h * 0.8, h)],
            [np.random.uniform(w * 0.8, w), np.random.uniform(h * 0.8, h)]
        ])
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        transformed_image = cv2.warpPerspective(image, matrix, (w, h))
        return transformed_image

    @staticmethod
    def image_to_base64(image):
        pil_image = Image.fromarray(image)
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return base64_str