import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
# from mobile_sam import sam_model_registry, SamPredictor
from segment_anything import sam_model_registry, SamPredictor
import threading
import queue
import random
import torch.nn as nn

# ====== CONFIG ======
IMAGE_ROOT = Path("/home/sarah/Documents/background_segmentation/testset_cartclipper/images_test")
FIXED_DIR  = Path("/home/sarah/Documents/background_segmentation/testset_cartclipper/masks_test")
FIXED_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "/home/sarah/Documents/background_segmentation/cart_segmentation_cartClipper/src/models/no_hole_model2.pth"  # UNet / Dice model
SAM_PATH   = Path("/home/sarah/Documents/background_segmentation/cart_segmentation_cartClipper/src/models/sam_vit_b_01ec64.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIDE_PADDING_RATIO = 0.1
IMAGE_SIZE = 1024
DISPLAY_SCALE = 1.0

# ====== SPEED TWEAKS (CPU) ======
torch.set_grad_enabled(False)
try:
    import os as _os
    _threads = max(1, (os.cpu_count() or 4) // 2)
    torch.set_num_threads(_threads)
    _os.environ["OMP_NUM_THREADS"] = str(_threads)
    _os.environ["MKL_NUM_THREADS"] = str(_threads)
except Exception:
    pass


# --- UNet Model (PyTorch) ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2,2)
        self.conv1 = DoubleConv(in_channels, 64)
        self.conv2 = DoubleConv(64, 128)
        self.conv3 = DoubleConv(128, 256)
        self.conv4 = DoubleConv(256, 512)
        self.conv5 = DoubleConv(512, 1024)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dconv4 = DoubleConv(1024, 512)
        self.dconv3 = DoubleConv(512, 256)
        self.dconv2 = DoubleConv(256, 128)
        self.dconv1 = DoubleConv(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.maxpool(x1))
        x3 = self.conv3(self.maxpool(x2))
        x4 = self.conv4(self.maxpool(x3))
        x5 = self.conv5(self.maxpool(x4))
        x  = self.upconv4(x5); x = torch.cat([x, x4], dim=1); x = self.dconv4(x)
        x  = self.upconv3(x);  x = torch.cat([x, x3], dim=1); x = self.dconv3(x)
        x  = self.upconv2(x);  x = torch.cat([x, x2], dim=1); x = self.dconv2(x)
        x  = self.upconv1(x);  x = torch.cat([x, x1], dim=1); x = self.dconv1(x)
        x  = self.final_conv(x)
        return torch.sigmoid(x)  # <- lässt Output schon in [0,1]

# Lade dein trainiertes Modell
SEG_MODEL = UNet().to(DEVICE)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
state = ckpt.get('model_state_dict', ckpt)  # unterstützt beide Varianten
SEG_MODEL.load_state_dict(state, strict=True)
SEG_MODEL.eval()

   
# ====== SAM (MobileSAM vit_t) ======
sam = sam_model_registry["vit_b"](checkpoint=SAM_PATH)
sam.to(device=DEVICE)
predictor = SamPredictor(sam)

# ====== Helpers ======
def letterbox_image_with_side_padding(image, padding_color=(0, 0, 0), side_padding_ratio=SIDE_PADDING_RATIO):
    image_np = np.array(image)
    orig_height, orig_width = image_np.shape[:2]

    side_padding = round(orig_width * side_padding_ratio)
    padded_width = orig_width + 2 * side_padding
    padded_height = orig_height

    padded_image = np.full((padded_height, padded_width, 3), padding_color, dtype=np.uint8)
    padded_image[:, side_padding:side_padding+orig_width] = image_np

    max_dim = max(padded_width, padded_height)
    letterboxed_image = np.full((max_dim, max_dim, 3), padding_color, dtype=np.uint8)
    x_offset = (max_dim - padded_width) // 2
    y_offset = (max_dim - padded_height) // 2
    letterboxed_image[y_offset:y_offset+padded_height, x_offset:x_offset+padded_width] = padded_image
    return letterboxed_image

def preprocess_image(image_pil):
    padded = letterbox_image_with_side_padding(image_pil, padding_color=(0, 0, 0), side_padding_ratio=SIDE_PADDING_RATIO)
    return cv2.resize(padded, (IMAGE_SIZE, IMAGE_SIZE))

def smooth_mask(mask, k=5, iters=1, blur_sigma=None):
    k = max(3, int(k) | 1)  # odd >=3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iters)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  kernel, iterations=iters)
    if blur_sigma is None:
        blur_sigma = k / 2.0
    m = cv2.GaussianBlur(m, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    return m.astype(np.uint8)



def predict_mask(image_pil: Image.Image, model, device: torch.device) -> np.ndarray:
    img_proc = preprocess_image(image_pil)  # HxWx3 uint8
    x = torch.from_numpy(img_proc).float().permute(2,0,1) / 255.0
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        y = model(x)  # dein UNet gibt bereits Sigmoid-Prob in [0,1] zurück: [1,1,H,W] oder [1,2,H,W]
        if y.shape[1] == 1:
            mask = (y[0,0].cpu().numpy() > 0.5).astype(np.uint8) * 255
        else:
            prob1 = torch.softmax(y, dim=1)[0,1].cpu().numpy()
            mask = (prob1 > 0.5).astype(np.uint8) * 255
    return mask


# ====== Build image list (no pre-existing masks needed) ======
paired_images = []
skipped_existing = 0
for img_path in IMAGE_ROOT.rglob("*.jpeg"):
    rel = img_path.relative_to(IMAGE_ROOT)
    out_path = FIXED_DIR / rel.with_suffix(".png")
    if out_path.exists():
        skipped_existing += 1
        continue
    paired_images.append(img_path)

print(f"Found {len(paired_images)} images to refine. Skipped {skipped_existing} already in {FIXED_DIR}.")
random.shuffle(paired_images)

# ====== Prefetch: compute initial UNet mask + SAM-sized image ======
prefetch_queue = queue.Queue(maxsize=2)  # current + next
stop_event = threading.Event()

def prefetch_worker(image_list, q, stop_evt):
    # Lokaler Predictor im Prefetch-Thread (nicht der globale)
    local_predictor = SamPredictor(sam)
    for img_path in image_list:
        if stop_evt.is_set():
            break
        try:
            with Image.open(img_path) as im:
                image_pil = im.convert("RGB")
            orig_w, orig_h = image_pil.size

            # SAM-Input vorbereiten
            image_np = preprocess_image(image_pil)

            # Embedding vorberechnen
            local_predictor.set_image(image_np)
            embedding = local_predictor.get_image_embedding()

            # UNet-Initialmaske (hier leer)
            work_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

            # In Queue legen
            q.put((img_path, image_np, work_mask, (orig_h, orig_w), embedding), block=True)

        except Exception as e:
            print(f"⚠️ Prefetch error for {img_path}: {e}")
            continue
    q.put(None)


prefetch_thread = threading.Thread(target=prefetch_worker, args=(paired_images, prefetch_queue, stop_event), daemon=True)
prefetch_thread.start()

# ====== Interactive loop ======
processed = 0
while True:
    item = prefetch_queue.get()
    if item is None:
        break

    image_path, image_np, work_mask, orig_hw, embedding = item
    orig_h, orig_w = orig_hw
    rel_path = image_path.relative_to(IMAGE_ROOT)

    # Precomputed Embedding direkt setzen
    predictor.is_image_set = True
    predictor.features = embedding
    predictor.input_size = (IMAGE_SIZE, IMAGE_SIZE)
    predictor.original_size = (IMAGE_SIZE, IMAGE_SIZE)

    # UI state
    state = {
        'mask': work_mask.copy().astype(np.uint8),
        'start_mask': work_mask.copy().astype(np.uint8),
        'drawing': False,
        'draw_label': 1,  # 1=FG, 0=BG
        'clicks': [],
        'rb_down': False,
        'box_start': None,
        'box_end': None,
        'apply_smoothing': True,
        'smooth_k': 5,
        'smooth_iters': 1,
        'brush_radius': 60,
        'cursor': None,
    }

    def fill_holes(mask: np.ndarray) -> np.ndarray:
        """Füllt alle Löcher in der Maske vollständig."""
        # Kopie in 8-Bit sicherstellen
        mask_filled = mask.copy().astype(np.uint8)
        contours, _ = cv2.findContours(mask_filled, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for i, cnt in enumerate(contours):
            # Hier füllen wir alle "inneren" Konturen (Holes)
            cv2.drawContours(mask_filled, [cnt], 0, 255, -1)
        return mask_filled
    

    def apply_post(mask):
        m = smooth_mask(mask, k=state['smooth_k'], iters=state['smooth_iters']) if state['apply_smoothing'] else mask
        # m = fill_holes(m)
        return m

    def click_event(event, x, y, flags, param):
        x = int(x / DISPLAY_SCALE); y = int(y / DISPLAY_SCALE)
        state['cursor'] = (x, y)

        ctrl = bool(flags & cv2.EVENT_FLAG_CTRLKEY)
        shift = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)

        if event == cv2.EVENT_LBUTTONDOWN:
            if ctrl:
                # Start Box-Modus (statt rechte Maustaste)
                state['box_active'] = True
                state['box_start'] = (x, y)
                state['box_end'] = (x, y)
            else:
                # Pinsel: Shift = BG, sonst FG
                state['draw_label'] = 0 if shift else 1
                state['drawing'] = True
                state['clicks'] = [[x, y, state['draw_label']]]

        elif event == cv2.EVENT_MOUSEMOVE:
            if state['box_active']:
                state['box_end'] = (x, y)
            elif state['drawing']:
                state['clicks'].append([x, y, state['draw_label']])

        elif event == cv2.EVENT_LBUTTONUP:
            if state['box_active']:
                state['box_active'] = False
                if state['box_start'] is not None and state['box_end'] is not None:
                    x1, y1 = state['box_start']; x2, y2 = state['box_end']
                    xmin, xmax = sorted([x1, x2]); ymin, ymax = sorted([y1, y2])
                    box = np.array([xmin, ymin, xmax, ymax])
                    masks, _, _ = predictor.predict(box=box, multimask_output=False)
                    new_mask = (masks[0] * 255).astype(np.uint8)
                    state['mask'] = cv2.bitwise_or(state['mask'], new_mask)
                    state['mask'] = apply_post(state['mask'])
                state['box_start'] = None
                state['box_end'] = None

            elif state['drawing']:
                state['drawing'] = False
                if state['clicks']:
                    pts = np.array([c[:2] for c in state['clicks']])
                    labels = np.array([c[2] for c in state['clicks']])
                    masks, _, _ = predictor.predict(
                        point_coords=pts,
                        point_labels=labels,
                        multimask_output=False
                    )
                    new_mask = (masks[0] * 255).astype(np.uint8)

                    brush = np.zeros(new_mask.shape, dtype=np.uint8)
                    for pt in pts:
                        cv2.circle(brush, tuple(pt), radius=state['brush_radius'], color=255, thickness=-1)
                    new_local = cv2.bitwise_and(new_mask, brush)

                    if state['draw_label'] == 1:
                        state['mask'] = cv2.bitwise_or(state['mask'], new_local)
                    else:
                        inv_brush = cv2.bitwise_not(brush)
                        state['mask'] = cv2.bitwise_and(state['mask'], inv_brush)

                    state['mask'] = apply_post(state['mask'])
                    state['clicks'] = []


    # Window/UI
    win = "Mask Fix (LMB=brush, RMB=box, Ctrl/Shift=BG, +/- brush, s=toggle smooth, [ ]=kern, r=reset, q/ESC=save)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, click_event)

    while True:
        display = image_np.copy()

        # Live box
        if state['box_start'] and state['box_end']:
            x1, y1 = state['box_start']; x2, y2 = state['box_end']
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 1)

        # Mask contours
        contours, _ = cv2.findContours(state['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(display, contours, -1, (0, 255, 0), 2)

        # Active clicks
        for pt in state['clicks']:
            color = (0, 255, 0) if pt[2] == 1 else (0, 0, 255)
            cv2.circle(display, (pt[0], pt[1]), 2, color, -1)

        # HUD
        hud = f"smooth:{state['apply_smoothing']} k:{state['smooth_k']} it:{state['smooth_iters']} br:{state['brush_radius']}"
        cv2.putText(display, hud, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

        # Always show brush outline
        if state['cursor'] is not None:
            color = (255, 0, 0) if state['rb_down'] else ((0, 255, 0) if state['draw_label'] == 1 else (0, 0, 255))
            cv2.circle(display, state['cursor'], state['brush_radius'], color, 1, lineType=cv2.LINE_AA)


        # --- nach HUD, VOR dem Resize/Imshow ---
        # Halbtransparentes grünes Overlay für die aktuelle Maske:
        mask3 = np.zeros_like(display)
        mask3[state['mask'] > 0] = (0, 255, 0)
        display = cv2.addWeighted(display, 1.0, mask3, 0.25, 0.0)

        # Brush-Outline am Cursor
        if state['cursor'] is not None:
            color = (255, 0, 0) if state['rb_down'] else ((0, 255, 0) if state['draw_label'] == 1 else (0, 0, 255))
            cv2.circle(display, state['cursor'], state['brush_radius'], color, 1, lineType=cv2.LINE_AA)

        display_resized = cv2.resize(display, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_NEAREST)
        cv2.imshow(win, display_resized)

        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord('q')):  # save & next
            break
        elif key == ord('s'):
            state['apply_smoothing'] = not state['apply_smoothing']
            if state['apply_smoothing']:
                state['mask'] = apply_post(state['mask'])
        elif key == ord('['):
            state['smooth_k'] = max(3, state['smooth_k'] - 2)
            if state['apply_smoothing']:
                state['mask'] = apply_post(state['mask'])
        elif key == ord(']'):
            state['smooth_k'] += 2
            if state['apply_smoothing']:
                state['mask'] = apply_post(state['mask'])
        elif key == ord('r'):
            state['mask'] = state['start_mask'].copy()
            if state['apply_smoothing']:
                state['mask'] = apply_post(state['mask'])
        elif key in (ord('+'), ord('=')):
            state['brush_radius'] = min(256, state['brush_radius'] + 2)
        elif key in (ord('-'), ord('_')):
            state['brush_radius'] = max(1, state['brush_radius'] - 2)
        elif key == ord('f'):
            state['mask'] = fill_holes(state['mask'])

    cv2.destroyAllWindows()

    # Save at original resolution
    out_mask = cv2.resize(state['mask'], (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    out_path = FIXED_DIR / rel_path.with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out_mask)
    print(f"✅ Saved refined mask: {out_path}")
    processed += 1

# Cleanup
stop_event.set()
prefetch_thread.join()
print(f"🎉 Done. Processed {processed} items.")
