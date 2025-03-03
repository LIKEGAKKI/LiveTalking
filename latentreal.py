import glob
import os
import numpy as np

from latentsync.utils import image_processor

from basereal import BaseReal


def load_avatar(opt):
    avatar_path = f"./data/avatars/{opt.avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs"

    input_img_list = glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    frame_list_cycle = read_imgs(input_img_list)

    faces = []
    boxes = []
    affine_matrices = []
    print(f"Affine transforming {len(frame_list_cycle)} faces...")
    for frame in tqdm.tqdm(frame_list_cycle):
        face, box, affine_matrix = image_processor.affine_transform(frame)
        faces.append(face)
        boxes.append(box)
        affine_matrices.append(affine_matrix)

    faces = torch.stack(faces)
    return faces, frame_list_cycle, boxes, affine_matrices


def load_model(opt):
    is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if is_fp16_supported else torch.float32

    print(f"Loaded checkpoint path: {opt.inference_ckpt_path}")

    config = OmegaConf.load(opt.unet_config_path)
    scheduler = DDIMScheduler.from_pretrained("configs")

    if config.model.cross_attention_dim == 768:
        whisper_model_path = "checkpoints/whisper/small.pt"
    elif config.model.cross_attention_dim == 384:
        whisper_model_path = "checkpoints/whisper/tiny.pt"
    else:
        raise NotImplementedError("cross_attention_dim must be 768 or 384")

    audio_encoder = Audio2Feature(model_path=whisper_model_path, device="cuda", num_frames=config.data.num_frames)

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        opt.inference_ckpt_path,  # load checkpoint
        device="cpu",
    )

    unet = unet.to(dtype=dtype)

    # set xformers
    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()

    return vae, unet, audio_encoder, scheduler


def read_imgs(img_list):
    frames = []
    logger.info('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames


class LatentReal(BaseReal):
    def __init__(self, opt, model, avatar):
        self.opt = opt
        self.model = model
        self.avatar = avatar
        self.frame_list_cycle = avatar.frame_list_cycle

        self.vae, self.unet, self.audio_encoder, self.scheduler = model

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.height = self.unet.config.sample_size * self.vae_scale_factor
        self.width = self.unet.config.sample_size * self.vae_scale_factor

    def render(self, quit_event, loop=None, audio_track=None, video_track=None):
