# -*- coding: utf-8 -*-
import torch
import more_itertools
from tqdm import tqdm


class Predictor:
    def __init__(self, clip_model, clip_processor, device, templates=None, bs=8, quiet=False, leave_progbar=False):
        self.device = device
        self.clip_model = clip_model.to(self.device)
        self.clip_model.eval()
        self.clip_processor = clip_processor
        self.bs = bs
        self.quiet = quiet
        self.leave_progbar = leave_progbar
        self.templates = templates or [
            '{}',
            'фото, на котором изображено {}',
            'изображение с {}',
            'картинка с {}',
            'фото с {}',
            'на фото видно {}',
        ]

    def get_text_latents(self, class_labels):
        """
        Encode a list of text labels into normalized CLIP text embeddings,
        with a single, concise progress bar.
        """
        all_latents = []
        total_steps = len(class_labels) * len(self.templates)

        # one bar for everything
        if not self.quiet:
            pbar = tqdm(
                total=total_steps,
                desc="Text→Latents",
                unit="it",
                leave=self.leave_progbar
            )

        # loop over templates exactly as before
        for template in self.templates:
            batch_latents = []
            for chunk in more_itertools.chunked(class_labels, self.bs):
                texts = [template.format(lbl.lower().strip()) for lbl in chunk]
                inputs = self.clip_processor(
                    text=texts,
                    return_tensors="pt",
                    padding=True
                )
                feats = self.clip_model.encode_text(
                    inputs["input_ids"].to(self.device)
                )
                batch_latents.append(feats)

                if not self.quiet:
                    pbar.update(len(chunk))

            all_latents.append(torch.cat(batch_latents, dim=0))

        if not self.quiet:
            pbar.close()

        # average over templates, then L2‑normalize
        text_latents = torch.stack(all_latents).mean(dim=0)
        return text_latents / (text_latents.norm(dim=-1, keepdim=True) + 1e-10)

    def run(self, images, text_latents):
        if not self.quiet:
            pbar = tqdm()
        labels = []
        logit_scale = self.clip_model.logit_scale.exp()
        for pil_images in more_itertools.chunked(images, self.bs):
            inputs = self.clip_processor(text='', images=list(pil_images), return_tensors='pt', padding=True)
            image_latents = self.clip_model.encode_image(inputs['pixel_values'].to(self.device))
            image_latents = image_latents / image_latents.norm(dim=-1, keepdim=True)
            logits_per_text = torch.matmul(text_latents.to(self.device), image_latents.t()) * logit_scale
            _labels = logits_per_text.argmax(0).cpu().numpy().tolist()
            if not self.quiet:
                pbar.update(len(_labels))
            labels.extend(_labels)
        pbar.close()
        return labels

    def get_image_latents(self, images):
        """
        Encode a list of PIL.Images into normalized CLIP image embeddings,
        with a single progress bar showing “processed / total”.
        """
        total = len(images)
        if not self.quiet:
            pbar = tqdm(
                total=total,
                desc="Image→Latents",
                unit="it",
                leave=self.leave_progbar
            )

        batch_latents = []
        for pil_images in more_itertools.chunked(images, self.bs):
            inputs = self.clip_processor(
                text="",
                images=list(pil_images),
                return_tensors="pt",
                padding=True
            )
            feats = self.clip_model.encode_image(
                inputs["pixel_values"].to(self.device)
            )
            batch_latents.append(feats)

            if not self.quiet:
                pbar.update(len(pil_images))

        if not self.quiet:
            pbar.close()

        # concatenate and L2‐normalize
        cat = torch.cat(batch_latents, dim=0)
        normed = cat / (cat.norm(dim=-1, keepdim=True) + 1e-10)
        return normed
