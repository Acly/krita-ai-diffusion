from __future__ import annotations

import json
import zlib
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Literal, NamedTuple, TypeVar, overload
from uuid import uuid4

from .image import Bounds, Extent, Image, ImageCollection
from .resources import Arch, ControlMode
from .util import base_type_match
from .util import client_logger as log


class ComfyRunMode(Enum):
    runtime = 0  # runs as part of same process, transfer images in memory
    server = 1  # runs as a server, transfer images via base64 or websocket


class Output(NamedTuple):
    node: int
    output: int


T = TypeVar("T")
Output2 = tuple[Output, Output]
Output3 = tuple[Output, Output, Output]
Output4 = tuple[Output, Output, Output, Output]
Input = int | float | bool | str | Output


class ConditioningOutput(NamedTuple):
    positive: Output
    negative: Output


class ComfyNode(NamedTuple):
    id: int
    type: str
    inputs: dict[str, Input]

    @overload
    def input(self, key: str, default: T) -> T: ...

    @overload
    def input(self, key: str, default: None = None) -> Input | None: ...

    def input(self, key: str, default: T | None = None) -> T | Input | None:
        result = self.inputs.get(key, default)
        assert default is None or base_type_match(result, default)
        return result

    def output(self, index=0) -> Output:
        return Output(int(self.id), index)


class ComfyWorkflow:
    """Builder for workflows which can be sent to the ComfyUI prompt API."""

    def __init__(self, node_defs: ComfyObjectInfo | None = None, run_mode=ComfyRunMode.server):
        self.root: dict[str, dict] = {}
        self.images: dict[str, Image] = {}
        self.image_data: dict[str, bytes] = {}
        self.node_count = 0
        self.sample_count = 0
        self.node_defs = node_defs or ComfyObjectInfo({})
        self._cache: dict[str, Output | Output2 | Output3 | Output4] = {}
        self._run_mode: ComfyRunMode = run_mode

    @staticmethod
    def import_graph(existing: dict, node_defs: ComfyObjectInfo):
        w = ComfyWorkflow(node_defs)
        existing = _convert_ui_workflow(existing, node_defs)
        node_map: dict[str, str] = {}
        queue = list(existing.keys())
        times_queued: dict[str, int] = {}
        # Re-order nodes such that a nodes inputs are always processed before the node itself.
        while queue:
            id = queue.pop(0)
            node = deepcopy(existing[id])
            class_type = node.get("class_type")
            if class_type is None:
                log.warning(f"Workflow import: Node {id} is not installed, aborting.")
                return w
            if node_defs and class_type not in node_defs:
                log.warning(
                    f"Workflow contains a node of type {class_type} which is not installed on the ComfyUI server."
                )
            edges = [e for e in node["inputs"].values() if isinstance(e, list)]
            if any(e[0] not in node_map for e in edges):
                count = times_queued.setdefault(id, 0)
                if count > len(existing):
                    log.warning(
                        "Failed to import custom workflow, the graph appears to contain a loop."
                    )
                    return w
                # re-queue node if an input is not yet mapped
                queue.append(id)
                times_queued[id] += 1
                continue

            for e in edges:
                e[0] = node_map[e[0]]
            node_map[id] = str(w.node_count)
            w.root[str(w.node_count)] = node
            w.node_count += 1
        return w

    @staticmethod
    def from_dict(existing: dict):
        w = ComfyWorkflow()
        w.root = existing
        w.node_count = len(w.root)
        return w

    def add_default_values(self, node_name: str, args: dict):
        if node_inputs := self.node_defs.params(node_name, "required"):
            for k, v in node_inputs.items():
                if v is not None:
                    args.setdefault(k, v)
                elif k not in args:
                    log.warning(f"Node {node_name} missing required input {k} (no default)")
        return args

    def embed_images(self):
        r: dict[str, dict] = {}
        masks = []
        for id, node in self.root.items():
            if node["class_type"] == "ETN_LoadImageCache":
                image_id = node["inputs"]["id"]
                image = Image.from_bytes(self.image_data[image_id])
                is_mask = image.is_mask
                node = deepcopy(node)
                if is_mask:
                    node["class_type"] = "ETN_LoadMaskBase64"
                    node["inputs"]["mask"] = image.to_base64()
                    masks.append(id)
                else:
                    node["class_type"] = "ETN_LoadImageBase64"
                    node["inputs"]["image"] = image.to_base64()
            elif node["class_type"] == "ETN_SaveImageCache":
                node = deepcopy(node)
                node["class_type"] = "PreviewImage"
                del node["inputs"]["format"]
            elif node["class_type"] == "ETN_InjectImage":
                image = self.images[node["inputs"]["id"]]
                node = deepcopy(node)
                node["class_type"] = "ETN_LoadImageBase64"
                node["inputs"]["image"] = image.to_base64()
            elif node["class_type"] == "ETN_InjectMask":
                image = self.images[node["inputs"]["id"]]
                node = deepcopy(node)
                node["class_type"] = "ETN_LoadMaskBase64"
                node["inputs"]["mask"] = image.to_base64()
            elif node["class_type"] == "ETN_ReturnImage":
                node = deepcopy(node)
                node["class_type"] = "PreviewImage"
            else:
                # Masks are output 1 for LoadImageCache, but output 0 for LoadMaskBase64
                inputs = node.get("inputs", {})
                for input_name, input_value in inputs.items():
                    if isinstance(input_value, list) and input_value[0] in masks:
                        node = deepcopy(node)
                        node["inputs"][input_name] = [input_value[0], 0]  # output 0 of same node
            r[id] = node

        result = ComfyWorkflow(self.node_defs, self._run_mode)
        result.root = r
        result.node_count = self.node_count
        result.sample_count = self.sample_count
        return result

    def dump(self, filepath: str | Path):
        filepath = Path(filepath)
        if filepath.suffix != ".json":
            filepath = filepath / "workflow.json"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.root, f, indent=4)
        for id, image in self.images.items():
            image.save(filepath.parent / f"{id}.png")

    @overload
    def add(self, class_type: str, output_count: Literal[1], **inputs) -> Output: ...

    @overload
    def add(self, class_type: str, output_count: Literal[2], **inputs) -> Output2: ...

    @overload
    def add(self, class_type: str, output_count: Literal[3], **inputs) -> Output3: ...

    @overload
    def add(self, class_type: str, output_count: Literal[4], **inputs) -> Output4: ...

    def add(self, class_type: str, output_count: int, **inputs):
        def normalize(x):
            return [str(x.node), x.output] if isinstance(x, Output) else x

        inputs = self.add_default_values(class_type, inputs)
        self.node_count += 1
        self.root[str(self.node_count)] = {
            "class_type": class_type,
            "inputs": {k: normalize(v) for k, v in inputs.items()},
        }
        output = tuple(Output(self.node_count, i) for i in range(output_count))
        return output[0] if output_count == 1 else output

    @overload
    def add_cached(self, class_type: str, output_count: Literal[1], **inputs) -> Output: ...

    @overload
    def add_cached(self, class_type: str, output_count: Literal[3], **inputs) -> Output3: ...

    def add_cached(self, class_type: str, output_count: Literal[1, 3], **inputs):
        key = class_type + str(inputs)
        result = self._cache.get(key, None)
        if result is None:
            result = self.add(class_type, output_count, **inputs)
            self._cache[key] = result
        return result

    def remove(self, node_id: int):
        del self.root[str(node_id)]

    def node(self, node_id: int):
        inputs = self.root[str(node_id)]["inputs"]
        inputs = {
            k: Output(int(v[0]), v[1]) if isinstance(v, list) else v for k, v in inputs.items()
        }
        return ComfyNode(node_id, self.root[str(node_id)]["class_type"], inputs)

    def copy(self, node: ComfyNode):
        return self.add(node.type, 1, **node.inputs)

    def find(self, type: str):
        return (self.node(int(k)) for k, v in self.root.items() if v["class_type"] == type)

    def find_connected(self, output: Output):
        for node in self:
            for input_name, input_value in node.inputs.items():
                if input_value == output:
                    yield node, input_name

    def guess_sample_count(self):
        self.sample_count = sum(
            int(value)
            for node in self
            for name, value in node.inputs.items()
            if name == "steps" and isinstance(value, (int, float))
        )
        return self.sample_count

    def __iter__(self):
        return iter(self.node(int(k)) for k in self.root)

    def __contains__(self, node: ComfyNode):
        return any(n == node for n in self)

    def _add_image(self, image: Image):
        id = str(uuid4())
        if exists := next((k for k, v in self.images.items() if v == image), None):
            return exists
        self.images[id] = image
        return id

    def _add_image_hashed(self, image: Image):
        data = image.to_bytes()
        hash = zlib.crc32(data)
        id = f"{hash:08x}"
        self.image_data[id] = data
        return id

    # Nodes

    def ksampler(
        self,
        model: Output,
        cond: ConditioningOutput,
        latent_image: Output,
        sampler="dpmpp_2m_sde_gpu",
        scheduler="normal",
        steps=20,
        cfg=7.0,
        denoise=1.0,
        seed=1234,
    ):
        self.sample_count += steps
        return self.add(
            "KSampler",
            1,
            seed=seed,
            sampler_name=sampler,
            scheduler=scheduler,
            model=model,
            positive=cond.positive,
            negative=cond.negative,
            latent_image=latent_image,
            steps=steps,
            cfg=cfg,
            denoise=denoise,
        )

    def ksampler_advanced(
        self,
        model: Output,
        cond: ConditioningOutput,
        latent_image: Output,
        sampler="dpmpp_2m_sde_gpu",
        scheduler="normal",
        steps=20,
        start_at_step=0,
        cfg=7.0,
        seed=-1,
    ):
        self.sample_count += steps - start_at_step

        return self.add(
            "KSamplerAdvanced",
            1,
            noise_seed=seed,
            sampler_name=sampler,
            scheduler=scheduler,
            model=model,
            positive=cond.positive,
            negative=cond.negative,
            latent_image=latent_image,
            steps=steps,
            start_at_step=start_at_step,
            end_at_step=steps,
            cfg=cfg,
            add_noise="enable",
            return_with_leftover_noise="disable",
        )

    def sampler_custom_advanced(
        self,
        model: Output,
        cond: ConditioningOutput,
        latent_image: Output,
        arch: Arch,
        sampler="euler",
        scheduler="normal",
        steps=20,
        start_at_step=0,
        cfg=7.0,
        seed=-1,
        extent: Extent | None = None,
    ):
        self.sample_count += steps - start_at_step

        if arch.is_flux_like:
            positive = self.flux_guidance(cond.positive, cfg if cfg > 1 else 3.5)
            guider = self.basic_guider(model, positive)
        elif cfg == 1.0:
            guider = self.basic_guider(model, cond.positive)
        else:
            guider = self.cfg_guider(model, cond, cfg)

        sigmas = self.scheduler_sigmas(model, scheduler, steps, arch, extent)
        if start_at_step > 0:
            _, sigmas = self.split_sigmas(sigmas, start_at_step)

        return self.add(
            "SamplerCustomAdvanced",
            output_count=2,
            noise=self.random_noise(seed),
            guider=guider,
            sampler=self.sampler_select(sampler),
            sigmas=sigmas,
            latent_image=latent_image,
        )[1]

    def scheduler_sigmas(
        self,
        model: Output,
        scheduler="normal",
        steps=20,
        arch=Arch.sdxl,
        extent: Extent | None = None,
    ):
        if scheduler in ("align_your_steps", "ays"):
            assert arch is Arch.sd15 or arch.is_sdxl_like

            if arch is Arch.sd15:
                model_type = "SD1"
            else:
                model_type = "SDXL"

            return self.add(
                "AlignYourStepsScheduler",
                output_count=1,
                steps=steps,
                model_type=model_type,
                denoise=1.0,
            )
        elif scheduler == "gits":
            return self.add(
                "GITSScheduler",
                output_count=1,
                steps=steps,
                coeff=1.2,
                denoise=1.0,
            )
        elif scheduler in ("polyexponential", "poly_exponential"):
            return self.add(
                "PolyexponentialScheduler",
                output_count=1,
                steps=steps,
                sigma_max=14.614642,
                sigma_min=0.0291675,
                rho=1.0,
            )
        elif scheduler == "laplace":
            return self.add(
                "LaplaceScheduler",
                output_count=1,
                steps=steps,
                sigma_max=14.614642,
                sigma_min=0.0291675,
                mu=0.0,
                beta=0.5,
            )
        elif scheduler == "flux2":
            extent = extent or Extent(1024, 1024)
            return self.add(
                "Flux2Scheduler",
                output_count=1,
                steps=steps,
                width=extent.width,
                height=extent.height,
            )
        else:
            return self.add(
                "BasicScheduler",
                output_count=1,
                model=model,
                scheduler=scheduler,
                steps=steps,
                denoise=1.0,
            )

    def split_sigmas(self, sigmas: Output, step=0):
        return self.add(
            "SplitSigmas",
            output_count=2,
            sigmas=sigmas,
            step=step,
        )

    def basic_guider(self, model: Output, positive: Output):
        return self.add("BasicGuider", 1, model=model, conditioning=positive)

    def cfg_guider(self, model: Output, cond: ConditioningOutput, cfg=7.0):
        return self.add(
            "CFGGuider",
            output_count=1,
            model=model,
            positive=cond.positive,
            negative=cond.negative,
            cfg=cfg,
        )

    def flux_guidance(self, conditioning: Output, guidance=3.5):
        return self.add("FluxGuidance", 1, conditioning=conditioning, guidance=guidance)

    def random_noise(self, noise_seed=-1):
        return self.add_cached(
            "RandomNoise",
            output_count=1,
            noise_seed=noise_seed,
        )

    def sampler_select(self, sampler_name="dpmpp_2m_sde_gpu"):
        if sampler_name == "euler_cfgpp":
            return self.add_cached(
                "SamplerEulerCFGpp",
                output_count=1,
                version="regular",
            )
        else:
            return self.add_cached(
                "KSamplerSelect",
                output_count=1,
                sampler_name=sampler_name,
            )

    def differential_diffusion(self, model: Output):
        return self.add("DifferentialDiffusion", 1, model=model)

    def model_sampling_discrete(self, model: Output, sampling: str, zsnr=False):
        return self.add("ModelSamplingDiscrete", 1, model=model, sampling=sampling, zsnr=zsnr)

    def model_sampling_sd3(self, model: Output, shift=3.0):
        return self.add("ModelSamplingSD3", 1, model=model, shift=shift)

    def rescale_cfg(self, model: Output, multiplier=0.7):
        return self.add("RescaleCFG", 1, model=model, multiplier=multiplier)

    def load_checkpoint(self, checkpoint: str):
        return self.add_cached("CheckpointLoaderSimple", 3, ckpt_name=checkpoint)

    def load_diffusion_model(self, model_name: str):
        if model_name.endswith(".gguf"):
            return self.add_cached("UnetLoaderGGUF", 1, unet_name=model_name)
        return self.add_cached("UNETLoader", 1, unet_name=model_name, weight_dtype="default")

    def load_clip(self, clip_name: str, type: str):
        node = "CLIPLoader"
        if clip_name.endswith(".gguf"):
            node = "CLIPLoaderGGUF"
        return self.add_cached(node, 1, clip_name=clip_name, type=type)

    def load_dual_clip(self, clip_name1: str, clip_name2: str, type: str):
        node = "DualCLIPLoader"
        if any(f.endswith(".gguf") for f in (clip_name1, clip_name2)):
            node = "DualCLIPLoaderGGUF"
        return self.add_cached(node, 1, clip_name1=clip_name1, clip_name2=clip_name2, type=type)

    def load_triple_clip(self, clip_name1: str, clip_name2: str, clip_name3: str):
        node = "TripleCLIPLoader"
        if any(f.endswith(".gguf") for f in (clip_name1, clip_name2, clip_name3)):
            node = "TripleCLIPLoaderGGUF"
        return self.add_cached(
            node, 1, clip_name1=clip_name1, clip_name2=clip_name2, clip_name3=clip_name3
        )

    def load_vae(self, vae_name: str):
        return self.add_cached("VAELoader", 1, vae_name=vae_name)

    def load_controlnet(self, controlnet: str):
        return self.add_cached("ControlNetLoader", 1, control_net_name=controlnet)

    def load_clip_vision(self, clip_name: str):
        return self.add_cached("CLIPVisionLoader", 1, clip_name=clip_name)

    def load_ip_adapter(self, ipadapter_file: str):
        return self.add_cached("IPAdapterModelLoader", 1, ipadapter_file=ipadapter_file)

    def load_upscale_model(self, model_name: str):
        return self.add_cached("UpscaleModelLoader", 1, model_name=model_name)

    def load_style_model(self, model_name: str):
        return self.add_cached("StyleModelLoader", 1, style_model_name=model_name)

    def load_model_patch(self, model_name: str):
        return self.add_cached("ModelPatchLoader", 1, name=model_name)

    def load_lora_model(self, model: Output, lora_name: str, strength: float):
        return self.add(
            "LoraLoaderModelOnly", 1, model=model, lora_name=lora_name, strength_model=strength
        )

    def load_lora(self, model: Output, clip: Output, lora_name, strength_model, strength_clip):
        return self.add(
            "LoraLoader",
            2,
            model=model,
            clip=clip,
            lora_name=lora_name,
            strength_model=strength_model,
            strength_clip=strength_clip,
        )

    def load_insight_face(self):
        return self.add_cached(
            "IPAdapterInsightFaceLoader", 1, provider="CPU", model_name="buffalo_l"
        )

    def load_inpaint_model(self, model_name: str):
        return self.add_cached("INPAINT_LoadInpaintModel", 1, model_name=model_name)

    def load_fooocus_inpaint(self, head: str, patch: str):
        return self.add_cached("INPAINT_LoadFooocusInpaint", 1, head=head, patch=patch)

    def nunchaku_load_flux_diffusion_model(self, model_path: str, cache_threshold: float):
        return self.add_cached(
            "NunchakuFluxDiTLoader", 1, model_path=model_path, cache_threshold=cache_threshold
        )

    def nunchaku_load_qwen_diffusion_model(self, model_name: str, num_blocks_on_gpu=1):
        return self.add_cached(
            "NunchakuQwenImageDiTLoader",
            1,
            model_name=model_name,
            cpu_offload="auto",
            num_blocks_on_gpu=num_blocks_on_gpu,
            use_pin_memory="disable",
        )

    def nunchaku_load_zimage_diffusion_model(self, model_name: str, num_blocks_on_gpu=1):
        return self.add_cached(
            "NunchakuZImageDiTLoader",
            1,
            model_name=model_name,
            cpu_offload="auto",
            num_blocks_on_gpu=num_blocks_on_gpu,
            use_pin_memory="disable",
        )

    def nunchaku_load_flux_lora(self, model: Output, name: str, strength: float):
        return self.add(
            "NunchakuFluxLoraLoader", 1, model=model, lora_name=name, lora_strength=strength
        )

    def t5_tokenizer_options(self, clip: Output, min_padding: int, min_length: int):
        return self.add(
            "T5TokenizerOptions", 1, clip=clip, min_padding=min_padding, min_length=min_length
        )

    def empty_latent_image(self, extent: Extent, arch: Arch, batch_size=1):
        w, h = extent.width, extent.height
        if arch.is_flux_like or arch.is_qwen_like or arch in (Arch.sd3, Arch.chroma, Arch.zimage):
            return self.add("EmptySD3LatentImage", 1, width=w, height=h, batch_size=batch_size)
        if arch.is_flux2:
            return self.add("EmptyFlux2LatentImage", 1, width=w, height=h, batch_size=batch_size)
        else:
            return self.add("EmptyLatentImage", 1, width=w, height=h, batch_size=batch_size)

    def empty_latent_layers(self, extent: Extent, layer_count: int, batch_size=1):
        w, h = extent.width, extent.height
        l = 1 + layer_count * 4  # number of layers for Qwen-Image-Layered
        return self.add(
            "EmptyHunyuanLatentVideo", 1, width=w, height=h, length=l, batch_size=batch_size
        )

    def cut_latent_to_batch(self, latent: Output, dim: str = "t", slice: int = 1):
        return self.add("LatentCutToBatch", 1, samples=latent, dim=dim, slice_size=slice)

    def clip_set_last_layer(self, clip: Output, clip_layer: int):
        return self.add("CLIPSetLastLayer", 1, clip=clip, stop_at_clip_layer=clip_layer)

    def clip_text_encode(self, clip: Output, text: str | Output):
        return self.add("CLIPTextEncode", 1, clip=clip, text=text)

    def conditioning_area(self, conditioning: Output, area: Bounds, strength=1.0):
        return self.add(
            "ConditioningSetArea",
            1,
            conditioning=conditioning,
            x=area.x,
            y=area.y,
            width=area.width,
            height=area.height,
            strength=strength,
        )

    def conditioning_set_mask(self, conditioning: Output, mask: Output, strength=1.0):
        return self.add(
            "ConditioningSetMask",
            1,
            conditioning=conditioning,
            mask=mask,
            strength=strength,
            set_cond_area="default",
        )

    def conditioning_combine(self, a: Output, b: Output):
        return self.add("ConditioningCombine", 1, conditioning_1=a, conditioning_2=b)

    def conditioning_average(self, a: Output, b: Output, strength_a: float):
        return self.add(
            "ConditioningAverage",
            1,
            conditioning_to=a,
            conditioning_from=b,
            conditioning_to_strength=strength_a,
        )

    def conditioning_step_range(self, conditioning: Output, range: tuple[float, float]):
        return self.add(
            "ConditioningSetTimestepRange",
            1,
            conditioning=conditioning,
            start=range[0],
            end=range[1],
        )

    def conditioning_zero_out(self, conditioning: Output):
        return self.add("ConditioningZeroOut", 1, conditioning=conditioning)

    def instruct_pix_to_pix_conditioning(
        self, cond: ConditioningOutput, vae: Output, pixels: Output
    ):
        pos, neg, model = self.add(
            "InstructPixToPixConditioning",
            3,
            positive=cond.positive,
            negative=cond.negative,
            vae=vae,
            pixels=pixels,
        )
        return ConditioningOutput(pos, neg), model

    def reference_latent(self, conditioning: Output, latent: Output):
        return self.add("ReferenceLatent", 1, conditioning=conditioning, latent=latent)

    def text_encode_qwen_image_edit(
        self, clip: Output, vae: Output | None, image: Output, prompt: str | Output
    ):
        return self.add(
            "TextEncodeQwenImageEdit", 1, clip=clip, vae=vae, image=image, prompt=prompt
        )

    def text_encode_qwen_image_edit_plus(
        self, clip: Output, vae: Output | None, images: list[Output], prompt: str | Output
    ):
        image1 = images[0] if len(images) > 0 else None
        image2 = images[1] if len(images) > 1 else None
        image3 = images[2] if len(images) > 2 else None

        return self.add(
            "TextEncodeQwenImageEditPlus",
            1,
            clip=clip,
            vae=vae,
            image1=image1,
            image2=image2,
            image3=image3,
            prompt=prompt,
        )

    def background_region(self, conditioning: Output):
        return self.add("ETN_BackgroundRegion", 1, conditioning=conditioning)

    def define_region(self, regions: Output, mask: Output, conditioning: Output):
        return self.add(
            "ETN_DefineRegion", 1, regions=regions, mask=mask, conditioning=conditioning
        )

    def list_region_masks(self, regions: Output):
        return self.add("ETN_ListRegionMasks", 1, regions=regions)

    def attention_mask(self, model: Output, regions: Output):
        return self.add("ETN_AttentionMask", 1, model=model, regions=regions)

    def apply_controlnet(
        self,
        cond: ConditioningOutput,
        controlnet: Output,
        image: Output,
        vae: Output,
        strength=1.0,
        range: tuple[float, float] = (0.0, 1.0),
    ):
        return ConditioningOutput(
            *self.add(
                "ControlNetApplyAdvanced",
                2,
                positive=cond.positive,
                negative=cond.negative,
                control_net=controlnet,
                image=image,
                vae=vae,
                strength=strength,
                start_percent=range[0],
                end_percent=range[1],
            )
        )

    def apply_controlnet_inpainting(
        self,
        cond: ConditioningOutput,
        controlnet: Output,
        vae: Output,
        image: Output,
        mask: Output,
        strength=1.0,
        range: tuple[float, float] = (0.0, 1.0),
    ):
        return ConditioningOutput(
            *self.add(
                "ControlNetInpaintingAliMamaApply",
                2,
                positive=cond.positive,
                negative=cond.negative,
                control_net=controlnet,
                vae=vae,
                image=image,
                mask=mask,
                strength=strength,
                start_percent=range[0],
                end_percent=range[1],
            )
        )

    def set_controlnet_type(self, controlnet: Output, mode: ControlMode):
        match mode:
            case ControlMode.pose:
                type = "openpose"
            case ControlMode.depth | ControlMode.hands:
                type = "depth"
            case ControlMode.scribble | ControlMode.soft_edge:
                type = "hed/pidi/scribble/ted"
            case ControlMode.line_art | ControlMode.canny_edge:
                type = "canny/lineart/anime_lineart/mlsd"
            case ControlMode.normal:
                type = "normal"
            case ControlMode.segmentation:
                type = "segment"
            case ControlMode.blur:
                type = "tile"
            case _:
                type = "auto"
        return self.add("SetUnionControlNetType", 1, control_net=controlnet, type=type)

    def encode_clip_vision(self, clip_vision: Output, image: Output):
        return self.add("CLIPVisionEncode", 1, clip_vision=clip_vision, image=image)

    def apply_style_model(self, conditioning: Output, style_model: Output, embeddings: Output):
        return self.add(
            "StyleModelApply",
            1,
            conditioning=conditioning,
            style_model=style_model,
            clip_vision_output=embeddings,
        )

    def apply_zimage_fun_controlnet(
        self,
        model: Output,
        patch: Output,
        vae: Output,
        strength: float,
        image: Output,
        mask: Output | None = None,
    ):
        return self.add(
            "ZImageFunControlnet",
            1,
            model=model,
            model_patch=patch,
            vae=vae,
            image=image if not mask else None,
            inpaint_image=image if mask else None,
            mask=mask,
            strength=strength,
        )

    def encode_ip_adapter(
        self, image: Output, weight: float, ip_adapter: Output, clip_vision: Output
    ):
        return self.add(
            "IPAdapterEncoder",
            2,
            image=image,
            weight=weight,
            ipadapter=ip_adapter,
            clip_vision=clip_vision,
        )

    def combine_ip_adapter_embeds(self, embeds: list[Output]):
        e = {f"embed{i + 1}": embed for i, embed in enumerate(embeds)}
        return self.add("IPAdapterCombineEmbeds", 1, method="concat", **e)

    def apply_ip_adapter(
        self,
        model: Output,
        ip_adapter: Output,
        clip_vision: Output,
        embeds: Output,
        weight: float,
        weight_type: str = "linear",
        range: tuple[float, float] = (0.0, 1.0),
        mask: Output | None = None,
    ):
        return self.add(
            "IPAdapterEmbeds",
            1,
            model=model,
            ipadapter=ip_adapter,
            pos_embed=embeds,
            clip_vision=clip_vision,
            weight=weight,
            weight_type=weight_type,
            embeds_scaling="V only",
            start_at=range[0],
            end_at=range[1],
            attn_mask=mask,
        )

    def apply_ip_adapter_face(
        self,
        model: Output,
        ip_adapter: Output,
        clip_vision: Output,
        insightface: Output,
        image: Output,
        weight=1.0,
        range: tuple[float, float] = (0.0, 1.0),
        mask: Output | None = None,
    ):
        return self.add(
            "IPAdapterFaceID",
            1,
            model=model,
            ipadapter=ip_adapter,
            image=image,
            clip_vision=clip_vision,
            insightface=insightface,
            weight=weight,
            weight_faceidv2=weight * 2,
            weight_type="linear",
            combine_embeds="concat",
            start_at=range[0],
            end_at=range[1],
            attn_mask=mask,
            embeds_scaling="V only",
        )

    def apply_self_attention_guidance(self, model: Output):
        return self.add("SelfAttentionGuidance", 1, model=model, scale=0.5, blur_sigma=2.0)

    def inpaint_preprocessor(self, image: Output, mask: Output, fill_black=False):
        return self.add(
            "InpaintPreprocessor", 1, image=image, mask=mask, black_pixel_for_xinsir_cn=fill_black
        )

    def apply_fooocus_inpaint(self, model: Output, patch: Output, latent: Output):
        return self.add("INPAINT_ApplyFooocusInpaint", 1, model=model, patch=patch, latent=latent)

    def vae_encode_inpaint_conditioning(
        self, vae: Output, image: Output, mask: Output, cond: ConditioningOutput
    ):
        pos, neg, latent_inpaint, latent = self.add(
            "INPAINT_VAEEncodeInpaintConditioning",
            4,
            vae=vae,
            pixels=image,
            mask=mask,
            positive=cond.positive,
            negative=cond.negative,
        )
        return ConditioningOutput(pos, neg), latent_inpaint, latent

    def vae_encode(self, vae: Output, image: Output):
        return self.add("VAEEncode", 1, vae=vae, pixels=image)

    def vae_encode_inpaint(self, vae: Output, image: Output, mask: Output):
        return self.add("VAEEncodeForInpaint", 1, vae=vae, pixels=image, mask=mask, grow_mask_by=0)

    def vae_encode_tiled(self, vae: Output, image: Output):
        return self.add(
            "VAEEncodeTiled",
            1,
            vae=vae,
            pixels=image,
            tile_size=512,
            overlap=64,
            temporal_size=64,
            temporal_overlap=8,
        )

    def vae_decode(self, vae: Output, latent_image: Output):
        return self.add("VAEDecode", 1, vae=vae, samples=latent_image)

    def vae_decode_tiled(self, vae: Output, latent_image: Output):
        return self.add(
            "VAEDecodeTiled",
            1,
            vae=vae,
            samples=latent_image,
            tile_size=512,
            overlap=64,
            temporal_size=64,
            temporal_overlap=8,
        )

    def set_latent_noise_mask(self, latent: Output, mask: Output):
        return self.add("SetLatentNoiseMask", 1, samples=latent, mask=mask)

    def batch_latent(self, latent: Output, batch_size: int):
        if batch_size == 1:
            return latent
        return self.add("RepeatLatentBatch", 1, samples=latent, amount=batch_size)

    def crop_latent(self, latent: Output, bounds: Bounds):
        return self.add(
            "LatentCrop",
            1,
            samples=latent,
            x=bounds.x,
            y=bounds.y,
            width=bounds.width,
            height=bounds.height,
        )

    def empty_image(self, extent: Extent, color=0):
        return self.add(
            "EmptyImage", 1, width=extent.width, height=extent.height, color=color, batch_size=1
        )

    def crop_image(self, image: Output, bounds: Bounds):
        return self.add(
            "ImageCrop",
            1,
            image=image,
            x=bounds.x,
            y=bounds.y,
            width=bounds.width,
            height=bounds.height,
        )

    def scale_image(self, image: Output, extent: Extent, method="lanczos"):
        return self.add(
            "ImageScale",
            1,
            image=image,
            width=extent.width,
            height=extent.height,
            upscale_method=method,
            crop="disabled",
        )

    def scale_control_image(self, image: Output, extent: Extent):
        return self.add(
            "HintImageEnchance",
            1,
            hint_image=image,
            image_gen_width=extent.width,
            image_gen_height=extent.height,
            resize_mode="Just Resize",
        )

    def upscale_image(self, upscale_model: Output, image: Output):
        self.sample_count += 4  # approx, actual number depends on model and image size
        return self.add("ImageUpscaleWithModel", 1, upscale_model=upscale_model, image=image)

    def invert_image(self, image: Output):
        return self.add("ImageInvert", 1, image=image)

    def batch_image(self, batch: Output, image: Output):
        return self.add("ImageBatch", 1, image1=batch, image2=image)

    def image_batch_element(self, batch: Output, index: int):
        return self.add("ImageFromBatch", 1, image=batch, batch_index=index, length=1)

    def image_stitch(self, images: list[Output], direction="right"):
        if len(images) == 1:
            return images[0]
        result = images[0]
        for i in images[1:]:
            result = self.add(
                "ImageStitch",
                1,
                image1=result,
                image2=i,
                direction=direction,
                match_image_size=False,
                spacing_width=0,
                spacing_color="white",
            )
        return result

    def inpaint_image(self, model: Output, image: Output, mask: Output):
        return self.add(
            "INPAINT_InpaintWithModel", 1, inpaint_model=model, image=image, mask=mask, seed=834729
        )

    def color_match(
        self, target: Output, reference: Output, exclude_mask: Output | None = None, strength=1.0
    ):
        if strength <= 0.0:
            return target
        return self.add(
            "INPAINT_ColorMatch",
            1,
            target=target,
            reference=reference,
            exclude_mask=exclude_mask,
            strength=strength,
        )

    def crop_mask(self, mask: Output, bounds: Bounds):
        return self.add(
            "CropMask",
            1,
            mask=mask,
            x=bounds.x,
            y=bounds.y,
            width=bounds.width,
            height=bounds.height,
        )

    def scale_mask(self, mask: Output, extent: Extent):
        img = self.mask_to_image(mask)
        scaled = self.scale_image(img, extent, method="bilinear")
        return self.image_to_mask(scaled)

    def image_to_mask(self, image: Output):
        return self.add("ImageToMask", 1, image=image, channel="red")

    def composite_image_masked(
        self, source: Output, destination: Output, mask: Output | None, x=0, y=0
    ):
        return self.add(
            "ImageCompositeMasked",
            1,
            source=source,
            destination=destination,
            mask=mask,
            x=x,
            y=y,
            resize_source=False,
        )

    def mask_to_image(self, mask: Output):
        return self.add("MaskToImage", 1, mask=mask)

    def batch_mask(self, batch: Output, mask: Output):
        image_batch = self.mask_to_image(batch)
        image = self.mask_to_image(mask)
        return self.image_to_mask(self.batch_image(image_batch, image))

    def mask_batch_element(self, mask_batch: Output, index: int):
        image_batch = self.mask_to_image(mask_batch)
        image = self.image_batch_element(image_batch, index)
        return self.image_to_mask(image)

    def solid_mask(self, extent: Extent, value=1.0):
        return self.add("SolidMask", 1, width=extent.width, height=extent.height, value=value)

    def threshold_mask(self, mask: Output, threshold: float):
        return self.add("ThresholdMask", 1, mask=mask, value=threshold)

    def fill_masked(self, image: Output, mask: Output, mode="neutral", falloff: int = 0):
        return self.add("INPAINT_MaskedFill", 1, image=image, mask=mask, fill=mode, falloff=falloff)

    def blur_masked(self, image: Output, mask: Output, blur: int, falloff: int = 0):
        return self.add("INPAINT_MaskedBlur", 1, image=image, mask=mask, blur=blur, falloff=falloff)

    def expand_mask(self, mask: Output, grow: int, blur: int, kernel="gaussian"):
        return self.add("INPAINT_ExpandMask", 1, mask=mask, grow=grow, blur=blur, blur_type=kernel)

    def shrink_mask(self, mask: Output, shrink: int, blur: int, kernel="gaussian"):
        return self.add(
            "INPAINT_ShrinkMask", 1, mask=mask, shrink=shrink, blur=blur, blur_type=kernel
        )

    def stabilize_mask(self, mask: Output, epsilon=0.01):
        return self.add("INPAINT_StabilizeMask", 1, mask=mask, epsilon=epsilon)

    def apply_mask(self, image: Output, mask: Output):
        return self.add("ETN_ApplyMaskToImage", 1, image=image, mask=mask)

    def translate(self, text: str | Output):
        return self.add("ETN_Translate", 1, text=text)

    def nsfw_filter(self, image: Output, sensitivity: float):
        if sensitivity > 0:
            return self.add("ETN_NSFWFilter", 1, image=image, sensitivity=sensitivity)
        return image

    def load_image(self, image: Image | ImageCollection):
        if self._run_mode is ComfyRunMode.runtime:
            return self._load_image_batch(image, self._load_image_runtime, self.batch_image)
        else:
            return self._load_image_batch(image, self._load_image_cache, self.batch_image)

    def load_mask(self, mask: Image | ImageCollection):
        if self._run_mode is ComfyRunMode.runtime:
            return self._load_image_batch(mask, self._load_mask_runtime, self.batch_mask)
        else:
            return self._load_image_batch(mask, self._load_mask_cache, self.batch_mask)

    def _load_image_runtime(self, image: Image):
        return self.add("ETN_InjectImage", 1, id=self._add_image(image))

    def _load_image_cache(self, image: Image):
        return self.add("ETN_LoadImageCache", 2, id=self._add_image_hashed(image))[0]

    def _load_mask_runtime(self, mask: Image):
        return self.add("ETN_InjectMask", 1, id=self._add_image(mask))

    def _load_mask_cache(self, mask: Image):
        return self.add("ETN_LoadImageCache", 2, id=self._add_image_hashed(mask))[1]

    def _load_image_batch(self, images: Image | ImageCollection, loader, batcher) -> Output:
        if isinstance(images, Image):
            return loader(images)
        result = None
        for image in images:
            img = loader(image)
            result = img if result is None else batcher(result, img)
        assert result is not None
        return result

    def load_image_and_mask(self, images: Image | ImageCollection):
        assert self._run_mode is ComfyRunMode.server
        if isinstance(images, Image):
            return self.add("ETN_LoadImageCache", 2, id=self._add_image_hashed(images))
        result = None
        for image in images:
            img, mask = self.add("ETN_LoadImageCache", 2, id=self._add_image_hashed(image))
            if result:
                result = (self.batch_image(result[0], img), self.batch_mask(result[1], mask))
            else:
                result = (img, mask)
        assert result is not None
        return result

    def send_image(self, image: Output):
        if self._run_mode is ComfyRunMode.runtime:
            return self.add("ETN_ReturnImage", 1, images=image)
        return self.add("ETN_SaveImageCache", 1, images=image, format="PNG")

    def save_image(self, image: Output, prefix: str):
        return self.add("SaveImage", 1, images=image, filename_prefix=prefix)

    def create_tile_layout(
        self, image: Output, tile_size: int, padding: int, blending: int, multiple: int
    ):
        return self.add(
            "ETN_TileLayout",
            1,
            image=image,
            min_tile_size=tile_size,
            padding=padding,
            blending=blending,
            multiple=multiple,
        )

    def extract_image_tile(self, image: Output, layout: Output, index: int):
        return self.add("ETN_ExtractImageTile", 1, image=image, layout=layout, index=index)

    def extract_mask_tile(self, mask: Output, layout: Output, index: int):
        return self.add("ETN_ExtractMaskTile", 1, mask=mask, layout=layout, index=index)

    def merge_image_tile(self, image: Output, layout: Output, index: int, tile: Output):
        return self.add("ETN_MergeImageTile", 1, layout=layout, index=index, tile=tile, image=image)

    def generate_tile_mask(self, layout: Output, index: int):
        return self.add("ETN_GenerateTileMask", 1, layout=layout, index=index)

    def define_reference_image(
        self, references: Output | None, image: Output, weight: float, range: tuple[float, float]
    ):
        return self.add(
            "ETN_ReferenceImage",
            1,
            image=image,
            weight=weight,
            range_start=range[0],
            range_end=range[1],
            reference_images=references,
        )

    def apply_reference_images(
        self, conditioning: Output, clip_vision: Output, style_model: Output, references: Output
    ):
        return self.add(
            "ETN_ApplyReferenceImages",
            1,
            conditioning=conditioning,
            clip_vision=clip_vision,
            style_model=style_model,
            references=references,
        )

    def estimate_pose(self, image: Output, resolution: int):
        feat = {"detect_hand": "enable", "detect_body": "enable", "detect_face": "enable"}
        mdls = {"bbox_detector": "yolox_l.onnx", "pose_estimator": "dw-ll_ucoco_384.onnx"}
        if self._run_mode is ComfyRunMode.runtime:
            # use smaller model, but it requires onnxruntime, see #630
            mdls["bbox_detector"] = "yolo_nas_l_fp16.onnx"
        return self.add("DWPreprocessor", 1, image=image, resolution=resolution, **feat, **mdls)

    def easy_cache(self, model: Output, arch: Arch):
        threshold = 0.2 if arch.is_sdxl_like else 0.12
        return self.add(
            "EasyCache",
            1,
            model=model,
            reuse_threshold=threshold,
            start_percent=0.15,
            end_percent=0.95,
            verbose=False,
        )

    def create_hook_lora(self, loras: list[tuple[str, float]]):
        key = "CreateHookLora" + str(loras)
        hooks = self._cache.get(key, None)
        if hooks is None:
            for lora, strength in loras:
                hooks = self.add(
                    "CreateHookLora",
                    1,
                    lora_name=lora,
                    strength_model=strength,
                    strength_clip=strength,
                    prev_hooks=hooks,
                )
            assert hooks is not None
            self._cache[key] = hooks

        assert isinstance(hooks, Output)
        return hooks

    def set_clip_hooks(self, clip: Output, hooks: Output):
        return self.add(
            "SetClipHooks", 1, clip=clip, hooks=hooks, apply_to_conds=True, schedule_clip=False
        )

    def combine_masked_conditioning(
        self,
        positive: Output,
        negative: Output,
        positive_conds: Output | None = None,
        negative_conds: Output | None = None,
        mask: Output | None = None,
    ):
        assert (positive_conds and negative_conds) or mask
        if mask is None:
            return self.add(
                "PairConditioningSetDefaultCombine",
                2,
                positive=positive_conds,
                negative=negative_conds,
                positive_DEFAULT=positive,
                negative_DEFAULT=negative,
            )
        if positive_conds is None and negative_conds is None:
            return self.add(
                "PairConditioningSetProperties",
                2,
                positive_NEW=positive,
                negative_NEW=negative,
                mask=mask,
                strength=1.0,
                set_cond_area="default",
            )
        return self.add(
            "PairConditioningSetPropertiesAndCombine",
            2,
            positive=positive_conds,
            negative=negative_conds,
            positive_NEW=positive,
            negative_NEW=negative,
            mask=mask,
            strength=1.0,
            set_cond_area="default",
        )


# Node descriptions available to the ComfyUI server from /object_info query
class ComfyObjectInfo:
    def __init__(self, nodes: dict[str, dict]):
        self.nodes = nodes

    def __contains__(self, node_class: str):
        return node_class in self.nodes

    def __bool__(self):
        return bool(self.nodes)

    def params(self, node_class: str, category=""):
        result: dict[str, Any] = {}
        if inputs := self.inputs(node_class, category):
            for k, v in inputs.items():
                default = None
                if len(v) > 1 and isinstance(v[1], dict):
                    default = v[1].get("default")
                if default is None and isinstance(v[0], list) and len(v[0]) > 0:
                    # legacy combo type, use first value in list of possible values
                    default = v[0][0]
                if default is None:
                    match v[0]:
                        case "INT":
                            default = 0
                        case "FLOAT":
                            default = 0.0
                        case "BOOL":
                            default = False
                        case "STRING":
                            default = ""
                        case "COMBO":
                            if options := v[1].get("options", None):
                                default = options[0]
                result[k] = default
        return result

    def options(self, node_class: str, param_name: str) -> list[str]:
        if inputs := self.inputs(node_class, "required"):
            if param := inputs.get(param_name, None):
                if param[0] == "COMBO":
                    return param[1]["options"]
                elif isinstance(param[0], list):
                    return param[0]
        else:
            log.warning(f"Failed to get {node_class} options for {param_name}")
        return []

    def inputs(self, node_name: str, category="") -> dict[str, list] | None:
        node = self.nodes.get(node_name)
        if node is None:
            return None
        inputs = node.get("input", {})
        if category:
            return inputs.get(category)
        result = {}
        result.update(inputs.get("required", {}))
        result.update(inputs.get("optional", {}))
        return result

    def outputs(self, node_name: str) -> list[str]:
        node = self.nodes.get(node_name)
        if node is None:
            return []
        return node.get("output_name", [])


def _convert_ui_workflow(w: dict, node_inputs: ComfyObjectInfo):
    version = w.get("version")
    nodes = w.get("nodes")
    links = w.get("links")
    if not (version and nodes and links):
        return w

    if not node_inputs.nodes:
        raise ValueError("An active ComfyUI connection is required to convert a UI workflow file.")

    primitives = {}
    for node in nodes:
        if node["type"] == "PrimitiveNode":
            primitives[node["id"]] = node["widgets_values"][0]

    r = {}
    for node in nodes:
        id = node["id"]
        type = node["type"]
        if type == "PrimitiveNode":
            continue

        inputs = {}
        fields = node_inputs.inputs(type)
        if fields is None:
            raise ValueError(
                f"Workflow uses node type {type}, but it is not installed on the ComfyUI server."
            )
        widget_count = 0
        for field_name, field in fields.items():
            field_type = field[0]
            if isinstance(field_type, list):
                field_type = "COMBO"
            if field_type in ("INT", "FLOAT", "BOOL", "STRING", "COMBO"):
                values = node["widgets_values"]
                inputs[field_name] = values[widget_count]
                widget_count += 1
                if len(values) > widget_count and values[widget_count] in _control_after_generate:
                    widget_count += 1
                if type == "ETN_Parameter" and widget_count >= len(values):
                    break  # min/max widgets are not visible for non-numeric parameters

            for connection in node["inputs"]:
                if connection["name"] == field_name and connection["link"] is not None:
                    link = next(x for x in links if x[0] == connection["link"])
                    prim = primitives.get(link[1])
                    if prim is not None:
                        inputs[field_name] = prim
                    else:
                        inputs[field_name] = [link[1], link[2]]
                    break
        r[id] = {"class_type": type, "inputs": inputs}

    return r


_control_after_generate = ["fixed", "increment", "decrement", "randomize"]
