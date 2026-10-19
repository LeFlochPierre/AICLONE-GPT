<h1 align="center">🤖 AICLONE-GPT</h1>
<p align="center"><strong>Fine-tune an AI digital twin from your WhatsApp conversations.</strong></p>

<p align="center">
  <img src="imgs/readme_header_banner.png" alt="AICLONE-GPT banner" width="75%">
</p>

<p align="center">
  <a href="https://pytorch.org/get-started/locally/"><img src="https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct"><img src="https://img.shields.io/badge/Llama%203-8B-4677c5" alt="Llama 3 8B"></a>
  <a href="https://github.com/plefloch123/torchtune"><img src="https://img.shields.io/badge/torchtune-custom%20fork-747d8c" alt="Custom torchtune fork"></a>
</p>

AICLONE-GPT trains a conversational language model on exported WhatsApp messages to approximate the writing style, tone, and conversational habits of a particular person. The current implementation uses a fine-tuned **Llama 3 8B** model and a **WhatsApp-inspired local web interface**.

## Demo

<p align="center">
  <a href="demo/media/aiclone-demo-casual.mp4">
    <img src="demo/media/aiclone-demo-casual.webp" alt="AICLONE-GPT demo: live chats, creativity controls, and Clone yourself" width="80%">
  </a>
</p>

## Features

- **WhatsApp preprocessing:** Convert exported `.txt` conversations into ShareGPT-style training data.
- **QLoRA fine-tuning:** Adapt Llama 3 8B to a personal conversation style using a customised torchtune fork.
- **Interactive chat:** Send messages in a WhatsApp-style browser interface with progressively generated replies and animated typing indicators.
- **Conversation controls:** Stop generation, regenerate the last answer, start a new chat, copy messages, and export a conversation.
- **Display options:** Timestamps, adjustable sampling temperature, responsive layout, and light/dark mode.
- **Local inference:** The app runs on your own GPU/server; the web UI is accessed locally or through an SSH tunnel.

> The interface is inspired by WhatsApp; it is **not connected to WhatsApp** and does not send or receive messages through the WhatsApp service. Conversation state is held by the running server and is lost when it restarts unless exported.

## Getting started

### 1. Clone the repository

```bash
git clone --recurse-submodules git@github.com:LeFlochPierre/AICLONE-GPT.git
cd AICLONE-GPT
```

The `torchtune/` submodule points to a customised fork. The SSH clone command requires GitHub SSH access.

### 2. Set up the environment

Use a Python 3.10 environment with a CUDA-capable PyTorch installation matching your GPU and driver. The working development environment used PyTorch 2.6.0+cu118 and a customised torchtune installation; newer upstream torchtune APIs are **not** interchangeable with this code without testing.

```bash
conda activate AICLONE  # if you have already created this environment
pip install -r requirements.txt
pip install -e ./torchtune
```

The local project also contains `environment.yml` as an alternative environment specification. Review the dependency versions before modifying an already working installation; do not indiscriminately upgrade torchtune.

### 3. Download the base model

First, request access to [Meta Llama 3 8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), if required.

```bash
tune download meta-llama/Meta-Llama-3-8B-Instruct \
  --output-dir model/llama3-8B/original \
  --hf-token YOUR_HF_TOKEN
```

Ensure the checkpoint filename and tokenizer paths match the entries in `config/llama3-8B/qlora_train_config.yaml`. The exact file layout may depend on the torchtune download command/version.

### 4. Prepare your WhatsApp data

1. Export chats as plain text **without media** from WhatsApp.
2. Place the exported `.txt` files in `data/unformatted_text/`.
3. Run the preprocessing script with your display name as it appears in the exports:

```bash
python process_data.py "Your WhatsApp Name"
```

This produces normalized text in `data/raw_data/` and ShareGPT-style JSON files in `data/preprocessed/`. Messages attributed to your display name are labelled as assistant responses during preprocessing.

The current parser expects date-prefixed lines in the form `DD/MM/YYYY, HH:MM - Name: Message`. If your export uses a different format, adjust the regex in `process_data.py`.

**Privacy:** WhatsApp exports can contain other people's personal messages. Obtain appropriate consent before using or sharing conversation data. The repository excludes local training data and model weights; do not commit personal chat exports, credentials, or private persona prompts.

## Fine-tuning

The Llama 3 8B QLoRA training configuration is located at `config/llama3-8B/qlora_train_config.yaml`.

```bash
tune run lora_finetune_single_device \
  --config config/llama3-8B/qlora_train_config.yaml
```

Review the training config for checkpoint paths, dataset location, batch size, learning rate, and GPU memory requirements before running. The project also includes `config/llama3-8B/train_config.yaml` for a separate LoRA setup.

The repository's current ready-to-use configuration targets **Llama 3 8B**. Older experimental Mistral references have been omitted because corresponding Mistral configurations are not currently provided.

## Chat with your AI clone

After preparing a compatible fine-tuned **full-model checkpoint**, run the WhatsApp-style app:

```bash
conda activate AICLONE
python chat_app.py --config config/llama3-8B/inference_config.yaml
```

Open **http://localhost:7860** in your browser if the app is running on your own machine.

### Using a remote GPU/server

If the app runs on a remote machine, keep it listening on `127.0.0.1` and connect through an SSH tunnel **from your laptop**:

```bash
ssh -N -L 7860:127.0.0.1:7860 user@remote-host
```

Then open **http://localhost:7860** on your laptop. Alternatively, use **VS Code → Ports → Forward a Port** and enter `7860`. You may need your institution's SSH gateway if the GPU host cannot be reached directly.

### Local inference configuration

`config/llama3-8B/inference_config.yaml` and the private persona prompt are intentionally excluded from Git. Create them locally and adjust the paths to your own checkpoint:

```yaml
# config/llama3-8B/inference_config.yaml
model:
  _component_: torchtune.models.llama3.llama3_8b

tokenizer:
  _component_: torchtune.models.llama3.llama3_tokenizer
  path: model/llama3-8B/original/tokenizer.model

checkpointer:
  _component_: torchtune.utils.FullModelMetaCheckpointer
  checkpoint_dir: model/finetune_checkpoint_8B
  checkpoint_files: [meta_model_3.pt]
  output_dir: model/finetune_checkpoint_8B
  model_type: LLAMA3

device: cuda
dtype: fp16
seed: 42
offload: true
quantizer: null
max_seq_len: 8192
max_new_tokens: 512
temperature: 0.8
top_k: 50
system_prompt_file: data/persona/system_prompt.txt
```

Save your persona instructions at `data/persona/system_prompt.txt`. The example assumes `meta_model_3.pt` is a full-model checkpoint already available in `model/finetune_checkpoint_8B/`; adapt it if your training run produced a different filename or only LoRA adapters. The model checkpoint and raw chat data are **not included** in the public repository.

## Hardware and compatibility

A GPU with enough VRAM for Llama 3 8B inference is required. Memory use depends on precision, quantization, context length, and the training configuration. QLoRA training also requires a supported GPU/CUDA setup; the amount of free memory matters more than the GPU model name alone.

This project was developed using an older, customised torchtune API. In particular, the streaming backend is tailored to decoder/cache behaviour from that version. If something breaks after upgrading PyTorch or torchtune, verify the original environment and dependency versions before changing the application.

## Troubleshooting

**`FileNotFoundError` for `config/llama3/inference_config.yaml`**  
The folder is `config/llama3-8B/`, not `config/llama3/`. The inference YAML is private and must exist locally.

**`TransformerDecoder` has no `caches_are_enabled` or `reset_caches`**  
These methods are absent from the customised torchtune installation. Use the compatible streaming `chat_app.py` and do not assume newer cache-management APIs are present.

**Browser cannot reach `localhost:7860`**  
Confirm the app is running on the GPU host and forward port `7860` to the correct machine. The SSH tunnel must be started on your laptop (or use VS Code port forwarding).

**Model responds differently from your writing style**  
Review the quality and balance of the training conversations, the persona prompt, and the sampling temperature. Increasing the epoch count alone can overfit the training data.

## Notes

- The public repository contains source code, not the personal conversation exports or trained model checkpoint.
- This is an experimental personal-model project, not a production messaging or identity-authentication service.
