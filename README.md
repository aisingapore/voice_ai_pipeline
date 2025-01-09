# Realtime Voice AI using Open Source models only

Like OpenAI's realtime voice AI, using open source models only. 

https://github.com/user-attachments/assets/4579e8fc-011e-4061-b987-9bc12b78848d

# Set Up

## Overview
There are four components involved, mainly a [Pipecat](https://github.com/pipecat-ai/pipecat) server, found in this repository which orchestrate the end-to-end pipeline, and three distinct models for each step in the pipeline: speech-to-text (STT) model, instruction-tuned text completion model and text-to-speech model (TTS). In this case, [Whisper](https://huggingface.co/openai/whisper-large-v2), [SEA-LIONv2](https://huggingface.co/aisingapore/llama3-8b-cpt-sea-lionv2.1-instruct) and [XTTS](https://huggingface.co/coqui/XTTS-v2) is used correspondingly. Other models can be substituted, such as [Parler-TTS](https://github.com/huggingface/parler-tts) for TTS with some adjustments to the current code. 

To support realtime capabilities, GPU-acceleration is required for running models, so you will need to host each model on a [L40s](https://aws.amazon.com/ec2/instance-types/g6e/) GPU minimally based on my experience, to achieve realtime performance. The hosted server should then be integrated to the interface provided by Pipecat, which might require additional effort to do so depending on how the substitute model encodes input and decodes output. This part can be tricky and model dependent, especially for the STT and TTS step.

## Pipecat Orchestration Server 

1. Install dependencies

```bash
python -m venv venv
source venv/bin/active # or OS equivalent
pip install -r requirements.txt
```

2. Setup .env

```bash
cp env.example .env
```

Alternatively, you can configure your Modal app to use [secrets](https://modal.com/docs/guide/secrets)

3. Test the app locally

```bash
modal serve app.py # run the server
curl -X POST https://{modal_dev_url} # POST request to create Daily Room
```

4. Deploy to production

```bash
modal deploy app.py
```

## Configuration options

This app sets some sensible defaults for reducing cold starts, such as `minkeep_warm=1`, which will keep at least 1 warm instance ready for your bot function.

It has been configured to only allow a concurrency of 1 (`max_inputs=1`) as each user will require their own running function.

# Configuring STT and TTS Servers
## Speech-to-Text (STT)
My Whisper streaming server implementation can be found [here](https://github.com/theogbrand/whisper-server), which is ran using docker.

## Text-to-Speech (TTS)
[XTTS streaming server](https://github.com/coqui-ai/xtts-streaming-server/blob/main/server/main.py) is ran here using docker.

## Cloud Hosting Set Up
Having tried model hosting services like Replicate, Modal, RunPod, LambdaLabs, I find the simplest way to deploy the inference servers to be a canonical AWS GPU server. You can use my [Packer image](https://github.com/theogbrand/ai-server-setup/blob/main/aws/packer/l40s-48gb-ubuntu-docker-nvidia.pkr.hcl) to easily set up CUDA and Docker dependencies. The same image is used to set up both Whisper and XTTS servers above.

Hosting inference servers can be quite expensive, a single L40S up for 24 hours will cost about ~$5000/month which make API services more attractive. 

Another option is to explore serverless inference for perpetual uptime or a using AWS's native scheduler to trigger a Lambda to shut down the server overnight if only used for development.

# User Interface
Code for the user interface used in the video can be found [here](https://github.com/theogbrand/realtime-voicebot-ui).

Alternatively, you can call the modal endpoint using ```curl -X POST {MODAL_URL}``` and recieve the daily room_url.

Example curl reponse:
```json
{"room_url":"https://ob1-aisg.daily.co/pcJMY4YkWMrKuNdvHSA5","eyJhbGciOiJIUzI1NiIsIn...":"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyI..."}
```

# References

## Audio foundation models
* Modal [Seamless M4T](https://modal.com/docs/examples/seamless-chat), [code](https://github.com/modal-labs/seamless-chat/blob/main/seamless.py)

## Fusion models
* [Moshi](https://github.com/modal-labs/quillman/blob/main/src/moshi.py)

## OpenAI Compatible Whisper Servers
* [Faster Whisper Server](https://github.com/fedirz/faster-whisper-server)
* [More barebones version](https://github.com/matatonic/openedai-whisper)
* [another small version](https://github.com/morioka/tiny-openai-whisper-api)

## Whisper Streaming Servers
* https://github.com/ufal/whisper_streaming?tab=readme-ov-file
* https://github.com/ggoonnzzaallo/llm_experiments/blob/main/streamed_text_plus_streamed_audio.py
* https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/openai_whisper/streaming/main.py

