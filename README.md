# Echikana 🫣

A Node.js library for NSFW (not safe for work) classification of images using [Falconsai/nsfw_image_detection](https://huggingface.co/Falconsai/nsfw_image_detection).

```ts
import fs from "node:fs";
import { EchikanaInferencer } from "echikana";

const model = fs.readFileSync("path/to/model.onnx");
const inferencer = new EchikanaInferencer(model);

await inferencer.initialize();

const image = fs.readFileSync("path/to/image.jpg");
const result: number = await inferencer.inference(image);

console.log(`The image is ${Math.round(result * 100)}% NSFW!`);
```

## API

First, instantiate the `EchikanaInferencer` class by passing the model data (`ArrayBufferLike`) as an argument to the constructor function.

```ts
import fs from "node:fs";
import { EchikanaInferencer } from "echikana";

const model = fs.readFileSync("path/to/model.onnx");
// const model = await fetch("https://www.example.com/path/to/model.onnx").then((response) => response.arrayBuffer());

const inferencer = new EchikanaInferencer(model);
```

Next, call the `initialize` method to initialize the model.
The model will not be loaded until this method is called.

```ts
await inferencer.initialize();
```

Now, let's perform inference!
The `inference` method accepts an image as an `ArrayBufferLike` and returns a fraction from 0 to 1 indicating the NSFW probability of the image.

```ts
const image = fs.readFileSync("path/to/image.jpg");
const result: number = await inferencer.inference(image);

console.log(`The image is ${Math.round(result * 100)}% NSFW!`);
```

> [!NOTE]
> During inference, the image is transformed using [sharp](https://github.com/lovell/sharp) for input to the model.
> The image formats that can be converted are limited to those supported by sharp.

If you need to discard a loaded model, use the `dispose` method.

To reload the model, call the `initialize` method again.
The previously loaded model will be disposed of and newly initialized.

# About the inference model

This library uses [onnxruntime-node](https://www.npmjs.com/package/onnxruntime-node) as the runtime for model execution.
Therefore, to use the [“Falconsai/nsfw_image_detection” model published on Hugging Face](https://huggingface.co/Falconsai/nsfw_image_detection), you must first convert it to ONNX format.

> [!WARNING]
> No inference model is included in this package or repository.

Please use [Optimum](https://github.com/huggingface/optimum), a conversion tool by Hugging Face.

```sh
optimum-cli export onnx --model Falconsai/nsfw_image_detection /path/to/model-dir
```

This repository includes a `Dockerfile` and a `compose.yaml` file, which are useful for converting the model to ONNX format.
To use them, run the following commands.

```sh
git clone https://github.com/okayurisotto/echikana.git echikana
cd echikana
docker compose run --rm --build optimum
```

The converted model file will be created in the `model` directory.
