import ort, { type InferenceSession } from "onnxruntime-node";
import sharp from "sharp";
import { softmax } from "./softmax.js";

export class InitializationError extends Error {
  public constructor() {
    super("The inference model has not been initialized.");
  }
}

export class SharpError extends Error {
  public constructor(public readonly error: Error) {
    super("Image format conversion by sharp failed.");
  }
}

export class InternalError extends Error {
  public constructor() {
    super("An internal error has occurred.");
  }
}

export class EchikanaInferencer {
  private readonly size = 224;
  private readonly output = "logits";

  private session: InferenceSession | null = null;

  public get initialized(): boolean {
    return this.session !== null;
  }

  /**
   * ```ts
   * // Simple Use Cases
   *
   * import fs from "node:fs";
   *
   * const model = fs.readFileSync("path/to/model.onnx");
   * const inferencer = new EchikanaInferencer(model);
   *
   * await inferencer.initialize();
   *
   * const image = fs.readFileSync("path/to/image.jpg");
   * const result: number = await inferencer.inference(image);
   *
   * console.log(`The image is ${Math.round(result * 100)}% NSFW!`);
   * ```
   */
  public constructor(private readonly model: ArrayBufferLike) {}

  /**
   * Initialize the inference session.
   * If the session has already been initialized, this method dispose the existing session and recreate it.
   *
   * This method is automatically called when the `inference` method is called if the session has not been initialized.
   * However, it is recommended that this method be called manually beforehand, because it takes time to create the session.
   */
  public async initialize(): Promise<void> {
    if (this.initialized) {
      await this.dispose();
    }

    this.session = await ort.InferenceSession.create(this.model);
  }

  /**
   * Takes an image buffer and returns the “NSFW degree of the image” as a number between 0.0 and 1.0.
   *
   * If the inference session has not yet been initialized, the `initialize()` method is automatically called here.
   * However, it is recommended that `initialize()` method be called manually beforehand, because it takes time to create the session.
   * If the session has already been initialized, this method does not create or recreate the session.
   */
  public async inference(image: ArrayBufferLike): Promise<number> {
    if (!this.initialized) {
      await this.initialize();
    }

    if (this.session === null) {
      throw new InitializationError();
    }

    const pixels = await (async () => {
      try {
        return await sharp(image)
          .resize(this.size, this.size, {
            kernel: "nearest",
            fit: "fill", // Resize without regard to aspect ratio
          })
          .removeAlpha() // Remove transparency information
          .raw() // Convert to an unsigned 8-bit integer array
          .toBuffer();
      } catch (error: unknown) {
        if (error instanceof Error) return new SharpError(error);
        return new InternalError();
      }
    })();
    if (pixels instanceof Error) throw pixels;

    const redComponents: number[] = [];
    const greenComponents: number[] = [];
    const blueComponents: number[] = [];

    for (let i = 0; i < pixels.byteLength; i += 3) {
      const red = pixels[i + 0];
      if (red === undefined) throw new InternalError();
      redComponents.push(red);

      const green = pixels[i + 1];
      if (green === undefined) throw new InternalError();
      greenComponents.push(green);

      const blue = pixels[i + 2];
      if (blue === undefined) throw new InternalError();
      blueComponents.push(blue);
    }

    const transposed = [
      ...redComponents,
      ...greenComponents,
      ...blueComponents,
    ];

    // Normalizes values in the range 0x00 to 0xff to a range from -1.0 to +1.0.
    const normalized = transposed.map((v) => (v / 0xff - 0.5) / 0.5);

    const inputTensor = new ort.Tensor(new Float32Array(normalized), [
      1,
      3,
      this.size,
      this.size,
    ]);

    const result = await this.session.run({
      pixel_values: inputTensor,
    });

    const outputTensor = result[this.output];
    if (outputTensor === undefined) throw new InternalError();

    const data = outputTensor.data;
    if (!(data instanceof Float32Array)) throw new InternalError();

    const [, nsfw] = softmax([...data]);
    if (nsfw === undefined) throw new InternalError();

    // Clean up
    inputTensor.dispose();
    outputTensor.dispose();

    return nsfw;
  }

  public async dispose(): Promise<void> {
    await this.session?.release();
  }
}
