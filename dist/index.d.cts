declare class InitializationError extends Error {
    constructor();
}
declare class SharpError extends Error {
    readonly error: Error;
    constructor(error: Error);
}
declare class InternalError extends Error {
    constructor();
}
declare class EchikanaInferencer {
    private readonly model;
    private readonly size;
    private readonly output;
    private session;
    get initialized(): boolean;
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
    constructor(model: ArrayBufferLike);
    /**
     * Initialize the inference session.
     * If the session has already been initialized, this method dispose the existing session and recreate it.
     *
     * This method is automatically called when the `inference` method is called if the session has not been initialized.
     * However, it is recommended that this method be called manually beforehand, because it takes time to create the session.
     */
    initialize(): Promise<void>;
    /**
     * Takes an image buffer and returns the “NSFW degree of the image” as a number between 0.0 and 1.0.
     *
     * If the inference session has not yet been initialized, the `initialize()` method is automatically called here.
     * However, it is recommended that `initialize()` method be called manually beforehand, because it takes time to create the session.
     * If the session has already been initialized, this method does not create or recreate the session.
     */
    inference(image: ArrayBufferLike): Promise<number>;
    dispose(): Promise<void>;
}

declare const errors: {
    InitializationError: typeof InitializationError;
    SharpError: typeof SharpError;
    InternalError: typeof InternalError;
};

export { EchikanaInferencer, errors };
