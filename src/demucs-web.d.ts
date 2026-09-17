declare module 'demucs-web' {
  export interface DemucsProcessorOptions {
    ort: any;
    onProgress?: (progress: any) => void;
    onLog?: (phase: string, msg: string) => void;
  }

  export interface SeparationResult {
    drums: { left: Float32Array; right: Float32Array };
    bass: { left: Float32Array; right: Float32Array };
    other: { left: Float32Array; right: Float32Array };
    vocals: { left: Float32Array; right: Float32Array };
  }

  export class DemucsProcessor {
    constructor(options: DemucsProcessorOptions);
    loadModel(modelPathOrBuffer: string | ArrayBuffer): Promise<void>;
    separate(leftChannel: Float32Array, rightChannel: Float32Array): Promise<SeparationResult>;
  }
}
