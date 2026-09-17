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

  export const CONSTANTS: {
    SAMPLE_RATE: number;
    FFT_SIZE: number;
    HOP_SIZE: number;
    TRAINING_SAMPLES: number;
    MODEL_SPEC_BINS: number;
    MODEL_SPEC_FRAMES: number;
    SEGMENT_OVERLAP: number;
    TRACKS: string[];
    DEFAULT_MODEL_URL: string;
  };
}
