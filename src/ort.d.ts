// Type declarations for onnxruntime-web (v1.21.0 doesn't ship types)
declare module 'onnxruntime-web' {
  export namespace env {
    let logLevel: 'verbose' | 'info' | 'warning' | 'error' | 'fatal';
    namespace wasm {
      let numThreads: number;
      let wasmPaths: string;
    }
  }

  export class InferenceSession {
    inputNames: string[];
    outputNames: string[];
    
    static create(
      model: ArrayBuffer | Uint8Array | string,
      options?: InferenceSession.SessionOptions
    ): Promise<InferenceSession>;
    
    run(
      feeds: Record<string, Tensor>,
      options?: InferenceSession.RunOptions
    ): Promise<Record<string, Tensor>>;
    
    release(): Promise<void>;
    startProfiling(): void;
    endProfiling(): void;
  }

  export namespace InferenceSession {
    interface SessionOptions {
      executionProviders?: ExecutionProviderConfig[] | string[];
      graphOptimizationLevel?: 'disabled' | 'basic' | 'extended' | 'all';
      enableCpuMemArena?: boolean;
      enableMemPattern?: boolean;
    }
    
    interface RunOptions {
      logSeverityLevel?: number;
    }
    
    type ExecutionProviderConfig = string | Record<string, unknown>;
  }

  export class Tensor {
    constructor(
      type: 'float32' | 'float64' | 'string' | 'int8' | 'uint8' | 'int32' | 'int64' | 'bool',
      data: Float32Array | Int32Array | BigInt64Array | Uint8Array | string[],
      dims?: number[]
    );
    
    data: Float32Array | Int32Array | BigInt64Array | Uint8Array | string[];
    dims: readonly number[];
    type: string;
  }
}
