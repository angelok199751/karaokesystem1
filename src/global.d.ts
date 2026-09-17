/// <reference types="@webgpu/types" />

// WebGPU types for older TypeScript versions
interface GPU {
  requestAdapter(options?: GPURequestAdapterOptions): Promise<GPUAdapter | null>;
}

interface GPURequestAdapterOptions {
  powerPreference?: 'low-power' | 'high-performance';
  forceFallbackAdapter?: boolean;
}

interface GPUAdapter {
  requestDevice(descriptor?: GPUDeviceDescriptor): Promise<GPUDevice>;
}

interface GPUDeviceDescriptor {
  requiredFeatures?: Iterable<string>;
  requiredLimits?: Record<string, number>;
}

interface GPUDevice extends EventTarget {}

interface Navigator {
  readonly gpu?: GPU;
}

export {};
