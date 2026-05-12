import { requireNativeModule } from 'expo-modules-core';
import { VectorizeOptions } from './ExpoImageToSvg.types';

// Fix: Declare the native JSI function in the global scope
declare global {
  var nativeVectorize: (options: any) => string;
}

// 1. Load the native module. 
// This triggers OnCreate/installJSIBindings on the native side.
requireNativeModule('ExpoImageToSvg');

/**
 * Vectorizes a bitmap buffer into an SVG string synchronously.
 * This utilizes a high-performance C++ engine via JSI.
 */
export function vectorize(options: VectorizeOptions): string {
  // Ensure the native side has been initialized
  const nativeFn = globalThis.nativeVectorize;

  if (!nativeFn) {
    throw new Error(
      '[ExpoImageToSvg] Native JSI function not found. Ensure the native module is linked correctly.'
    );
  }

  // Pass parameters directly to the C++ engine
  return globalThis.nativeVectorize({
    buffer: options.buffer,
    width: options.width,
    height: options.height,
    precision: options.precision ?? 6,
    threshold: options.threshold ?? 60,
    filterSpeckle: options.filterSpeckle ?? 4,
  });
} 
 
export { default } from './ExpoImageToSvgModule';
export { default as ExpoImageToSvgView } from './ExpoImageToSvgView';
export * from  './ExpoImageToSvg.types';

