import { NativeModule, requireNativeModule } from 'expo';

import { ExpoImageToSvgModuleEvents } from './ExpoImageToSvg.types';



declare class ExpoImageToSvgModule extends NativeModule<ExpoImageToSvgModuleEvents> {
  PI: number;
  hello(): string;
  setValueAsync(value: string): Promise<void>;
}

// This call loads the native module object from the JSI.
export default requireNativeModule<ExpoImageToSvgModule>('ExpoImageToSvg');
