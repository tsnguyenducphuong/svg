import { registerWebModule, NativeModule } from 'expo';

import { ExpoImageToSvgModuleEvents } from './ExpoImageToSvg.types';

class ExpoImageToSvgModule extends NativeModule<ExpoImageToSvgModuleEvents> {
  PI = Math.PI;
  async setValueAsync(value: string): Promise<void> {
    this.emit('onChange', { value });
  }
  hello() {
    return 'Hello world! 👋';
  }
}

export default registerWebModule(ExpoImageToSvgModule, 'ExpoImageToSvgModule');
